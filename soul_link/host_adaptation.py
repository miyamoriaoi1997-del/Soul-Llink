from __future__ import annotations

import argparse
import json
import shlex
import shutil
import subprocess
import sys
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from pathlib import Path
from uuid import uuid4

import yaml

CommandRunner = Callable[[Sequence[str], Path], int]


@dataclass(frozen=True, slots=True)
class CompatibilityManifest:
    host: str
    adapter_version: str
    required_paths: tuple[str, ...]
    patch_path: Path
    created_paths: tuple[str, ...] = ()
    verify_commands: tuple[tuple[str, ...], ...] = ()

    def __post_init__(self) -> None:
        if not self.host.strip() or not self.adapter_version.strip():
            raise ValueError("host and adapter_version are required")
        patch_path = Path(self.patch_path)
        if not patch_path.is_absolute():
            raise ValueError("patch_path must be absolute")
        object.__setattr__(self, "patch_path", patch_path.resolve())
        required_paths = tuple(self.required_paths)
        created_paths = tuple(self.created_paths)
        for relative in required_paths + created_paths:
            normalized = str(relative).replace("\\", "/")
            path = Path(normalized)
            if (
                not normalized
                or normalized.startswith("/")
                or path.is_absolute()
                or path.anchor
                or path.drive
                or ".." in path.parts
            ):
                raise ValueError(f"unsafe host path: {relative!r}")
        if set(required_paths) & set(created_paths):
            raise ValueError("required_paths and created_paths overlap")
        object.__setattr__(self, "required_paths", required_paths)
        object.__setattr__(self, "created_paths", created_paths)
        object.__setattr__(self, "verify_commands", tuple(tuple(command) for command in self.verify_commands))

    @classmethod
    def load(cls, path: Path) -> "CompatibilityManifest":
        manifest_path = Path(path).resolve()
        data = yaml.safe_load(manifest_path.read_text(encoding="utf-8")) or {}
        patch_path = (manifest_path.parent / str(data.get("patch") or "")).resolve()
        raw_commands = data.get("verify_commands") or ()
        verify_commands = tuple(
            tuple(str(part) for part in command)
            if isinstance(command, list)
            else tuple(shlex.split(str(command), posix=False))
            for command in raw_commands
        )
        return cls(
            host=str(data.get("host") or ""),
            adapter_version=str(data.get("adapter_version") or ""),
            required_paths=tuple(str(item) for item in data.get("required_paths") or ()),
            patch_path=patch_path,
            created_paths=tuple(str(item) for item in data.get("created_paths") or ()),
            verify_commands=verify_commands,
        )


@dataclass(frozen=True, slots=True)
class CompatibilityResult:
    classification: str
    patch_state: str
    missing_paths: tuple[str, ...]


@dataclass(frozen=True, slots=True)
class AdaptationReceipt:
    host_root: Path
    backup_path: Path
    adapter_version: str

    def write(self, path: Path) -> None:
        destination = Path(path).resolve()
        destination.parent.mkdir(parents=True, exist_ok=True)
        temp = destination.with_name(destination.name + ".tmp")
        try:
            temp.write_text(
                json.dumps(
                    {
                        "host_root": str(self.host_root),
                        "backup_path": str(self.backup_path),
                        "adapter_version": self.adapter_version,
                    },
                    ensure_ascii=False,
                    indent=2,
                ),
                encoding="utf-8",
            )
            temp.replace(destination)
        except BaseException:
            temp.unlink(missing_ok=True)
            raise

    @classmethod
    def load(cls, path: Path) -> "AdaptationReceipt":
        data = json.loads(Path(path).read_text(encoding="utf-8"))
        return cls(
            host_root=Path(data["host_root"]).resolve(),
            backup_path=Path(data["backup_path"]).resolve(),
            adapter_version=str(data["adapter_version"]),
        )


class HostAdapterController:
    def __init__(
        self,
        manifest: CompatibilityManifest,
        *,
        command_runner: CommandRunner | None = None,
    ) -> None:
        self.manifest = manifest
        self._run = command_runner or self._run_command

    def detect(self, host_root: Path) -> CompatibilityResult:
        root = Path(host_root).resolve()
        targets = {relative: self._host_path(root, relative) for relative in self.manifest.required_paths}
        missing = tuple(relative for relative, target in targets.items() if not target.is_file())
        if missing:
            return CompatibilityResult("incompatible", "not_checked", missing)

        patch = str(self.manifest.patch_path)
        reverse = ("git", "apply", "--check", "--reverse", patch)
        if self._run(reverse, root) == 0:
            return CompatibilityResult("supported", "applied", ())

        forward = ("git", "apply", "--check", patch)
        if self._run(forward, root) == 0:
            return CompatibilityResult("transformable", "applicable", ())

        return CompatibilityResult("incompatible", "mismatch", ())

    def verify(self, host_root: Path) -> bool:
        root = Path(host_root).resolve()
        if self.detect(root).classification != "supported":
            return False
        return all(self._run(self._expand_command(command), root) == 0 for command in self.manifest.verify_commands)

    def apply(
        self,
        host_root: Path,
        *,
        verifier: Callable[[Path], bool],
    ) -> tuple[CompatibilityResult, AdaptationReceipt | None]:
        root = Path(host_root).resolve()
        detected = self.detect(root)
        if detected.classification == "supported":
            if not verifier(root):
                raise RuntimeError("host verification failed")
            return detected, None
        if detected.classification != "transformable":
            raise RuntimeError("host is incompatible with this adapter")

        backup = root / f".soullink-adapter-backup-{uuid4().hex}"
        try:
            backup.mkdir()
            (backup / ".soullink-backup.json").write_text(
                json.dumps({"host_root": str(root), "adapter_version": self.manifest.adapter_version}),
                encoding="utf-8",
            )
            for relative in self.manifest.required_paths:
                source = self._host_path(root, relative)
                destination = backup / relative
                destination.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(source, destination)

            command = ("git", "apply", str(self.manifest.patch_path))
            if self._run(command, root) != 0:
                raise RuntimeError("adapter patch application failed")
            if not verifier(root):
                raise RuntimeError("adapter verification failed")
            result = self.detect(root)
            if result.classification != "supported":
                raise RuntimeError("adapter verification failed: patch state is not applied")
            receipt = AdaptationReceipt(root, backup, self.manifest.adapter_version)
            return result, receipt
        except BaseException:
            self._restore_backup(root, backup)
            shutil.rmtree(backup, ignore_errors=True)
            raise

    def rollback(self, receipt: AdaptationReceipt) -> bool:
        root = Path(receipt.host_root).resolve()
        backup = Path(receipt.backup_path).resolve()
        if receipt.adapter_version != self.manifest.adapter_version:
            raise RuntimeError("adaptation receipt version does not match manifest")
        marker_path = backup / ".soullink-backup.json"
        if (
            not backup.is_dir()
            or backup.parent != root
            or not backup.name.startswith(".soullink-adapter-backup-")
            or not marker_path.is_file()
        ):
            raise RuntimeError("invalid or missing adaptation backup")
        try:
            marker = json.loads(marker_path.read_text(encoding="utf-8"))
        except (OSError, ValueError, TypeError) as exc:
            raise RuntimeError("invalid adaptation backup marker") from exc
        if marker != {"host_root": str(root), "adapter_version": self.manifest.adapter_version}:
            raise RuntimeError("adaptation backup marker mismatch")
        missing_or_unsafe = []
        for relative in self.manifest.required_paths:
            saved = backup / relative
            try:
                resolved = saved.resolve(strict=True)
                resolved.relative_to(backup)
            except (OSError, ValueError):
                missing_or_unsafe.append(relative)
                continue
            if saved.is_symlink() or not resolved.is_file():
                missing_or_unsafe.append(relative)
        if missing_or_unsafe:
            raise RuntimeError(f"incomplete or unsafe adaptation backup: {missing_or_unsafe}")
        self._restore_backup(root, backup)
        shutil.rmtree(backup, ignore_errors=True)
        return all((root / relative).is_file() for relative in self.manifest.required_paths)

    def _restore_backup(self, root: Path, backup: Path) -> None:
        for relative in self.manifest.created_paths:
            created = self._host_path(root, relative)
            if created.is_file() or created.is_symlink():
                created.unlink()
        for relative in self.manifest.required_paths:
            saved = backup / relative
            if saved.is_file():
                destination = self._host_path(root, relative)
                destination.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(saved, destination)

    @staticmethod
    def _host_path(root: Path, relative: str) -> Path:
        target = (root / relative).resolve()
        try:
            target.relative_to(root)
        except ValueError as exc:
            raise RuntimeError(f"host path escapes host root: {relative}") from exc
        return target

    @staticmethod
    def _expand_command(command: Sequence[str]) -> tuple[str, ...]:
        return tuple(sys.executable if part == "{python}" else part for part in command)

    @staticmethod
    def _run_command(command: Sequence[str], cwd: Path) -> int:
        completed = subprocess.run(
            list(command),
            cwd=cwd,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            check=False,
        )
        return completed.returncode


def _result_payload(result: CompatibilityResult) -> dict[str, object]:
    return {
        "classification": result.classification,
        "patch_state": result.patch_state,
        "missing_paths": list(result.missing_paths),
    }


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="soullink-host-adapt")
    subparsers = parser.add_subparsers(dest="action", required=True)
    for action in ("detect", "verify"):
        action_parser = subparsers.add_parser(action)
        action_parser.add_argument("--manifest", type=Path, required=True)
        action_parser.add_argument("--host-root", type=Path, required=True)
    apply_parser = subparsers.add_parser("apply")
    apply_parser.add_argument("--manifest", type=Path, required=True)
    apply_parser.add_argument("--host-root", type=Path, required=True)
    apply_parser.add_argument("--receipt", type=Path, required=True)
    rollback_parser = subparsers.add_parser("rollback")
    rollback_parser.add_argument("--manifest", type=Path, required=True)
    rollback_parser.add_argument("--receipt", type=Path, required=True)
    args = parser.parse_args(argv)

    controller = HostAdapterController(CompatibilityManifest.load(args.manifest))
    try:
        if args.action == "detect":
            result = controller.detect(args.host_root)
            print(json.dumps(_result_payload(result), ensure_ascii=False))
            return 0 if result.classification != "incompatible" else 2
        if args.action == "verify":
            verified = controller.verify(args.host_root)
            print(json.dumps({"verified": verified}, ensure_ascii=False))
            return 0 if verified else 3
        if args.action == "apply":
            result, receipt = controller.apply(args.host_root, verifier=controller.verify)
            if receipt is not None:
                try:
                    receipt.write(args.receipt)
                except BaseException:
                    controller.rollback(receipt)
                    raise
            print(json.dumps({**_result_payload(result), "receipt": str(args.receipt) if receipt else ""}, ensure_ascii=False))
            return 0
        receipt = AdaptationReceipt.load(args.receipt)
        rolled_back = controller.rollback(receipt)
        if rolled_back:
            args.receipt.unlink(missing_ok=True)
        print(json.dumps({"rolled_back": rolled_back}, ensure_ascii=False))
        return 0 if rolled_back else 4
    except Exception as exc:
        print(json.dumps({"success": False, "error": str(exc)}, ensure_ascii=False))
        return 5


if __name__ == "__main__":
    raise SystemExit(main())
