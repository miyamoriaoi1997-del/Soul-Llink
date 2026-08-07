from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import sys
import tomllib
from dataclasses import dataclass
from pathlib import Path, PureWindowsPath
from uuid import uuid4

BEGIN = "# BEGIN SOULLINK MANAGED CODEX ADAPTER"
END = "# END SOULLINK MANAGED CODEX ADAPTER"
ADAPTER_VERSION = "1"


@dataclass(frozen=True, slots=True)
class CodexDeploymentReceipt:
    codex_home: Path
    backup_path: Path
    receipt_path: Path
    adapter_version: str
    entries: dict[str, bool]
    fingerprints: dict[str, str]

    def write(self, path: Path) -> None:
        path = CodexDeployment._safe_path(path, allow_missing=True)
        path.parent.mkdir(parents=True, exist_ok=True)
        temp = path.with_name(path.name + ".tmp")
        try:
            temp.write_text(json.dumps({
                "codex_home": str(self.codex_home),
                "backup_path": str(self.backup_path),
                "receipt_path": str(self.receipt_path),
                "adapter_version": self.adapter_version,
                "entries": self.entries,
                "fingerprints": self.fingerprints,
            }, indent=2), encoding="utf-8")
            os.replace(temp, path)
        except BaseException:
            temp.unlink(missing_ok=True)
            raise

    @classmethod
    def load(cls, path: Path) -> "CodexDeploymentReceipt":
        path = CodexDeployment._safe_path(path)
        data = json.loads(path.read_text(encoding="utf-8"))
        return cls(
            codex_home=Path(data["codex_home"]).resolve(),
            backup_path=Path(data["backup_path"]).resolve(),
            # The caller-selected receipt file is authoritative. Never trust a
            # serialized path that could be edited to make rollback unlink an
            # unrelated file.
            receipt_path=path,
            adapter_version=str(data["adapter_version"]),
            entries=dict(data["entries"]),
            fingerprints=dict(data["fingerprints"]),
        )


class CodexDeployment:
    managed = ("config.toml", "hooks.json", "soullink")
    adapter_version = ADAPTER_VERSION

    def detect(self, codex_home: Path) -> dict[str, object]:
        try:
            home = self._safe_path(codex_home, allow_missing=True)
        except RuntimeError:
            return {"classification": "incompatible", "host_source_mutation_required": False, "installed": False, "blockers": ["unsafe_codex_home"], "codex_home": str(Path(codex_home).absolute())}
        blockers: list[str] = []
        unsafe: set[str] = set()
        for relative in self.managed:
            try:
                self._managed_path(home, relative)
            except RuntimeError:
                blockers.append(f"unsafe_managed_path:{relative}")
                unsafe.add(relative)
        config = home / "config.toml"
        if config.exists() and "config.toml" not in unsafe:
            try:
                text = config.read_text(encoding="utf-8-sig")
                parsed = tomllib.loads(text)
            except (OSError, UnicodeError, tomllib.TOMLDecodeError):
                blockers.append("invalid_config_toml")
            else:
                servers = parsed.get("mcp_servers", {})
                if not isinstance(servers, dict):
                    blockers.append("invalid_config_shape")
                else:
                    soullink = servers.get("soullink")
                    if soullink is not None and not isinstance(soullink, dict):
                        blockers.append("invalid_config_shape")
                    elif soullink and BEGIN not in text:
                        blockers.append("foreign_mcp_table")
                if (BEGIN in text) != (END in text):
                    blockers.append("damaged_managed_block")
        hooks = home / "hooks.json"
        if hooks.exists() and "hooks.json" not in unsafe:
            try:
                payload = json.loads(hooks.read_text(encoding="utf-8-sig"))
            except (OSError, json.JSONDecodeError):
                blockers.append("invalid_hooks_json")
            else:
                if not self._hooks_shape_valid(payload):
                    blockers.append("invalid_hooks_shape")
        installed = not blockers and self._installed(home)
        return {
            "classification": "incompatible" if blockers else ("supported" if installed else "transformable"),
            "host_source_mutation_required": False,
            "installed": installed,
            "blockers": blockers,
            "codex_home": str(home),
        }

    def apply(
        self,
        codex_home: Path,
        *,
        db_path: Path,
        memfs_root: Path,
        receipt_path: Path | None = None,
    ) -> CodexDeploymentReceipt | None:
        home = self._safe_path(codex_home, allow_missing=True)
        state = self.detect(home)
        if state["classification"] == "incompatible":
            raise RuntimeError(f"Codex home is incompatible: {state['blockers']}")
        if state["classification"] == "supported" and self.verify(home):
            return None
        receipt_path = self._safe_path(receipt_path or home / "soullink-deployment-receipt.json", allow_missing=True)
        if self._receipt_overlaps_managed_path(home, receipt_path):
            raise RuntimeError("receipt path overlaps managed runtime")
        if receipt_path.exists():
            raise RuntimeError("receipt path already exists; refusing to overwrite")
        home.mkdir(parents=True, exist_ok=True)
        baseline = {p.relative_to(home).as_posix() for p in home.rglob("*")}
        backup = home / f".soullink-codex-backup-{uuid4().hex}"
        backup.mkdir()
        marker: dict[str, object] = {
            "codex_home": str(home), "adapter_version": self.adapter_version,
            "receipt_path": str(receipt_path),
            "entries": {}, "fingerprints": {},
        }
        mutation_started = False
        try:
            for relative in self.managed:
                target = self._managed_path(home, relative)
                existed = target.exists()
                marker["entries"][relative] = existed  # type: ignore[index]
                if existed:
                    dest = backup / relative
                    dest.parent.mkdir(parents=True, exist_ok=True)
                    if target.is_dir():
                        shutil.copytree(target, dest)
                    else:
                        shutil.copy2(target, dest)
                    marker["fingerprints"][relative] = self._fingerprint(dest)  # type: ignore[index]
            self._write_marker(backup, marker)
            mutation_started = True
            self._install(home, Path(db_path).resolve(), Path(memfs_root).resolve())
            if not self.verify(home):
                raise RuntimeError("SoulLink Codex verification failed")
            self._record_created(home, baseline, marker)
            self._write_marker(backup, marker)
            receipt = CodexDeploymentReceipt(
                home, backup, receipt_path, self.adapter_version,
                dict(marker["entries"]), dict(marker["fingerprints"]),  # type: ignore[arg-type]
            )
            receipt.write(receipt_path)
            return receipt
        except BaseException:
            if mutation_started:
                self._record_created(home, baseline, marker)
                self._write_marker(backup, marker)
                self._validate_backup(backup, marker)
                self._restore(home, backup, marker)
                if receipt_path.exists():
                    receipt_path.unlink()
            shutil.rmtree(backup, ignore_errors=True)
            raise

    def verify(self, codex_home: Path) -> bool:
        try:
            home = self._safe_path(codex_home)
        except RuntimeError:
            return False
        if not self._installed(home):
            return False
        try:
            for relative in self.managed:
                self._managed_path(home, relative)
            config_text = (home / "config.toml").read_text(encoding="utf-8")
            config = tomllib.loads(config_text)
            hooks = json.loads((home / "hooks.json").read_text(encoding="utf-8"))
            adapter = json.loads((home / "soullink/adapter.json").read_text(encoding="utf-8"))
        except (OSError, UnicodeError, json.JSONDecodeError, tomllib.TOMLDecodeError):
            return False
        required = {"SessionStart", "UserPromptSubmit", "PreToolUse", "PostToolUse", "PreCompact", "PostCompact", "Stop"}
        server = config.get("mcp_servers", {}).get("soullink", {})
        managed_hooks = hooks.get("hooks", {}) if isinstance(hooks, dict) else {}
        command, command_windows = self._hook_commands()
        hooks_valid = isinstance(managed_hooks, dict) and all(
            self._event_hook_valid(managed_hooks.get(event), command, command_windows)
            for event in required
        )
        expected_server = {
            "command": sys.executable,
            "args": ["-m", "soul_link.codex_mcp"],
            "required": True,
            "startup_timeout_sec": 20,
            "tool_timeout_sec": 60,
            "enabled": True,
            "enabled_tools": [
                "soullink_memory_search", "soullink_memory_open", "soullink_memory_recall_exact",
                "soullink_identity_status", "soullink_runtime_status",
            ],
            "default_tools_approval_mode": "writes",
            "env": {
                "HERMES_PCLTM_DB": adapter.get("db_path"),
                "HERMES_PCLTM_MEMFS_ROOT": adapter.get("memfs_root"),
            },
        }
        return (
            BEGIN in config_text and END in config_text
            and server == expected_server
            and hooks_valid
            and adapter.get("adapter_version") == self.adapter_version
            and adapter.get("final_forward_observation") == "unavailable_host_boundary"
        )

    def rollback(self, receipt: CodexDeploymentReceipt) -> bool:
        if receipt.adapter_version != self.adapter_version:
            raise RuntimeError("deployment receipt version mismatch")
        home = self._safe_path(receipt.codex_home)
        backup = self._safe_path(receipt.backup_path)
        receipt_path = self._safe_path(receipt.receipt_path)
        if backup.parent != home or not backup.name.startswith(".soullink-codex-backup-"):
            raise RuntimeError("invalid deployment backup path")
        marker_path = backup / ".soullink-codex-deploy.json"
        if not marker_path.is_file():
            raise RuntimeError("deployment backup marker missing")
        marker = json.loads(marker_path.read_text(encoding="utf-8"))
        if marker.get("codex_home") != str(home) or marker.get("adapter_version") != self.adapter_version:
            raise RuntimeError("deployment backup marker mismatch")
        if marker.get("receipt_path") != str(receipt_path):
            raise RuntimeError("deployment receipt path does not match backup")
        if marker.get("entries") != receipt.entries or marker.get("fingerprints") != receipt.fingerprints:
            raise RuntimeError("deployment receipt does not match backup")
        self._validate_manifest(marker)
        self._validate_backup(backup, marker)
        self._restore(home, backup, marker)
        shutil.rmtree(backup)
        if receipt_path.exists():
            receipt_path.unlink()
        return True

    def _install(self, home: Path, db_path: Path, memfs_root: Path) -> None:
        runtime = home / "soullink"
        runtime.mkdir(parents=True, exist_ok=True)
        (runtime / "adapter.json").write_text(json.dumps({
            "adapter_version": self.adapter_version,
            "db_path": str(db_path), "memfs_root": str(memfs_root),
            "final_forward_observation": "unavailable_host_boundary",
        }, indent=2), encoding="utf-8")
        config_path = home / "config.toml"
        existing = config_path.read_text(encoding="utf-8-sig") if config_path.exists() else ""
        if BEGIN in existing:
            before, remainder = existing.split(BEGIN, 1)
            _, after = remainder.split(END, 1)
            existing = (before.rstrip() + "\n\n" + after.lstrip()).strip()
        block = self._config_block(db_path, memfs_root)
        config_path.write_text(existing.rstrip() + ("\n\n" if existing.strip() else "") + block, encoding="utf-8")
        hooks_path = home / "hooks.json"
        payload = json.loads(hooks_path.read_text(encoding="utf-8-sig")) if hooks_path.exists() else {"hooks": {}}
        groups = payload.setdefault("hooks", {})
        command, command_windows = self._hook_commands()
        for event in ("SessionStart", "UserPromptSubmit", "PreToolUse", "PostToolUse", "PreCompact", "PostCompact", "Stop"):
            current = groups.setdefault(event, [])
            current[:] = [g for g in current if g.get("description") != "SoulLink/PCLTM managed hook"]
            current.append({
                "description": "SoulLink/PCLTM managed hook",
                "hooks": [{"type": "command", "command": command, "commandWindows": command_windows, "timeout": 30}],
            })
        hooks_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")

    @staticmethod
    def _config_block(db_path: Path, memfs_root: Path) -> str:
        quote = lambda value: json.dumps(str(value), ensure_ascii=False)
        return "\n".join((
            BEGIN,
            "[mcp_servers.soullink]",
            f"command = {quote(sys.executable)}",
            'args = ["-m", "soul_link.codex_mcp"]',
            "required = true",
            "startup_timeout_sec = 20",
            "tool_timeout_sec = 60",
            "enabled = true",
            'enabled_tools = ["soullink_memory_search", "soullink_memory_open", "soullink_memory_recall_exact", "soullink_identity_status", "soullink_runtime_status"]',
            "default_tools_approval_mode = \"writes\"",
            "[mcp_servers.soullink.env]",
            f"HERMES_PCLTM_DB = {quote(db_path)}",
            f"HERMES_PCLTM_MEMFS_ROOT = {quote(memfs_root)}",
            END,
            "",
        ))

    @staticmethod
    def _installed(home: Path) -> bool:
        config = home / "config.toml"
        return config.is_file() and BEGIN in config.read_text(encoding="utf-8-sig") and (home / "hooks.json").is_file() and (home / "soullink/adapter.json").is_file()

    @staticmethod
    def _fingerprint(path: Path) -> str:
        digest = hashlib.sha256()
        if path.is_file():
            digest.update(path.read_bytes())
        else:
            for item in sorted(p for p in path.rglob("*") if p.is_file()):
                digest.update(item.relative_to(path).as_posix().encode())
                digest.update(b"\0")
                digest.update(item.read_bytes())
        return digest.hexdigest()

    def _validate_backup(self, backup: Path, marker: dict[str, object]) -> None:
        self._validate_manifest(marker)
        for relative, existed in dict(marker["entries"]).items():  # type: ignore[arg-type]
            if existed:
                item = self._managed_path(backup, relative)
                if not item.exists() or self._fingerprint(item) != dict(marker["fingerprints"])[relative]:  # type: ignore[arg-type]
                    raise RuntimeError(f"deployment backup is incomplete or tampered: {relative}")

    def _restore(self, home: Path, backup: Path, marker: dict[str, object]) -> None:
        for relative, existed in dict(marker["entries"]).items():  # type: ignore[arg-type]
            target = self._managed_path(home, relative)
            if target.is_dir(): shutil.rmtree(target)
            elif target.exists(): target.unlink()
            if existed:
                source = self._managed_path(backup, relative)
                target.parent.mkdir(parents=True, exist_ok=True)
                if source.is_dir(): shutil.copytree(source, target)
                else: shutil.copy2(source, target)

    def _record_created(self, home: Path, baseline: set[str], marker: dict[str, object]) -> None:
        entries = marker["entries"]  # type: ignore[assignment]
        for relative in self.managed:
            if relative not in entries:
                entries[relative] = relative in baseline

    @staticmethod
    def _write_marker(backup: Path, marker: dict[str, object]) -> None:
        (backup / ".soullink-codex-deploy.json").write_text(json.dumps(marker, indent=2), encoding="utf-8")

    @staticmethod
    def _inside(root: Path, value: str | Path) -> Path:
        raw = str(value)
        foreign = PureWindowsPath(raw)
        if foreign.is_absolute() and not Path(raw).is_absolute():
            raise RuntimeError("foreign absolute path rejected")
        root = root.absolute()
        candidate = (root / value).absolute() if not Path(value).is_absolute() else Path(value).absolute()
        try:
            if os.path.commonpath((str(root), str(candidate))) != str(root):
                raise RuntimeError("path escapes managed root")
        except ValueError as exc:
            raise RuntimeError("path escapes managed root") from exc
        return candidate

    @classmethod
    def _managed_path(cls, root: Path, value: str | Path) -> Path:
        path = cls._inside(root, value)
        current = root.absolute()
        relative = path.relative_to(current)
        for part in relative.parts:
            current = current / part
            if not current.exists() and not current.is_symlink():
                continue
            stat = current.lstat()
            is_reparse = bool(getattr(stat, "st_file_attributes", 0) & 0x400)
            if current.is_symlink() or is_reparse:
                raise RuntimeError(f"symlink or reparse point rejected: {current}")
        return path

    @staticmethod
    def _safe_path(value: str | Path, *, allow_missing: bool = False) -> Path:
        path = Path(value).absolute()
        current = Path(path.anchor)
        for part in path.parts[1:]:
            current /= part
            if not current.exists() and not current.is_symlink():
                if allow_missing:
                    continue
                raise RuntimeError(f"managed path is missing: {current}")
            stat = current.lstat()
            if current.is_symlink() or bool(getattr(stat, "st_file_attributes", 0) & 0x400):
                raise RuntimeError(f"symlink or reparse point rejected: {current}")
        return path

    @classmethod
    def _validate_manifest(cls, marker: dict[str, object]) -> None:
        entries = marker.get("entries")
        fingerprints = marker.get("fingerprints")
        if not isinstance(entries, dict) or set(entries) != set(cls.managed):
            raise RuntimeError("deployment marker has invalid managed entries")
        if any(type(value) is not bool for value in entries.values()):
            raise RuntimeError("deployment marker has invalid managed entries")
        expected = {name for name, existed in entries.items() if existed}
        if not isinstance(fingerprints, dict) or set(fingerprints) != expected:
            raise RuntimeError("deployment marker has invalid fingerprints")
        if any(not isinstance(value, str) or len(value) != 64 for value in fingerprints.values()):
            raise RuntimeError("deployment marker has invalid fingerprints")

    @staticmethod
    def _event_hook_valid(groups: object, command: str, command_windows: str) -> bool:
        if not isinstance(groups, list):
            return False
        for group in groups:
            if not isinstance(group, dict) or group.get("description") != "SoulLink/PCLTM managed hook":
                continue
            handlers = group.get("hooks")
            if not isinstance(handlers, list):
                continue
            if any(
                isinstance(handler, dict)
                and handler.get("type") == "command"
                and handler.get("command") == command
                and handler.get("commandWindows") == command_windows
                and handler.get("timeout") == 30
                for handler in handlers
            ):
                return True
        return False

    @staticmethod
    def _hooks_shape_valid(payload: object) -> bool:
        if not isinstance(payload, dict):
            return False
        if "hooks" not in payload:
            return False
        groups = payload["hooks"]
        if not isinstance(groups, dict):
            return False
        return all(
            isinstance(event_groups, list)
            and all(
                isinstance(group, dict)
                and "hooks" in group
                and isinstance(group["hooks"], list)
                and all(isinstance(handler, dict) for handler in group["hooks"])
                for group in event_groups
            )
            for event_groups in groups.values()
        )

    @classmethod
    def _receipt_overlaps_managed_path(cls, home: Path, receipt_path: Path) -> bool:
        receipt = receipt_path.absolute()
        for relative in cls.managed:
            managed = (home / relative).absolute()
            if receipt == managed:
                return True
            if relative == "soullink":
                try:
                    receipt.relative_to(managed)
                except ValueError:
                    pass
                else:
                    return True
        return False

    @staticmethod
    def _hook_commands() -> tuple[str, str]:
        command = f'"{sys.executable}" -m soul_link.codex_hook'
        hook_executable = Path(sys.executable).with_name("soullink-codex-hook.exe")
        command_windows = f'"{hook_executable}"' if hook_executable.is_file() else command
        return command, command_windows


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="soullink-codex-deploy")
    sub = parser.add_subparsers(dest="command", required=True)
    for name in ("detect", "apply", "verify"):
        cmd = sub.add_parser(name)
        cmd.add_argument("--codex-home", default=os.getenv("CODEX_HOME", str(Path.home() / ".codex")))
        if name == "apply":
            cmd.add_argument("--db", required=True)
            cmd.add_argument("--memfs", required=True)
            cmd.add_argument("--receipt")
    rollback = sub.add_parser("rollback")
    rollback.add_argument("--receipt", required=True)
    args = parser.parse_args(argv)
    deployment = CodexDeployment()
    if args.command == "detect": result = deployment.detect(Path(args.codex_home))
    elif args.command == "verify": result = {"verified": deployment.verify(Path(args.codex_home))}
    elif args.command == "apply":
        receipt_path = Path(args.receipt) if args.receipt else None
        receipt = deployment.apply(Path(args.codex_home), db_path=Path(args.db), memfs_root=Path(args.memfs), receipt_path=receipt_path)
        result = {"applied": receipt is not None, "receipt": str(receipt.receipt_path if receipt is not None else receipt_path or Path(args.codex_home) / "soullink-deployment-receipt.json")}
    else:
        receipt = CodexDeploymentReceipt.load(Path(args.receipt))
        result = {"rolled_back": deployment.rollback(receipt)}
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
