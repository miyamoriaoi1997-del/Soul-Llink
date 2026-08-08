from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path, PureWindowsPath
from uuid import uuid4

import yaml

from soul_link.host_adaptation import (
    AdaptationReceipt,
    CompatibilityManifest,
    HostAdapterController,
    _reject_reparse_path,
)


@dataclass(frozen=True, slots=True)
class DeploymentReceipt:
    host_root: Path
    hermes_home: Path
    soullink_root: Path
    backup_path: Path
    host_adaptation_receipt: Path | None = None
    adapter_version: str = "3"
    fingerprints: dict[str, str] | None = None
    entries: dict[str, bool] | None = None

    def write(self, path: Path) -> None:
        path = Path(path).resolve()
        path.parent.mkdir(parents=True, exist_ok=True)
        temp = path.with_name(path.name + ".tmp")
        temp.write_text(json.dumps({
            "host_root": str(self.host_root),
            "hermes_home": str(self.hermes_home),
            "soullink_root": str(self.soullink_root),
            "backup_path": str(self.backup_path),
            "host_adaptation_receipt": (
                str(self.host_adaptation_receipt) if self.host_adaptation_receipt else ""
            ),
            "adapter_version": self.adapter_version,
            "fingerprints": self.fingerprints,
            "entries": self.entries,
        }, indent=2), encoding="utf-8")
        os.replace(temp, path)

    @classmethod
    def load(cls, path: Path) -> "DeploymentReceipt":
        data = json.loads(Path(path).read_text(encoding="utf-8"))
        paths = {
            key: Path(os.path.abspath(os.fspath(data[key])))
            for key in ("host_root", "hermes_home", "soullink_root", "backup_path")
        }
        raw_host_receipt = str(data.get("host_adaptation_receipt") or "").strip()
        return cls(
            **paths,
            host_adaptation_receipt=(
                Path(os.path.abspath(raw_host_receipt)) if raw_host_receipt else None
            ),
            adapter_version=str(data["adapter_version"]),
            fingerprints=data.get("fingerprints"),
            entries=data.get("entries"),
        )


PCLTM_CONTEXT_BUDGET_TOKENS = 200_000


class HermesDeployment:
    adapter_version = "3"
    rollback_compatible_versions = frozenset({"2", "3"})
    host_manifest_relative = Path("adapters/hermes/compatibility-soullink-runtime.yaml")
    managed = ("plugins/soullink", "plugins/pcltm-context", "config.yaml", "SOUL.md")
    host_contract = {
        "agent/context_engine.py": "class ContextEngine",
        "agent/memory_provider.py": "class MemoryProvider",
        "plugins/memory/__init__.py": "load_memory_provider",
        "hermes_cli/plugins.py": "register_context_engine",
    }

    def __init__(self, soullink_root: Path) -> None:
        self.root = _reject_reparse_path(soullink_root, label="SoulLink root")
        self.asset = Path(__file__).resolve().parent / "hermes_assets"
        for required in (
            self.asset / "memory/__init__.py",
            self.asset / "memory/plugin.yaml",
            self.asset / "context/__init__.py",
            self.asset / "context/plugin.yaml",
            self.root / "packages/pcltm",
            self.root / "packages/persona_engine/soul_layers/SOUL.core.template.md",
            self.root / self.host_manifest_relative,
        ):
            if not required.exists():
                raise RuntimeError(f"SoulLink deployment asset missing: {required}")

    def detect(self, host_root: Path, hermes_home: Path) -> dict[str, object]:
        try:
            host = _reject_reparse_path(host_root, label="host root")
        except FileNotFoundError:
            missing = list(self.host_contract)
            return {
                "classification": "incompatible",
                "runtime_health": "unavailable",
                "authority_health": "degraded",
                "host_source_mutation_required": False,
                "host_adaptation": {
                    "classification": "incompatible",
                    "patch_state": "not_checked",
                    "missing_paths": missing,
                },
                "missing_host_paths": missing,
                "missing_host_capabilities": [],
                "installed": False,
            }
        home = _reject_reparse_path(hermes_home, label="Hermes home", allow_missing_leaf=True)
        host_adaptation = self._host_controller().detect(host)
        missing: list[str] = []
        drifted: list[str] = []
        for relative, marker in self.host_contract.items():
            path = self._inside(host, relative)
            if not path.is_file():
                missing.append(relative)
            elif marker not in path.read_text(encoding="utf-8", errors="replace"):
                drifted.append(relative)
        installed = self._installed(home)
        host_incompatible = bool(missing or drifted)
        authority_healthy = host_adaptation.classification == "supported"
        runtime_healthy = installed and not host_incompatible
        if host_incompatible:
            classification = "incompatible"
        elif installed:
            classification = "supported" if authority_healthy else "degraded"
        elif host_adaptation.classification in {"supported", "transformable"}:
            classification = "transformable"
        else:
            classification = "incompatible"
        return {
            "classification": classification,
            "runtime_health": "healthy" if runtime_healthy else "unavailable",
            "authority_health": "healthy" if authority_healthy else "degraded",
            "host_source_mutation_required": host_adaptation.classification == "transformable",
            "host_adaptation": {
                "classification": host_adaptation.classification,
                "patch_state": host_adaptation.patch_state,
                "missing_paths": list(host_adaptation.missing_paths),
            },
            "missing_host_paths": missing,
            "missing_host_capabilities": drifted,
            "installed": installed,
        }

    def apply(self, host_root: Path, hermes_home: Path) -> DeploymentReceipt | None:
        host = _reject_reparse_path(host_root, label="host root")
        home = _reject_reparse_path(hermes_home, label="Hermes home", allow_missing_leaf=True)
        state = self.detect(host, home)
        if state["classification"] == "incompatible":
            raise RuntimeError(f"Hermes host is incompatible: {state}")
        if state["classification"] == "supported" and self.verify(host, home):
            return None

        home.mkdir(parents=True, exist_ok=True)
        baseline_paths = {path.relative_to(home).as_posix() for path in home.rglob("*")}
        backup = home / f".soullink-deploy-backup-{uuid4().hex}"
        backup.mkdir()
        host_receipt: AdaptationReceipt | None = None
        host_receipt_path = backup / "host-adaptation-receipt.json"
        marker = {
            "host_root": str(host), "hermes_home": str(home),
            "soullink_root": str(self.root), "adapter_version": self.adapter_version,
            "entries": {}, "fingerprints": {},
            "root_entries": sorted(path.name for path in home.iterdir()),
        }
        try:
            for relative in self.managed:
                source = self._inside(home, relative)
                self._reject_symlinks(source)
                marker["entries"][relative] = source.exists()
                if source.exists():
                    destination = self._inside(backup, relative)
                    destination.parent.mkdir(parents=True, exist_ok=True)
                    if source.is_dir():
                        shutil.copytree(source, destination)
                    else:
                        shutil.copy2(source, destination)
                    marker["fingerprints"][relative] = self._fingerprint(destination)
            (backup / ".soullink-deploy.json").write_text(
                json.dumps(marker, indent=2), encoding="utf-8"
            )

            host_controller = self._host_controller()
            _, host_receipt = host_controller.apply(
                host, verifier=host_controller.verify, backup_root=backup
            )
            if host_receipt is not None:
                host_receipt.write(host_receipt_path)

            self._install_plugin(home)
            self._install_config(home)
            self._install_soul(home)
            if not self.verify(host, home):
                raise RuntimeError("SoulLink verification failed")
            self._record_created_paths(home, backup, baseline_paths, marker)
            (backup / ".soullink-deploy.json").write_text(
                json.dumps(marker, indent=2), encoding="utf-8"
            )
            return DeploymentReceipt(
                host,
                home,
                self.root,
                backup,
                host_receipt_path if host_receipt is not None else None,
                fingerprints=dict(marker["fingerprints"]),
                entries=dict(marker["entries"]),
            )
        except BaseException:
            self._record_created_paths(home, backup, baseline_paths, marker)
            (backup / ".soullink-deploy.json").write_text(
                json.dumps(marker, indent=2), encoding="utf-8"
            )
            self._validate_backup(backup, marker)
            self._restore(home, backup, marker)
            if host_receipt is not None:
                self._host_controller().rollback(host_receipt, trusted_backup_root=backup)
            shutil.rmtree(backup, ignore_errors=True)
            raise

    def verify(self, host_root: Path, hermes_home: Path) -> bool:
        host = _reject_reparse_path(host_root, label="host root")
        home = _reject_reparse_path(hermes_home, label="Hermes home")
        if self.detect(host, home)["classification"] == "incompatible":
            return False
        if not self._host_controller().verify(host):
            return False
        if not self._installed(home):
            return False
        python = self._host_python(host)
        probe = (
            "from plugins.memory import load_memory_provider; "
            "p=load_memory_provider('soullink'); "
            "assert p and p.is_available(), 'soullink memory provider unavailable'; "
            "from hermes_cli.plugins import discover_plugins,get_plugin_context_engine; "
            "discover_plugins(); e=get_plugin_context_engine(); "
            "assert e and e.name=='pcltm-context' and e.is_available(), "
            "'pcltm context engine unavailable'; "
            "print('SOULLINK_HERMES_PLUGIN_OK')"
        )
        env = os.environ.copy()
        env.update({
            "HERMES_HOME": str(home), "SOULLINK_ROOT": str(self.root),
            "PYTHONPATH": os.pathsep.join((str(host), str(self.root), str(self.root / "packages"))),
            "PYTHONDONTWRITEBYTECODE": "1",
        })
        try:
            result = subprocess.run(
                [str(python), "-c", probe], cwd=host, env=env,
                text=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                check=False, timeout=60,
            )
        except subprocess.TimeoutExpired as exc:
            raise RuntimeError("Hermes plugin verification timed out") from exc
        if result.returncode != 0 or "SOULLINK_HERMES_PLUGIN_OK" not in result.stdout:
            detail = (result.stderr or result.stdout).strip()[-4000:]
            raise RuntimeError(f"Hermes plugin verification failed: {detail}")
        return True

    def rollback(self, receipt: DeploymentReceipt) -> bool:
        if receipt.adapter_version not in self.rollback_compatible_versions:
            raise RuntimeError("deployment receipt version mismatch")
        home = _reject_reparse_path(receipt.hermes_home, label="Hermes home")
        backup = _reject_reparse_path(receipt.backup_path, label="deployment backup")
        if backup.parent != home or not backup.name.startswith(".soullink-deploy-backup-"):
            raise RuntimeError("invalid deployment backup path")
        marker_path = _reject_reparse_path(
            backup / ".soullink-deploy.json", label="deployment backup marker"
        )
        if not marker_path.is_file():
            raise RuntimeError("deployment backup marker missing")
        marker = json.loads(marker_path.read_text(encoding="utf-8"))
        entries = marker.get("entries", {})
        if receipt.adapter_version == "2" and set(entries) != set(self.managed):
            raise RuntimeError("legacy deployment backup incomplete")
        expected = {
            "host_root": str(_reject_reparse_path(receipt.host_root, label="host root")),
            "hermes_home": str(home), "soullink_root": str(self.root),
            "adapter_version": receipt.adapter_version,
        }
        if any(marker.get(key) != value for key, value in expected.items()):
            raise RuntimeError("deployment backup marker mismatch")
        if receipt.adapter_version == "2":
            for relative, existed in marker.get("entries", {}).items():
                if existed and not self._inside(backup, relative).exists():
                    raise RuntimeError(f"legacy deployment backup incomplete: {relative}")
        else:
            self._validate_backup(backup, marker)
            if not isinstance(receipt.fingerprints, dict) or not isinstance(receipt.entries, dict):
                raise RuntimeError("deployment receipt manifest missing")
            if receipt.fingerprints != marker.get("fingerprints", {}):
                raise RuntimeError("deployment backup fingerprint mismatch between receipt and marker")
            if receipt.entries != marker.get("entries", {}):
                raise RuntimeError("deployment backup entries mismatch between receipt and marker")
        host_receipt: AdaptationReceipt | None = None
        if receipt.host_adaptation_receipt is not None:
            host_receipt_path = _reject_reparse_path(
                receipt.host_adaptation_receipt, label="host adaptation receipt"
            )
            try:
                host_receipt_path.relative_to(backup)
            except ValueError as exc:
                raise RuntimeError("host adaptation receipt escapes deployment backup") from exc
            if not host_receipt_path.is_file():
                raise RuntimeError("host adaptation receipt missing")
            host_receipt = AdaptationReceipt.load(host_receipt_path)

        self._restore(home, backup, marker)
        if host_receipt is not None:
            self._host_controller().rollback(host_receipt, trusted_backup_root=backup)
        shutil.rmtree(backup)
        return True

    def _host_controller(self) -> HostAdapterController:
        manifest = CompatibilityManifest.load(self.root / self.host_manifest_relative)
        return HostAdapterController(manifest)

    def _installed(self, home: Path) -> bool:
        plugin = home / "plugins/soullink"
        context_plugin = home / "plugins/pcltm-context"
        config = self._read_yaml(home / "config.yaml")
        return (
            (plugin / "__init__.py").is_file()
            and (plugin / "plugin.yaml").is_file()
            and (plugin / "soullink-root.txt").read_text(encoding="utf-8").strip() == str(self.root)
            and (context_plugin / "__init__.py").is_file()
            and (context_plugin / "plugin.yaml").is_file()
            and (context_plugin / "soullink-root.txt").read_text(encoding="utf-8").strip() == str(self.root)
            and (config.get("memory") or {}).get("provider") == "soullink"
            and (config.get("context") or {}).get("engine") == "pcltm-context"
            and (config.get("context") or {}).get("budget_tokens") == PCLTM_CONTEXT_BUDGET_TOKENS
            and (config.get("compression") or {}).get("threshold_tokens") == PCLTM_CONTEXT_BUDGET_TOKENS
            and "pcltm-context" in ((config.get("plugins") or {}).get("enabled") or [])
            and "managed-by: SoulLink/PCLTM" in (home / "SOUL.md").read_text(encoding="utf-8")
        ) if (plugin / "soullink-root.txt").is_file() and (home / "SOUL.md").is_file() else False

    def _install_plugin(self, home: Path) -> None:
        for source_name, plugin_name in (("memory", "soullink"), ("context", "pcltm-context")):
            target = home / "plugins" / plugin_name
            if target.exists():
                shutil.rmtree(target)
            shutil.copytree(self.asset / source_name, target)
            (target / "soullink-root.txt").write_text(str(self.root), encoding="utf-8")

    def _install_config(self, home: Path) -> None:
        path = home / "config.yaml"
        config = self._read_yaml(path)
        config.setdefault("memory", {})["provider"] = "soullink"
        context = config.setdefault("context", {})
        context["engine"] = "pcltm-context"
        # One public deployment policy controls both the PCLTM engine budget
        # and the host compression trigger.  The host patch passes context
        # configuration to plugin engines before model metadata is resolved.
        context["budget_tokens"] = PCLTM_CONTEXT_BUDGET_TOKENS
        compression = config.setdefault("compression", {})
        compression["threshold_tokens"] = PCLTM_CONTEXT_BUDGET_TOKENS
        # Hermes gates every context-engine lifecycle behind this host switch.
        # With context.engine=pcltm-context, enabling it activates PCLTM; it
        # does not select the built-in ContextCompressor.
        compression["enabled"] = True
        plugins = config.setdefault("plugins", {})
        enabled = list(plugins.get("enabled") or [])
        if "pcltm-context" not in enabled:
            enabled.append("pcltm-context")
        plugins["enabled"] = enabled
        self._atomic_yaml(path, config)

    def _install_soul(self, home: Path) -> None:
        import hashlib

        core_path = self.root / "packages/persona_engine/soul_layers/SOUL.core.template.md"
        core = core_path.read_text(encoding="utf-8").strip()
        if not core:
            raise RuntimeError(f"SoulLink core identity is empty: {core_path}")
        digest = hashlib.sha256(core.encode("utf-8")).hexdigest()[:16]
        text = (
            "<!-- managed-by: SoulLink/PCLTM; do not replace with Hermes default identity -->\n"
            "<!-- source: SoulLink packages/persona_engine/soul_layers/SOUL.core.template.md -->\n"
            f"<!-- core-sha256-16: {digest} -->\n\n"
            "# SoulLink Active Identity Anchor\n\n"
            "SoulLink/PCLTM owns persona injection for this Hermes profile. "
            "Hermes is the host/runtime/tool carrier, not the persona identity source.\n\n"
            "---\n\n"
            f"{core}\n"
        )
        self._atomic_text(home / "SOUL.md", text)

    def _restore(self, home: Path, backup: Path, marker: dict) -> None:
        original_roots = marker.get("root_entries")
        if isinstance(original_roots, list):
            keep = {str(name) for name in original_roots}
            keep.add(backup.name)
            for path in list(home.iterdir()):
                if path.name in keep:
                    continue
                if path.is_dir():
                    shutil.rmtree(path)
                else:
                    path.unlink()
        for relative, existed in marker.get("entries", {}).items():
            target = self._inside(home, relative)
            if target.is_dir():
                shutil.rmtree(target)
            elif target.exists() or target.is_symlink():
                target.unlink()
            if existed:
                saved = self._inside(backup, relative)
                target.parent.mkdir(parents=True, exist_ok=True)
                if saved.is_dir():
                    shutil.copytree(saved, target)
                else:
                    shutil.copy2(saved, target)

    def _record_created_paths(
        self, home: Path, backup: Path, baseline_paths: set[str], marker: dict
    ) -> None:
        managed = tuple(Path(relative) for relative in self.managed)
        backup_relative = backup.relative_to(home)
        recorded: list[Path] = []
        for path in sorted(home.rglob("*"), key=lambda item: len(item.relative_to(home).parts)):
            relative = path.relative_to(home)
            if relative.as_posix() in baseline_paths:
                continue
            if relative == backup_relative or backup_relative in relative.parents:
                continue
            if any(relative == item or item in relative.parents for item in managed):
                continue
            if any(parent == relative or parent in relative.parents for parent in recorded):
                continue
            marker["entries"][relative.as_posix()] = False
            recorded.append(relative)

    def _validate_backup(self, backup: Path, marker: dict) -> None:
        entries = marker.get("entries")
        fingerprints = marker.get("fingerprints")
        if not isinstance(entries, dict) or not isinstance(fingerprints, dict):
            raise RuntimeError("deployment backup incomplete: manifest missing")
        if not set(entries).issuperset(self.managed):
            raise RuntimeError("deployment backup entries mismatch")
        for relative, existed in entries.items():
            saved = self._inside(backup, relative)
            if existed and not saved.exists():
                raise RuntimeError(f"deployment backup incomplete: {relative}")
            if existed and fingerprints.get(relative) != self._fingerprint(saved):
                raise RuntimeError(f"deployment backup fingerprint mismatch: {relative}")

    @staticmethod
    def _reject_symlinks(path: Path) -> None:
        if path.is_symlink():
            raise RuntimeError(f"managed path may not be a symlink: {path}")
        if path.is_dir():
            for child in path.rglob("*"):
                if child.is_symlink():
                    raise RuntimeError(f"managed path contains a symlink: {child}")

    @staticmethod
    def _fingerprint(path: Path) -> str:
        digest = hashlib.sha256()
        if path.is_file():
            digest.update(b"F\0")
            digest.update(path.read_bytes())
            return digest.hexdigest()
        digest.update(b"D\0")
        for child in sorted(path.rglob("*"), key=lambda item: item.relative_to(path).as_posix()):
            digest.update(child.relative_to(path).as_posix().encode("utf-8") + b"\0")
            if child.is_file():
                digest.update(child.read_bytes())
        return digest.hexdigest()

    @staticmethod
    def _host_python(host: Path) -> Path:
        choices = (host / "venv/bin/python", host / "venv/Scripts/python.exe", Path(sys.executable))
        return next((path for path in choices if path.is_file()), Path(sys.executable))

    @staticmethod
    def _read_yaml(path: Path) -> dict:
        if not path.is_file():
            return {}
        data = yaml.safe_load(path.read_text(encoding="utf-8-sig")) or {}
        if not isinstance(data, dict):
            raise RuntimeError(f"Hermes config must be a mapping: {path}")
        return data

    @staticmethod
    def _atomic_yaml(path: Path, data: dict) -> None:
        HermesDeployment._atomic_text(path, yaml.safe_dump(data, sort_keys=False, allow_unicode=True))

    @staticmethod
    def _atomic_text(path: Path, text: str) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        temp = path.with_name(path.name + ".soullink.tmp")
        temp.write_text(text, encoding="utf-8")
        os.replace(temp, path)

    @staticmethod
    def _inside(root: Path, relative: str) -> Path:
        if not relative:
            raise RuntimeError("unsafe managed path: empty path")
        path = Path(relative)
        windows_path = PureWindowsPath(relative)
        if (
            path.is_absolute() or path.anchor or path.drive
            or windows_path.is_absolute() or windows_path.anchor or windows_path.drive
            or ".." in path.parts or ".." in windows_path.parts
        ):
            raise RuntimeError(f"unsafe managed path: {relative}")
        safe_root = _reject_reparse_path(root, label="managed root", allow_missing_leaf=True)
        target = _reject_reparse_path(
            safe_root / relative, label="managed path", allow_missing_leaf=True
        )
        try:
            target.relative_to(safe_root)
        except ValueError as exc:
            raise RuntimeError(f"managed path escapes root: {relative}") from exc
        return target


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="soullink-hermes-deploy")
    parser.add_argument("action", choices=("detect", "apply", "verify", "rollback"))
    parser.add_argument("--soullink-root", type=Path, required=True)
    parser.add_argument("--host-root", type=Path)
    parser.add_argument("--hermes-home", type=Path)
    parser.add_argument("--receipt", type=Path)
    args = parser.parse_args(argv)
    deployment = HermesDeployment(args.soullink_root)
    try:
        if args.action == "rollback":
            if not args.receipt:
                raise RuntimeError("--receipt is required")
            ok = deployment.rollback(DeploymentReceipt.load(args.receipt))
            if ok:
                args.receipt.unlink(missing_ok=True)
            payload = {"rolled_back": ok}
        else:
            if not args.host_root or not args.hermes_home:
                raise RuntimeError("--host-root and --hermes-home are required")
            if args.action == "detect":
                payload = deployment.detect(args.host_root, args.hermes_home)
            elif args.action == "verify":
                payload = {"verified": deployment.verify(args.host_root, args.hermes_home)}
            else:
                receipt = deployment.apply(args.host_root, args.hermes_home)
                if receipt and args.receipt:
                    try:
                        receipt.write(args.receipt)
                    except BaseException:
                        deployment.rollback(receipt)
                        raise
                payload = {**deployment.detect(args.host_root, args.hermes_home), "receipt": str(args.receipt or "")}
        print(json.dumps(payload, ensure_ascii=False))
        return 0 if payload.get("classification") != "incompatible" and payload.get("verified", True) else 2
    except Exception as exc:
        print(json.dumps({"success": False, "error": str(exc)}, ensure_ascii=False))
        return 5


if __name__ == "__main__":
    raise SystemExit(main())
