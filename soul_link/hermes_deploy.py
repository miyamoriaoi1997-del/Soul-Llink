from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from uuid import uuid4

import yaml


@dataclass(frozen=True, slots=True)
class DeploymentReceipt:
    host_root: Path
    hermes_home: Path
    soullink_root: Path
    backup_path: Path
    adapter_version: str = "2"

    def write(self, path: Path) -> None:
        path = Path(path).resolve()
        path.parent.mkdir(parents=True, exist_ok=True)
        temp = path.with_name(path.name + ".tmp")
        temp.write_text(json.dumps({
            "host_root": str(self.host_root),
            "hermes_home": str(self.hermes_home),
            "soullink_root": str(self.soullink_root),
            "backup_path": str(self.backup_path),
            "adapter_version": self.adapter_version,
        }, indent=2), encoding="utf-8")
        os.replace(temp, path)

    @classmethod
    def load(cls, path: Path) -> "DeploymentReceipt":
        data = json.loads(Path(path).read_text(encoding="utf-8"))
        return cls(**{key: Path(data[key]).resolve() for key in (
            "host_root", "hermes_home", "soullink_root", "backup_path"
        )}, adapter_version=str(data["adapter_version"]))


class HermesDeployment:
    adapter_version = "2"
    managed = ("plugins/soullink", "plugins/pcltm-context", "config.yaml", "SOUL.md")
    host_contract = {
        "agent/context_engine.py": "class ContextEngine",
        "agent/memory_provider.py": "class MemoryProvider",
        "plugins/memory/__init__.py": "load_memory_provider",
        "hermes_cli/plugins.py": "register_context_engine",
    }

    def __init__(self, soullink_root: Path) -> None:
        self.root = Path(soullink_root).resolve()
        self.asset = Path(__file__).resolve().parent / "hermes_assets"
        for required in (
            self.asset / "memory/__init__.py",
            self.asset / "memory/plugin.yaml",
            self.asset / "context/__init__.py",
            self.asset / "context/plugin.yaml",
            self.root / "packages/pcltm",
            self.root / "packages/persona_engine/soul_layers/SOUL.core.template.md",
        ):
            if not required.exists():
                raise RuntimeError(f"SoulLink deployment asset missing: {required}")

    def detect(self, host_root: Path, hermes_home: Path) -> dict[str, object]:
        host = Path(host_root).resolve()
        home = Path(hermes_home).resolve()
        missing: list[str] = []
        drifted: list[str] = []
        for relative, marker in self.host_contract.items():
            path = self._inside(host, relative)
            if not path.is_file():
                missing.append(relative)
            elif marker not in path.read_text(encoding="utf-8", errors="replace"):
                drifted.append(relative)
        installed = self._installed(home)
        return {
            "classification": "incompatible" if missing or drifted else (
                "supported" if installed else "transformable"
            ),
            "host_source_mutation_required": False,
            "missing_host_paths": missing,
            "missing_host_capabilities": drifted,
            "installed": installed,
        }

    def apply(self, host_root: Path, hermes_home: Path) -> DeploymentReceipt | None:
        host = Path(host_root).resolve()
        home = Path(hermes_home).resolve()
        state = self.detect(host, home)
        if state["classification"] == "incompatible":
            raise RuntimeError(f"Hermes host is incompatible: {state}")
        if state["classification"] == "supported" and self.verify(host, home):
            return None

        home.mkdir(parents=True, exist_ok=True)
        backup = home / f".soullink-deploy-backup-{uuid4().hex}"
        backup.mkdir()
        marker = {
            "host_root": str(host), "hermes_home": str(home),
            "soullink_root": str(self.root), "adapter_version": self.adapter_version,
            "entries": {}, "fingerprints": {},
        }
        mutation_started = False
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
            mutation_started = True
            self._install_plugin(home)
            self._install_config(home)
            self._install_soul(home)
            if not self.verify(host, home):
                raise RuntimeError("SoulLink verification failed")
            return DeploymentReceipt(host, home, self.root, backup)
        except BaseException:
            if mutation_started:
                self._validate_backup(backup, marker)
                self._restore(home, backup, marker)
            shutil.rmtree(backup, ignore_errors=not mutation_started)
            raise

    def verify(self, host_root: Path, hermes_home: Path) -> bool:
        host = Path(host_root).resolve()
        home = Path(hermes_home).resolve()
        if self.detect(host, home)["classification"] == "incompatible":
            return False
        if not self._installed(home):
            return False
        python = self._host_python(host)
        probe = (
            "from plugins.memory import load_memory_provider; "
            "p=load_memory_provider('soullink'); assert p and p.is_available(); "
            "from hermes_cli.plugins import discover_plugins,get_plugin_context_engine; "
            "discover_plugins(); e=get_plugin_context_engine(); "
            "assert e and e.name=='pcltm-context' and e.is_available(); "
            "print('SOULLINK_HERMES_PLUGIN_OK')"
        )
        env = os.environ.copy()
        env.update({
            "HERMES_HOME": str(home), "SOULLINK_ROOT": str(self.root),
            "PYTHONPATH": os.pathsep.join((str(host), str(self.root), str(self.root / "packages"))),
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
        if receipt.adapter_version != self.adapter_version:
            raise RuntimeError("deployment receipt version mismatch")
        home = receipt.hermes_home.resolve()
        backup = receipt.backup_path.resolve()
        if backup.parent != home or not backup.name.startswith(".soullink-deploy-backup-"):
            raise RuntimeError("invalid deployment backup path")
        marker_path = backup / ".soullink-deploy.json"
        if not marker_path.is_file():
            raise RuntimeError("deployment backup marker missing")
        marker = json.loads(marker_path.read_text(encoding="utf-8"))
        expected = {
            "host_root": str(receipt.host_root.resolve()),
            "hermes_home": str(home), "soullink_root": str(self.root),
            "adapter_version": self.adapter_version,
        }
        if any(marker.get(key) != value for key, value in expected.items()):
            raise RuntimeError("deployment backup marker mismatch")
        self._validate_backup(backup, marker)
        self._restore(home, backup, marker)
        shutil.rmtree(backup)
        return True

    def _installed(self, home: Path) -> bool:
        plugin = home / "plugins/soullink"
        context_plugin = home / "plugins/pcltm-context"
        config = self._read_yaml(home / "config.yaml")
        disabled = {
            str(name)
            for name in (((config.get("agent") or {}).get("disabled_toolsets")) or [])
        }
        return (
            (plugin / "__init__.py").is_file()
            and (plugin / "plugin.yaml").is_file()
            and (plugin / "soullink-root.txt").read_text(encoding="utf-8").strip() == str(self.root)
            and (context_plugin / "__init__.py").is_file()
            and (context_plugin / "plugin.yaml").is_file()
            and (context_plugin / "soullink-root.txt").read_text(encoding="utf-8").strip() == str(self.root)
            and (config.get("memory") or {}).get("provider") == "soullink"
            and (config.get("context") or {}).get("engine") == "pcltm-context"
            and "pcltm-context" in ((config.get("plugins") or {}).get("enabled") or [])
            and "session_search" in disabled
            and "memory" not in disabled
            and "context_engine" not in disabled
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
        config.setdefault("context", {})["engine"] = "pcltm-context"
        # Hermes gates every context-engine lifecycle behind this host switch.
        # With context.engine=pcltm-context, enabling it activates PCLTM; it
        # does not select the built-in ContextCompressor.
        config.setdefault("compression", {})["enabled"] = True
        plugins = config.setdefault("plugins", {})
        enabled = list(plugins.get("enabled") or [])
        if "pcltm-context" not in enabled:
            enabled.append("pcltm-context")
        plugins["enabled"] = enabled

        # PCLTM is the authoritative cross-session memory surface for a
        # SoulLink-managed profile.  Suppress Hermes transcript search so the
        # agent cannot silently bypass PCLTM provenance, while ensuring the
        # selected memory provider and context engine remain callable.  Keep
        # every unrelated user-disabled toolset unchanged.
        agent = config.setdefault("agent", {})
        disabled = [str(name) for name in (agent.get("disabled_toolsets") or [])]
        disabled = [name for name in disabled if name not in {"memory", "context_engine"}]
        if "session_search" not in disabled:
            disabled.append("session_search")
        agent["disabled_toolsets"] = disabled

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

    def _validate_backup(self, backup: Path, marker: dict) -> None:
        fingerprints = marker.get("fingerprints", {})
        for relative, existed in marker.get("entries", {}).items():
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
        target = (root / relative).resolve()
        try:
            target.relative_to(root.resolve())
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
