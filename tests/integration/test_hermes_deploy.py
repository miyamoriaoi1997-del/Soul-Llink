from __future__ import annotations

import hashlib
import json
import shutil
from pathlib import Path

import pytest
import yaml

from soul_link.hermes_deploy import DeploymentReceipt, HermesDeployment
from soul_link.host_adaptation import AdaptationReceipt, CompatibilityResult


class _NoopHostController:
    def detect(self, host: Path) -> CompatibilityResult:
        return CompatibilityResult("supported", "applied", ())

    def verify(self, host: Path) -> bool:
        return True

    def apply(self, host: Path, *, verifier, backup_root=None):
        return self.detect(host), None

    def rollback(self, receipt: AdaptationReceipt, *, trusted_backup_root=None) -> bool:
        return True


def _profile_only(deployment: HermesDeployment, monkeypatch) -> HermesDeployment:
    controller = _NoopHostController()
    monkeypatch.setattr(deployment, "_host_controller", lambda: controller)
    return deployment


def _host(tmp_path: Path) -> Path:
    host = tmp_path / "hermes"
    files = {
        "agent/context_engine.py": "class ContextEngine: pass\n",
        "agent/memory_provider.py": "class MemoryProvider: pass\n",
        "plugins/memory/__init__.py": "def load_memory_provider(): pass\n",
        "hermes_cli/plugins.py": "def register_context_engine(): pass\n",
    }
    for relative, text in files.items():
        path = host / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(text, encoding="utf-8")
    return host


def _tree_hash(root: Path) -> str:
    digest = hashlib.sha256()
    for path in sorted(p for p in root.rglob("*") if p.is_file()):
        digest.update(path.relative_to(root).as_posix().encode())
        digest.update(path.read_bytes())
    return digest.hexdigest()


def _deployment() -> HermesDeployment:
    return HermesDeployment(Path(__file__).resolve().parents[2])


def test_packaged_and_documented_plugin_assets_are_identical() -> None:
    root = Path(__file__).resolve().parents[2]
    for category in ("memory", "context"):
        for filename in ("__init__.py", "plugin.yaml"):
            packaged = root / "soul_link/hermes_assets" / category / filename
            documented = root / "adapters/hermes/plugin" / category / filename
            assert packaged.read_bytes() == documented.read_bytes()


def test_hermes_memory_provider_has_one_runtime_implementation() -> None:
    root = Path(__file__).resolve().parents[2]
    canonical = root / "soul_link/hermes_plugin/memory_provider.py"

    assert canonical.is_file()
    assert not (root / "adapters/hermes/memory_provider/__init__.py").exists()
    assert not (root / "adapters/hermes/memory_provider/plugin.yaml").exists()

    entrypoints = (
        root / "soul_link/hermes_assets/memory/__init__.py",
        root / "adapters/hermes/plugin/memory/__init__.py",
    )
    for entrypoint in entrypoints:
        source = entrypoint.read_text(encoding="utf-8")
        assert "from soul_link.hermes_plugin.memory_provider import SoulLinkMemoryProvider" in source


def test_legacy_adapter_controller_is_retired() -> None:
    root = Path(__file__).resolve().parents[2]

    assert not (root / "adapters/hermes/controller.py").exists()
    assert not (root / "adapters/hermes/manifest.yaml").exists()
    assert not (root / "adapters/hermes/tests/test_controller.py").exists()


def test_adapter_readme_uses_live_controller_entrypoints() -> None:
    root = Path(__file__).resolve().parents[2]
    readme = (root / "adapters/hermes/README.md").read_text(encoding="utf-8")

    assert "python -m adapters.hermes.controller" not in readme
    assert "`controller.py`" not in readme
    assert "`manifest.yaml`" not in readme
    assert "python -m soul_link.hermes_deploy detect" in readme
    assert "python -m soul_link.hermes_deploy rollback" in readme
    assert "python -m soul_link.host_adaptation detect" in readme
    assert "python -m soul_link.host_adaptation rollback" in readme
    assert "compatibility-soullink-runtime.yaml" in readme
    assert "one transaction" in readme


def test_detect_latest_spi_needs_no_host_source_mutation(tmp_path: Path, monkeypatch) -> None:
    deployment = _profile_only(_deployment(), monkeypatch)
    result = deployment.detect(_host(tmp_path), tmp_path / "home")

    assert result["host_source_mutation_required"] is False


def test_apply_and_rollback_restore_profile_byte_for_byte(tmp_path: Path, monkeypatch) -> None:
    host = _host(tmp_path)
    home = tmp_path / "home"
    old_plugin = home / "plugins/soullink"
    old_plugin.mkdir(parents=True)
    (old_plugin / "legacy.txt").write_text("legacy", encoding="utf-8")
    (home / "config.yaml").write_text("memory:\n  provider: legacy\ncustom: keep\n", encoding="utf-8")
    (home / "SOUL.md").write_text("old identity\n", encoding="utf-8")
    before_host = _tree_hash(host)
    before_profile = _tree_hash(home)
    deployment = _profile_only(_deployment(), monkeypatch)
    monkeypatch.setattr(deployment, "verify", lambda *_: True)

    receipt = deployment.apply(host, home)

    assert receipt is not None
    assert _tree_hash(host) == before_host
    config = yaml.safe_load((home / "config.yaml").read_text(encoding="utf-8"))
    assert config["custom"] == "keep"
    assert config["memory"]["provider"] == "soullink"
    assert config["context"]["engine"] == "pcltm-context"
    assert config["context"]["budget_tokens"] == 200_000
    assert config["compression"]["threshold_tokens"] == 200_000
    assert config["compression"]["enabled"] is True
    assert (home / "plugins/soullink/soullink-root.txt").is_file()
    assert "managed-by: SoulLink/PCLTM" in (home / "SOUL.md").read_text(encoding="utf-8")

    assert deployment.rollback(receipt) is True
    assert _tree_hash(home) == before_profile
    assert _tree_hash(host) == before_host


def test_real_apply_keeps_host_rollback_material_outside_host_checkout(tmp_path: Path, monkeypatch) -> None:
    host = _host(tmp_path)
    home = tmp_path / "home"
    home.mkdir()
    deployment = _deployment()
    controller = deployment._host_controller()
    for relative in controller.manifest.required_paths:
        path = host / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(f"baseline {relative}\n", encoding="utf-8")
    (host / "agent/memory_provider.py").write_text("class MemoryProvider: pass\n", encoding="utf-8")
    states = iter((
        CompatibilityResult("transformable", "applicable", ()),
        CompatibilityResult("transformable", "applicable", ()),
        CompatibilityResult("supported", "applied", ()),
    ))
    monkeypatch.setattr(controller, "detect", lambda _root: next(states))
    monkeypatch.setattr(controller, "verify", lambda _root: True)
    monkeypatch.setattr(controller, "_run", lambda _command, _root: 0)
    monkeypatch.setattr(deployment, "_host_controller", lambda: controller)
    monkeypatch.setattr(deployment, "verify", lambda *_args: True)

    receipt = deployment.apply(host, home)

    assert receipt is not None
    assert receipt.host_adaptation_receipt is not None
    host_receipt = AdaptationReceipt.load(receipt.host_adaptation_receipt)
    assert host_receipt.backup_path.is_relative_to(receipt.backup_path)
    assert not list(host.glob(".soullink-adapter-backup-*"))


def test_rollback_removes_runtime_directories_created_during_verification(tmp_path: Path, monkeypatch) -> None:
    host = _host(tmp_path)
    home = tmp_path / "home"
    home.mkdir()
    (home / "config.yaml").write_text("custom: original\n", encoding="utf-8")
    deployment = _profile_only(_deployment(), monkeypatch)

    def verify(*_args) -> bool:
        (home / "cache").mkdir(exist_ok=True)
        (home / "cache/runtime.json").write_text("derived\n", encoding="utf-8")
        (home / "logs").mkdir(exist_ok=True)
        return True

    monkeypatch.setattr(deployment, "verify", verify)
    receipt = deployment.apply(host, home)

    assert receipt is not None
    assert (home / "cache/runtime.json").is_file()
    assert deployment.rollback(receipt) is True
    assert not (home / "cache").exists()
    assert not (home / "logs").exists()


def test_apply_failure_compensates_without_residue(tmp_path: Path, monkeypatch) -> None:
    host = _host(tmp_path)
    home = tmp_path / "home"
    home.mkdir()
    (home / "config.yaml").write_text("custom: original\n", encoding="utf-8")
    before = _tree_hash(home)
    deployment = _profile_only(_deployment(), monkeypatch)
    monkeypatch.setattr(deployment, "verify", lambda *_: False)

    with pytest.raises(RuntimeError, match="verification failed"):
        deployment.apply(host, home)

    assert _tree_hash(home) == before
    assert not list(home.glob(".soullink-deploy-backup-*"))


def test_combined_apply_rolls_back_host_and_profile_as_one_transaction(tmp_path: Path, monkeypatch) -> None:
    host = _host(tmp_path)
    home = tmp_path / "home"
    home.mkdir()
    (home / "config.yaml").write_text("custom: original\n", encoding="utf-8")
    before_host = _tree_hash(host)
    before_profile = _tree_hash(home)
    deployment = _deployment()

    class RecordingController:
        def __init__(self) -> None:
            self.applied = False
            self.rolled_back = False
            self.receipt: AdaptationReceipt | None = None

        def detect(self, root: Path) -> CompatibilityResult:
            return CompatibilityResult(
                "supported" if self.applied else "transformable",
                "applied" if self.applied else "applicable",
                (),
            )

        def verify(self, root: Path) -> bool:
            return self.applied

        def apply(self, root: Path, *, verifier, backup_root=None):
            backup = Path(backup_root) / ".soullink-adapter-backup-combined"
            backup.mkdir()
            (backup / ".soullink-backup.json").write_text(
                json.dumps({"host_root": str(root.resolve()), "adapter_version": "combined"}),
                encoding="utf-8",
            )
            self.applied = True
            self.receipt = AdaptationReceipt(root.resolve(), backup.resolve(), "combined")
            assert verifier(root)
            return self.detect(root), self.receipt

        def rollback(self, receipt: AdaptationReceipt, *, trusted_backup_root=None) -> bool:
            assert receipt == self.receipt
            assert receipt.backup_path.parent == Path(trusted_backup_root).resolve()
            self.applied = False
            self.rolled_back = True
            shutil.rmtree(receipt.backup_path)
            return True

    controller = RecordingController()
    monkeypatch.setattr(deployment, "_host_controller", lambda: controller)
    monkeypatch.setattr(deployment, "verify", lambda *_: controller.applied and deployment._installed(home))

    receipt = deployment.apply(host, home)

    assert receipt is not None
    assert controller.applied is True
    assert receipt.host_adaptation_receipt is not None
    assert receipt.host_adaptation_receipt.is_file()
    assert deployment.rollback(receipt) is True
    assert controller.rolled_back is True
    assert _tree_hash(host) == before_host
    assert _tree_hash(home) == before_profile


def test_rollback_fails_closed_for_incomplete_backup(tmp_path: Path) -> None:

    host = _host(tmp_path)
    home = tmp_path / "home"
    home.mkdir()
    backup = home / ".soullink-deploy-backup-test"
    backup.mkdir()
    marker = {
        "host_root": str(host.resolve()), "hermes_home": str(home.resolve()),
        "soullink_root": str(Path(__file__).resolve().parents[2]),
        "adapter_version": "3", "entries": {"config.yaml": True},
    }
    (backup / ".soullink-deploy.json").write_text(json.dumps(marker), encoding="utf-8")
    (home / "config.yaml").write_text("active: true\n", encoding="utf-8")
    receipt = DeploymentReceipt(
        host.resolve(),
        home.resolve(),
        Path(__file__).resolve().parents[2],
        backup,
        None,
        "3",
    )

    with pytest.raises(RuntimeError, match="incomplete"):
        _deployment().rollback(receipt)

    assert (home / "config.yaml").read_text(encoding="utf-8") == "active: true\n"
    assert backup.is_dir()


def test_rollback_accepts_complete_legacy_v2_receipt(tmp_path: Path) -> None:
    host = _host(tmp_path)
    home = tmp_path / "home"
    home.mkdir()
    backup = home / ".soullink-deploy-backup-legacy"
    backup.mkdir()
    entries = {}
    for relative in HermesDeployment.managed:
        saved = backup / relative
        saved.parent.mkdir(parents=True, exist_ok=True)
        if relative.startswith("plugins/"):
            saved.mkdir()
            (saved / "legacy.txt").write_text(relative, encoding="utf-8")
        else:
            saved.write_text(f"legacy {relative}\n", encoding="utf-8")
        entries[relative] = True
    marker = {
        "host_root": str(host.resolve()),
        "hermes_home": str(home.resolve()),
        "soullink_root": str(Path(__file__).resolve().parents[2]),
        "adapter_version": "2",
        "entries": entries,
    }
    (backup / ".soullink-deploy.json").write_text(json.dumps(marker), encoding="utf-8")
    (home / "config.yaml").write_text("memory:\n  provider: soullink\n", encoding="utf-8")
    receipt = DeploymentReceipt(
        host.resolve(), home.resolve(), Path(__file__).resolve().parents[2], backup, None, "2"
    )

    assert _deployment().rollback(receipt) is True
    assert (home / "config.yaml").read_text(encoding="utf-8") == "legacy config.yaml\n"
    assert not backup.exists()


def test_rollback_rejects_partial_legacy_v2_receipt(tmp_path: Path) -> None:
    host = _host(tmp_path)
    home = tmp_path / "home"
    home.mkdir()
    backup = home / ".soullink-deploy-backup-partial-v2"
    backup.mkdir()
    (backup / "config.yaml").write_text("legacy\n", encoding="utf-8")
    marker = {
        "host_root": str(host.resolve()), "hermes_home": str(home.resolve()),
        "soullink_root": str(Path(__file__).resolve().parents[2]), "adapter_version": "2",
        "entries": {"config.yaml": True},
    }
    (backup / ".soullink-deploy.json").write_text(json.dumps(marker), encoding="utf-8")
    receipt = DeploymentReceipt(
        host.resolve(), home.resolve(), Path(__file__).resolve().parents[2], backup, None, "2"
    )

    with pytest.raises(RuntimeError, match="legacy deployment backup incomplete"):
        _deployment().rollback(receipt)
