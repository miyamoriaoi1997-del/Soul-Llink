from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest
import yaml

from soul_link.hermes_deploy import DeploymentReceipt, HermesDeployment


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


def test_detect_latest_spi_needs_no_host_source_mutation(tmp_path: Path) -> None:
    result = _deployment().detect(_host(tmp_path), tmp_path / "home")
    assert result["classification"] == "transformable"
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
    deployment = _deployment()
    monkeypatch.setattr(deployment, "verify", lambda *_: True)

    receipt = deployment.apply(host, home)

    assert receipt is not None
    assert _tree_hash(host) == before_host
    config = yaml.safe_load((home / "config.yaml").read_text(encoding="utf-8"))
    assert config["custom"] == "keep"
    assert config["memory"]["provider"] == "soullink"
    assert config["context"]["engine"] == "pcltm-context"
    assert config["compression"]["enabled"] is False
    assert (home / "plugins/soullink/soullink-root.txt").is_file()
    assert "managed-by: SoulLink/PCLTM" in (home / "SOUL.md").read_text(encoding="utf-8")

    assert deployment.rollback(receipt) is True
    assert _tree_hash(home) == before_profile
    assert _tree_hash(host) == before_host


def test_backup_copy_failure_never_mutates_original_profile(tmp_path: Path, monkeypatch) -> None:
    host = _host(tmp_path)
    home = tmp_path / "home"
    home.mkdir()
    config = home / "config.yaml"
    config.write_text("custom: original\n", encoding="utf-8")
    before = _tree_hash(home)

    def fail_copy(*_args, **_kwargs):
        raise OSError("simulated backup disk failure")

    monkeypatch.setattr("soul_link.hermes_deploy.shutil.copy2", fail_copy)
    with pytest.raises(OSError, match="disk failure"):
        _deployment().apply(host, home)

    assert config.read_text(encoding="utf-8") == "custom: original\n"
    assert _tree_hash(home) == before
    assert not list(home.glob(".soullink-deploy-backup-*"))


def test_rollback_rejects_tampered_backup_without_deleting_active_install(
    tmp_path: Path, monkeypatch
) -> None:
    host = _host(tmp_path)
    home = tmp_path / "home"
    home.mkdir()
    (home / "config.yaml").write_text("custom: original\n", encoding="utf-8")
    deployment = _deployment()
    monkeypatch.setattr(deployment, "verify", lambda *_: True)
    receipt = deployment.apply(host, home)
    assert receipt is not None
    active_files = {
        relative: _tree_hash(home / relative) if (home / relative).is_dir()
        else hashlib.sha256((home / relative).read_bytes()).hexdigest()
        for relative in deployment.managed
    }
    (receipt.backup_path / "config.yaml").write_text("tampered\n", encoding="utf-8")

    with pytest.raises(RuntimeError, match="fingerprint mismatch"):
        deployment.rollback(receipt)

    current_files = {
        relative: _tree_hash(home / relative) if (home / relative).is_dir()
        else hashlib.sha256((home / relative).read_bytes()).hexdigest()
        for relative in deployment.managed
    }
    assert current_files == active_files
    assert receipt.backup_path.is_dir()


def test_apply_failure_compensates_without_residue(tmp_path: Path, monkeypatch) -> None:
    host = _host(tmp_path)
    home = tmp_path / "home"
    home.mkdir()
    (home / "config.yaml").write_text("custom: original\n", encoding="utf-8")
    before = _tree_hash(home)
    deployment = _deployment()
    monkeypatch.setattr(deployment, "verify", lambda *_: False)

    with pytest.raises(RuntimeError, match="verification failed"):
        deployment.apply(host, home)

    assert _tree_hash(home) == before
    assert not list(home.glob(".soullink-deploy-backup-*"))


def test_rollback_fails_closed_for_incomplete_backup(tmp_path: Path) -> None:
    host = _host(tmp_path)
    home = tmp_path / "home"
    home.mkdir()
    backup = home / ".soullink-deploy-backup-test"
    backup.mkdir()
    marker = {
        "host_root": str(host.resolve()), "hermes_home": str(home.resolve()),
        "soullink_root": str(Path(__file__).resolve().parents[2]),
        "adapter_version": "2", "entries": {"config.yaml": True},
    }
    (backup / ".soullink-deploy.json").write_text(json.dumps(marker), encoding="utf-8")
    (home / "config.yaml").write_text("active: true\n", encoding="utf-8")
    receipt = DeploymentReceipt(host.resolve(), home.resolve(), Path(__file__).resolve().parents[2], backup, "2")

    with pytest.raises(RuntimeError, match="incomplete"):
        _deployment().rollback(receipt)

    assert (home / "config.yaml").read_text(encoding="utf-8") == "active: true\n"
    assert backup.is_dir()
