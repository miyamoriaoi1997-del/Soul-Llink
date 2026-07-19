"""Rollback binding validation tests.

Tests that rollback verifies receipt binding to current controller state before mutating host.
"""
from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import Mock

import pytest

from soul_link.hermes_deploy import DeploymentReceipt, HermesDeployment


def _host(tmp_path: Path) -> Path:
    """Create minimal Hermes host fixture."""
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


def _deployment() -> HermesDeployment:
    """Get HermesDeployment pointing to this repository."""
    return HermesDeployment(Path(__file__).resolve().parents[2])


def test_rollback_rejects_mismatched_adapter_version(tmp_path: Path, monkeypatch) -> None:
    """Rollback must reject receipt with different adapter version before any mutation."""
    host = _host(tmp_path)
    home = tmp_path / "home"
    home.mkdir()
    (home / "config.yaml").write_text("original: true\n", encoding="utf-8")
    deployment = _deployment()
    monkeypatch.setattr(deployment, "verify", lambda *_: True)

    receipt = deployment.apply(host, home)
    assert receipt is not None

    # Create receipt with wrong version
    wrong_receipt = DeploymentReceipt(
        receipt.host_root, receipt.hermes_home, receipt.soullink_root,
        receipt.backup_path, adapter_version="999",
        fingerprints=receipt.fingerprints
    )

    # Should fail immediately without touching host/home
    with pytest.raises(RuntimeError, match="version mismatch"):
        deployment.rollback(wrong_receipt)

    # Home should still have installed files (not rolled back)
    assert (home / "SOUL.md").exists()


def test_rollback_rejects_receipt_pointing_to_different_host(tmp_path: Path, monkeypatch) -> None:
    """Rollback must reject receipt with host_root != current operation target."""
    host = _host(tmp_path)
    home = tmp_path / "home"
    home.mkdir()
    (home / "config.yaml").write_text("original: true\n", encoding="utf-8")
    deployment = _deployment()
    monkeypatch.setattr(deployment, "verify", lambda *_: True)

    receipt = deployment.apply(host, home)
    assert receipt is not None

    # Create a different host
    other_host = tmp_path / "other_hermes"
    other_host.mkdir()

    # Create receipt pointing to wrong host
    wrong_host_receipt = DeploymentReceipt(
        other_host, receipt.hermes_home, receipt.soullink_root,
        receipt.backup_path, adapter_version=receipt.adapter_version,
        fingerprints=receipt.fingerprints
    )

    # Should fail because receipt.host_root doesn't match marker
    with pytest.raises(RuntimeError, match="mismatch|host"):
        deployment.rollback(wrong_host_receipt)


def test_rollback_rejects_receipt_pointing_to_different_home(tmp_path: Path, monkeypatch) -> None:
    """Rollback must reject receipt with hermes_home != marker hermes_home."""
    host = _host(tmp_path)
    home = tmp_path / "home"
    home.mkdir()
    (home / "config.yaml").write_text("original: true\n", encoding="utf-8")
    deployment = _deployment()
    monkeypatch.setattr(deployment, "verify", lambda *_: True)

    receipt = deployment.apply(host, home)
    assert receipt is not None

    # Create receipt with different home but backup still under original home
    # This creates a mismatch between receipt.hermes_home and marker hermes_home
    other_home = tmp_path / "other_home"
    other_home.mkdir()

    # Need to update the marker to point to other_home for this test
    marker_path = receipt.backup_path / ".soullink-deploy.json"
    marker = json.loads(marker_path.read_text(encoding="utf-8"))
    marker["hermes_home"] = str(other_home)
    marker_path.write_text(json.dumps(marker), encoding="utf-8")

    # Now rollback with receipt pointing to different home
    wrong_home_receipt = DeploymentReceipt(
        receipt.host_root, other_home, receipt.soullink_root,
        receipt.backup_path, adapter_version=receipt.adapter_version,
        fingerprints=receipt.fingerprints
    )

    # Should fail because backup is not under receipt.hermes_home
    with pytest.raises(RuntimeError, match="invalid.*backup|mismatch"):
        deployment.rollback(wrong_home_receipt)


def test_rollback_rejects_backup_outside_home(tmp_path: Path, monkeypatch) -> None:
    """Rollback must reject receipt with backup_path outside hermes_home."""
    host = _host(tmp_path)
    home = tmp_path / "home"
    home.mkdir()
    deployment = _deployment()
    monkeypatch.setattr(deployment, "verify", lambda *_: True)

    receipt = deployment.apply(host, home)
    assert receipt is not None

    # Create receipt with backup outside home
    evil_backup = tmp_path / "evil_backup"
    evil_backup.mkdir()
    (evil_backup / ".soullink-deploy.json").write_text(
        json.dumps({
            "host_root": str(receipt.host_root),
            "hermes_home": str(home),
            "soullink_root": str(receipt.soullink_root),
            "adapter_version": deployment.adapter_version,
            "entries": {},
            "fingerprints": {}
        }), encoding="utf-8"
    )

    evil_receipt = DeploymentReceipt(
        receipt.host_root, receipt.hermes_home, receipt.soullink_root,
        evil_backup, adapter_version=receipt.adapter_version,
        fingerprints={}
    )

    # Should fail because backup is not under home
    with pytest.raises(RuntimeError, match="invalid.*backup"):
        deployment.rollback(evil_receipt)


def test_rollback_rejects_backup_with_wrong_naming(tmp_path: Path, monkeypatch) -> None:
    """Rollback must reject receipt with backup_path not matching .soullink-deploy-backup-* pattern."""
    host = _host(tmp_path)
    home = tmp_path / "home"
    home.mkdir()
    deployment = _deployment()
    monkeypatch.setattr(deployment, "verify", lambda *_: True)

    receipt = deployment.apply(host, home)
    assert receipt is not None

    # Create backup with wrong name
    wrong_name_backup = home / "wrong_name"
    wrong_name_backup.mkdir()
    (wrong_name_backup / ".soullink-deploy.json").write_text(
        json.dumps({
            "host_root": str(receipt.host_root),
            "hermes_home": str(home),
            "soullink_root": str(receipt.soullink_root),
            "adapter_version": deployment.adapter_version,
            "entries": {},
            "fingerprints": {}
        }), encoding="utf-8"
    )

    wrong_name_receipt = DeploymentReceipt(
        receipt.host_root, receipt.hermes_home, receipt.soullink_root,
        wrong_name_backup, adapter_version=receipt.adapter_version,
        fingerprints={}
    )

    # Should fail because backup name is wrong
    with pytest.raises(RuntimeError, match="invalid.*backup"):
        deployment.rollback(wrong_name_receipt)
