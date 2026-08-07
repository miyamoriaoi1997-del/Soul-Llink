"""Fingerprint security boundary tests.

Tests that fingerprints correctly detect single-sided corruption but document
limitations regarding coordinated tampering of receipt+marker+backup.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
from types import SimpleNamespace

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


def _deployment(monkeypatch) -> HermesDeployment:
    """Get a deployment with a supported no-op host adapter seam."""
    deployment = HermesDeployment(Path(__file__).resolve().parents[2])
    controller = SimpleNamespace(
        detect=lambda _host: SimpleNamespace(
            classification="supported", patch_state="applied", missing_paths=()
        ),
        apply=lambda _host, **_kwargs: (SimpleNamespace(classification="supported"), None),
        verify=lambda _host: True,
        rollback=lambda *_args, **_kwargs: True,
    )
    monkeypatch.setattr(deployment, "_host_controller", lambda: controller)
    return deployment


def test_fingerprints_detect_tampered_backup_file_only(tmp_path: Path, monkeypatch) -> None:
    """Fingerprints detect corruption when only backup files are tampered (single-sided)."""
    host = _host(tmp_path)
    home = tmp_path / "home"
    home.mkdir()
    (home / "config.yaml").write_text("original: true\n", encoding="utf-8")
    deployment = _deployment(monkeypatch)
    monkeypatch.setattr(deployment, "verify", lambda *_: True)

    receipt = deployment.apply(host, home)
    assert receipt is not None

    # Tamper ONLY backup file (not marker, not receipt)
    backup_config = receipt.backup_path / "config.yaml"
    backup_config.write_text("tampered_backup\n", encoding="utf-8")

    # Should detect tampering via _validate_backup
    with pytest.raises(RuntimeError, match="fingerprint mismatch"):
        deployment.rollback(receipt)


def test_fingerprints_detect_tampered_marker_only(tmp_path: Path, monkeypatch) -> None:
    """Fingerprints detect corruption when only marker is tampered (single-sided)."""
    host = _host(tmp_path)
    home = tmp_path / "home"
    home.mkdir()
    (home / "config.yaml").write_text("original: true\n", encoding="utf-8")
    deployment = _deployment(monkeypatch)
    monkeypatch.setattr(deployment, "verify", lambda *_: True)

    receipt = deployment.apply(host, home)
    assert receipt is not None

    # Tamper ONLY marker fingerprints (not backup files, not receipt)
    marker_path = receipt.backup_path / ".soullink-deploy.json"
    marker = json.loads(marker_path.read_text(encoding="utf-8"))
    marker["fingerprints"]["config.yaml"] = "0" * 64  # wrong hash
    marker_path.write_text(json.dumps(marker), encoding="utf-8")

    # Should detect mismatch between receipt and marker fingerprints
    with pytest.raises(RuntimeError, match="fingerprint mismatch"):
        deployment.rollback(receipt)


def test_fingerprints_detect_missing_backup_file(tmp_path: Path, monkeypatch) -> None:
    """Fingerprints detect when backup file is missing (single-sided deletion)."""
    host = _host(tmp_path)
    home = tmp_path / "home"
    home.mkdir()
    (home / "config.yaml").write_text("original: true\n", encoding="utf-8")
    deployment = _deployment(monkeypatch)
    monkeypatch.setattr(deployment, "verify", lambda *_: True)

    receipt = deployment.apply(host, home)
    assert receipt is not None

    # Delete ONLY backup file
    backup_config = receipt.backup_path / "config.yaml"
    backup_config.unlink()

    # Should detect incomplete backup
    with pytest.raises(RuntimeError, match="backup incomplete"):
        deployment.rollback(receipt)


def test_receipt_detects_marker_entries_injection_before_rollback(tmp_path: Path, monkeypatch) -> None:
    """A marker-only entries edit must not authorize deletion of an unmanaged file."""
    host = _host(tmp_path)
    home = tmp_path / "home"
    home.mkdir()
    unmanaged = home / "unmanaged.txt"
    unmanaged.write_text("preserve me\n", encoding="utf-8")
    deployment = _deployment(monkeypatch)
    monkeypatch.setattr(deployment, "verify", lambda *_: True)

    receipt = deployment.apply(host, home)
    assert receipt is not None
    marker_path = receipt.backup_path / ".soullink-deploy.json"
    marker = json.loads(marker_path.read_text(encoding="utf-8"))
    marker["entries"]["unmanaged.txt"] = False
    marker_path.write_text(json.dumps(marker), encoding="utf-8")

    with pytest.raises(RuntimeError, match="manifest|entries|mismatch"):
        deployment.rollback(receipt)

    assert unmanaged.read_text(encoding="utf-8") == "preserve me\n"
    assert receipt.backup_path.exists()


def test_coordinated_tampering_not_detected_without_external_anchor() -> None:
    """Document: coordinated tampering of receipt+marker+backup is NOT in threat model.

    This test documents the security boundary: fingerprints prevent single-sided
    corruption but do NOT prevent an attacker who can modify receipt.json, marker,
    AND backup files together.

    Without external trusted anchors (HSM, remote signature, user-verified hashes),
    all three artifacts reside in the filesystem and can be modified together.
    """
    # This is a documentation test - it always passes
    # It exists to make the security boundary explicit in test coverage

    security_boundary = {
        "protects_against": [
            "Tampered backup file alone (fingerprint mismatch detected)",
            "Tampered marker alone (receipt vs marker mismatch detected)",
            "Missing backup file (incomplete backup detected)",
            "Receipt with wrong adapter version (version mismatch detected)",
            "Receipt pointing to different host/home (binding mismatch detected)",
        ],
        "does_not_protect_against": [
            "Coordinated modification of receipt.json + marker + backup files together",
            "Attacker with write access to both $HERMES_HOME and receipt storage",
        ],
        "rationale": (
            "All three artifacts (receipt, marker, backup) are filesystem files "
            "with no external trusted anchor. Fingerprints create independence between "
            "copies to detect single-sided corruption, but cannot prevent an attacker "
            "who modifies all copies consistently."
        ),
        "future_mitigation": [
            "Receipt signing with user private key or HSM",
            "Remote receipt storage with server-side integrity checks",
            "User confirmation of receipt hash at apply time (external to filesystem)",
        ]
    }

    # Test passes to document that this limitation is understood and intentional
    assert any("Coordinated modification" in item for item in security_boundary["does_not_protect_against"])
    assert len(security_boundary["protects_against"]) >= 5
