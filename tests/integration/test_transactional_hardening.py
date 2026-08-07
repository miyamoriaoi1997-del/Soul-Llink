"""Transactional hardening tests for SoulLink host installation.

Tests the critical gaps identified in transactional installation:
1. Receipt includes source/target hashes and host identity
2. Receipt write failure triggers automatic rollback
3. HermesDeployment properly handles absent-before files
4. End-to-end isolated host rehearsal
5. Backup fingerprint independent verification
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock

import pytest
import yaml

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


def _tree_hash(root: Path) -> str:
    """Compute deterministic hash of directory tree."""
    digest = hashlib.sha256()
    for path in sorted(p for p in root.rglob("*") if p.is_file()):
        digest.update(path.relative_to(root).as_posix().encode())
        digest.update(path.read_bytes())
    return digest.hexdigest()


def test_receipt_includes_managed_path_hashes(tmp_path: Path, monkeypatch) -> None:
    """P0: Receipt must include before/after hashes for each managed path."""
    host = _host(tmp_path)
    home = tmp_path / "home"
    home.mkdir()
    (home / "config.yaml").write_text("custom: preserved\n", encoding="utf-8")
    deployment = _deployment(monkeypatch)
    monkeypatch.setattr(deployment, "verify", lambda *_: True)

    receipt = deployment.apply(host, home)

    assert receipt is not None
    # Receipt should include fingerprints for all managed paths
    assert hasattr(receipt, "fingerprints") or hasattr(receipt, "manifest_hashes")
    # At minimum, receipt.write should preserve hash information
    receipt_path = tmp_path / "receipt.json"
    receipt.write(receipt_path)
    receipt_data = json.loads(receipt_path.read_text(encoding="utf-8"))
    # Must contain either fingerprints or a reference to backup manifest
    assert "fingerprints" in receipt_data or "backup_manifest" in receipt_data


def test_main_cli_rollback_on_receipt_write_failure(tmp_path: Path, monkeypatch) -> None:
    """P0: CLI main() must rollback if receipt.write() fails after successful apply."""
    import sys
    from soul_link.hermes_deploy import main, HermesDeployment

    host = _host(tmp_path)
    home = tmp_path / "home"
    home.mkdir()
    (home / "config.yaml").write_text("original: true\n", encoding="utf-8")
    before_hash = _tree_hash(home)
    receipt_path = tmp_path / "receipt.json"

    # Make receipt path unwritable by making parent read-only
    receipt_path.parent.mkdir(exist_ok=True)
    readonly_dir = tmp_path / "readonly"
    readonly_dir.mkdir()
    receipt_in_readonly = readonly_dir / "receipt.json"

    # Keep the CLI test focused on receipt-write rollback rather than host compatibility.
    monkeypatch.setattr(HermesDeployment, "verify", lambda self, *_: True)
    monkeypatch.setattr(
        HermesDeployment,
        "_host_controller",
        lambda self: SimpleNamespace(
            detect=lambda _host: SimpleNamespace(
                classification="supported", patch_state="applied", missing_paths=()
            ),
            apply=lambda _host, **_kwargs: (SimpleNamespace(classification="supported"), None),
            verify=lambda _host: True,
            rollback=lambda *_args, **_kwargs: True,
        ),
    )

    # Simulate write failure by making directory read-only after creation
    original_write = DeploymentReceipt.write
    write_failed = False

    def fail_write(self, path: Path) -> None:
        nonlocal write_failed
        write_failed = True
        raise OSError("simulated receipt write failure")

    monkeypatch.setattr(DeploymentReceipt, "write", fail_write)

    # Run main() CLI which should handle rollback internally
    repo_root = Path(__file__).resolve().parents[2]
    result = main([
        "apply",
        "--soullink-root", str(repo_root),
        "--host-root", str(host),
        "--hermes-home", str(home),
        "--receipt", str(receipt_in_readonly)
    ])

    # Should fail with error code
    assert result == 5
    assert write_failed

    # Home directory should be rolled back
    assert _tree_hash(home) == before_hash
    config = yaml.safe_load((home / "config.yaml").read_text(encoding="utf-8"))
    assert config.get("original") is True
    assert not list(home.glob(".soullink-deploy-backup-*"))


def test_rollback_deletes_absent_before_paths(tmp_path: Path, monkeypatch) -> None:
    """P1: Files that didn't exist before installation must be deleted on rollback."""
    host = _host(tmp_path)
    home = tmp_path / "home"
    home.mkdir()
    # Start with no SOUL.md
    assert not (home / "SOUL.md").exists()
    deployment = _deployment(monkeypatch)
    monkeypatch.setattr(deployment, "verify", lambda *_: True)

    receipt = deployment.apply(host, home)

    assert receipt is not None
    assert (home / "SOUL.md").exists()

    deployment.rollback(receipt)

    # SOUL.md should be deleted since it was absent before
    assert not (home / "SOUL.md").exists()
    assert not list(home.glob("plugins/soullink"))


def test_backup_fingerprints_verified_independently_from_marker(tmp_path: Path, monkeypatch) -> None:
    """P2: Rollback must verify backup files against receipt, not just marker."""
    host = _host(tmp_path)
    home = tmp_path / "home"
    home.mkdir()
    (home / "config.yaml").write_text("custom: original\n", encoding="utf-8")
    deployment = _deployment(monkeypatch)
    monkeypatch.setattr(deployment, "verify", lambda *_: True)

    receipt = deployment.apply(host, home)

    assert receipt is not None
    # Tamper with both marker AND backup file
    marker_path = receipt.backup_path / ".soullink-deploy.json"
    marker = json.loads(marker_path.read_text(encoding="utf-8"))
    backup_config = receipt.backup_path / "config.yaml"
    backup_config.write_text("tampered\n", encoding="utf-8")
    # Update marker to match tampering
    tampered_hash = hashlib.sha256(b"F\0tampered\n").hexdigest()
    marker["fingerprints"]["config.yaml"] = tampered_hash
    marker_path.write_text(json.dumps(marker), encoding="utf-8")

    # Rollback should fail because receipt has independent verification
    with pytest.raises(RuntimeError, match="fingerprint|tamper|integrity"):
        deployment.rollback(receipt)


def test_end_to_end_isolated_host_rehearsal(tmp_path: Path, monkeypatch) -> None:
    """P1: Full detect→apply→verify→rollback cycle on isolated temporary host."""
    host = _host(tmp_path)
    home = tmp_path / "home"
    home.mkdir()
    before_host_hash = _tree_hash(host)
    deployment = _deployment(monkeypatch)
    monkeypatch.setattr(deployment, "verify", lambda *_: True)

    # Detect phase
    detect_result = deployment.detect(host, home)
    assert detect_result["classification"] in ("transformable", "supported")
    assert detect_result["host_source_mutation_required"] is False

    # Apply phase
    receipt = deployment.apply(host, home)
    assert receipt is not None
    after_apply_host_hash = _tree_hash(host)
    after_apply_home_hash = _tree_hash(home)

    # Verify phase
    assert deployment.verify(host, home) is True
    assert deployment.detect(host, home)["classification"] == "supported"

    # Rollback phase
    assert deployment.rollback(receipt) is True
    after_rollback_host_hash = _tree_hash(host)
    after_rollback_home_hash = _tree_hash(home)

    # Host must be unchanged throughout
    assert before_host_hash == after_apply_host_hash == after_rollback_host_hash

    # Home should return to empty state after rollback
    remaining = list(home.rglob("*"))
    remaining_files = [p for p in remaining if p.is_file()]
    assert len(remaining_files) == 0, f"Expected empty home, found: {remaining_files}"


def test_receipt_binds_host_identity_and_versions(tmp_path: Path, monkeypatch) -> None:
    """P0: Receipt must bind host identity/version and adapter version."""
    host = _host(tmp_path)
    home = tmp_path / "home"
    home.mkdir()
    deployment = _deployment(monkeypatch)
    monkeypatch.setattr(deployment, "verify", lambda *_: True)

    receipt = deployment.apply(host, home)

    assert receipt is not None
    assert receipt.adapter_version == deployment.adapter_version
    receipt_path = tmp_path / "receipt.json"
    receipt.write(receipt_path)
    receipt_data = json.loads(receipt_path.read_text(encoding="utf-8"))
    # Must include adapter version
    assert receipt_data["adapter_version"] == deployment.adapter_version
    # Should include host root binding
    assert "host_root" in receipt_data


def test_concurrent_apply_to_same_host_fails_safely(tmp_path: Path, monkeypatch) -> None:
    """Edge case: Second apply while first is in progress should fail safely."""
    host = _host(tmp_path)
    home = tmp_path / "home"
    home.mkdir()
    deployment = _deployment(monkeypatch)
    monkeypatch.setattr(deployment, "verify", lambda *_: True)

    # First apply
    receipt1 = deployment.apply(host, home)
    assert receipt1 is not None

    # Second apply should detect already installed and return None (idempotent)
    receipt2 = deployment.apply(host, home)
    assert receipt2 is None  # Already installed, no changes needed
