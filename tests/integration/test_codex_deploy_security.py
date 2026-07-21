from pathlib import Path
import json

import pytest

from soul_link.codex_deploy import CodexDeployment, CodexDeploymentReceipt


def test_apply_rejects_symlinked_codex_home_without_external_write(tmp_path: Path) -> None:
    outside = tmp_path / "outside-home"
    outside.mkdir()
    linked_home = tmp_path / "codex-home"
    try:
        linked_home.symlink_to(outside, target_is_directory=True)
    except OSError as exc:
        pytest.skip(f"directory symlinks unavailable: {exc}")
    with pytest.raises(RuntimeError, match="symlink|reparse"):
        CodexDeployment().apply(linked_home, db_path=tmp_path / "runtime/pcltm.db", memfs_root=tmp_path / "runtime/memfs")
    assert not (outside / "config.toml").exists()


@pytest.mark.parametrize("mutation", ["env", "timeout", "approval"])
def test_verify_rejects_tampered_mcp_policy(tmp_path: Path, mutation: str) -> None:
    codex_home = tmp_path / "codex-home"
    deployment = CodexDeployment()
    deployment.apply(codex_home, db_path=tmp_path / "runtime/pcltm.db", memfs_root=tmp_path / "runtime/memfs")
    config = codex_home / "config.toml"
    text = config.read_text(encoding="utf-8")
    if mutation == "env":
        text = text.replace("HERMES_PCLTM_DB = ", "FOREIGN_DB = ")
    elif mutation == "timeout":
        text = text.replace("startup_timeout_sec = 20", "startup_timeout_sec = 1")
    else:
        text = text.replace('default_tools_approval_mode = "writes"', 'default_tools_approval_mode = "never"')
    config.write_text(text, encoding="utf-8")
    assert deployment.verify(codex_home) is False


def test_verify_returns_false_for_malformed_hook_schema(tmp_path: Path) -> None:
    codex_home = tmp_path / "codex-home"
    deployment = CodexDeployment()
    deployment.apply(codex_home, db_path=tmp_path / "runtime/pcltm.db", memfs_root=tmp_path / "runtime/memfs")
    (codex_home / "hooks.json").write_text(json.dumps({"hooks": {"SessionStart": ["invalid"]}}), encoding="utf-8")
    assert deployment.verify(codex_home) is False


def test_receipt_symlink_cannot_delete_external_target(tmp_path: Path) -> None:
    receipt_path = tmp_path / "receipt.json"
    receipt = CodexDeployment().apply(tmp_path / "codex-home", db_path=tmp_path / "runtime/pcltm.db", memfs_root=tmp_path / "runtime/memfs", receipt_path=receipt_path)
    assert receipt is not None
    external = tmp_path / "external-receipt.json"
    receipt_path.replace(external)
    try:
        receipt_path.symlink_to(external)
    except OSError as exc:
        pytest.skip(f"file symlinks unavailable: {exc}")
    with pytest.raises(RuntimeError, match="symlink|reparse"):
        CodexDeploymentReceipt.load(receipt_path)
    assert external.exists()


def test_rollback_rejects_incomplete_managed_set(tmp_path: Path) -> None:
    deployment = CodexDeployment()
    receipt = deployment.apply(tmp_path / "codex-home", db_path=tmp_path / "runtime/pcltm.db", memfs_root=tmp_path / "runtime/memfs", receipt_path=tmp_path / "receipt.json")
    assert receipt is not None
    marker_path = receipt.backup_path / ".soullink-codex-deploy.json"
    marker = json.loads(marker_path.read_text(encoding="utf-8"))
    marker["entries"] = {}
    marker["fingerprints"] = {}
    marker_path.write_text(json.dumps(marker), encoding="utf-8")
    forged = CodexDeploymentReceipt(receipt.codex_home, receipt.backup_path, receipt.receipt_path, receipt.adapter_version, {}, {})
    with pytest.raises(RuntimeError, match="managed entries"):
        deployment.rollback(forged)
