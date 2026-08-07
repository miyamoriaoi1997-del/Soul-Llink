from pathlib import Path
import json
import os
import subprocess

import pytest

from soul_link.codex_deploy import CodexDeployment, CodexDeploymentReceipt, main


def _junction(link: Path, target: Path) -> None:
    if os.name != "nt":
        pytest.skip("Windows junction test")
    target.mkdir(parents=True, exist_ok=True)
    completed = subprocess.run(
        ["cmd", "/c", "mklink", "/J", str(link), str(target)],
        capture_output=True,
        check=False,
    )
    if completed.returncode:
        pytest.skip(f"junction creation unavailable (exit {completed.returncode})")


def test_apply_rejects_symlinked_codex_home_without_external_write(tmp_path: Path) -> None:
    outside = tmp_path / "outside-home"
    outside.mkdir()
    linked_home = tmp_path / "codex-home"
    try:
        linked_home.symlink_to(outside, target_is_directory=True)
    except OSError as exc:
        pytest.skip(f"directory symlinks unavailable: {exc}")
    with pytest.raises(RuntimeError, match="symlink|reparse|unsafe_managed_path"):
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


def test_cli_apply_rejects_symlinked_receipt_without_external_write(tmp_path: Path) -> None:
    receipt_path = tmp_path / "receipt.json"
    outside = tmp_path / "outside-receipt.json"
    outside.write_bytes(b"operator-owned")
    try:
        receipt_path.symlink_to(outside)
    except OSError as exc:
        pytest.skip(f"file symlinks unavailable: {exc}")
    with pytest.raises(RuntimeError, match="symlink|reparse"):
        main([
            "apply",
            "--codex-home", str(tmp_path / "codex-home"),
            "--db", str(tmp_path / "runtime/pcltm.db"),
            "--memfs", str(tmp_path / "runtime/memfs"),
            "--receipt", str(receipt_path),
        ])
    assert outside.read_bytes() == b"operator-owned"
    assert not (tmp_path / "codex-home").exists()


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


@pytest.mark.parametrize("relative", ["config.toml", "hooks.json", "soullink/adapter.json"])
def test_apply_rejects_receipt_overlapping_managed_runtime(tmp_path: Path, relative: str) -> None:
    codex_home = tmp_path / "codex-home"
    with pytest.raises(RuntimeError, match="overlaps managed runtime"):
        CodexDeployment().apply(
            codex_home,
            db_path=tmp_path / "runtime/pcltm.db",
            memfs_root=tmp_path / "runtime/memfs",
            receipt_path=codex_home / relative,
        )


def test_receipt_cannot_redirect_rollback_to_another_deployment(tmp_path: Path) -> None:
    deployment = CodexDeployment()
    first_path = tmp_path / "first-receipt.json"
    second_path = tmp_path / "second-receipt.json"
    first = deployment.apply(tmp_path / "first", db_path=tmp_path / "runtime/first.db", memfs_root=tmp_path / "runtime/first", receipt_path=first_path)
    second = deployment.apply(tmp_path / "second", db_path=tmp_path / "runtime/second.db", memfs_root=tmp_path / "runtime/second", receipt_path=second_path)
    assert first is not None and second is not None
    first_path.write_bytes(second_path.read_bytes())
    redirected = CodexDeploymentReceipt.load(first_path)
    with pytest.raises(RuntimeError, match="receipt path does not match backup"):
        deployment.rollback(redirected)
    assert deployment.verify(tmp_path / "first")
    assert deployment.verify(tmp_path / "second")


@pytest.mark.parametrize("layer", ["home", "ancestor", "managed_child", "receipt_parent"])
def test_apply_rejects_native_windows_junctions(tmp_path: Path, layer: str) -> None:
    outside = tmp_path / f"outside-{layer}"
    real_home = tmp_path / "real-home"
    codex_home = real_home
    receipt_path = tmp_path / "receipt.json"

    if layer == "home":
        codex_home = tmp_path / "linked-home"
        _junction(codex_home, outside)
    elif layer == "ancestor":
        ancestor = tmp_path / "linked-ancestor"
        _junction(ancestor, outside)
        codex_home = ancestor / "codex-home"
    elif layer == "managed_child":
        real_home.mkdir()
        _junction(real_home / "soullink", outside)
    else:
        receipt_parent = tmp_path / "linked-receipts"
        _junction(receipt_parent, outside)
        receipt_path = receipt_parent / "receipt.json"

    with pytest.raises(RuntimeError, match="symlink|reparse|unsafe_managed_path"):
        CodexDeployment().apply(
            codex_home,
            db_path=tmp_path / "runtime/pcltm.db",
            memfs_root=tmp_path / "runtime/memfs",
            receipt_path=receipt_path,
        )
    assert not (outside / "adapter.json").exists()
    assert not (outside / "receipt.json").exists()
