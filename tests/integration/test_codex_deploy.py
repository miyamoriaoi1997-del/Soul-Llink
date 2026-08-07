from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest

from soul_link.codex_deploy import BEGIN, CodexDeployment, CodexDeploymentReceipt


def _tree_digest(root: Path) -> dict[str, str]:
    if not root.exists():
        return {}
    return {
        path.relative_to(root).as_posix(): hashlib.sha256(path.read_bytes()).hexdigest()
        for path in sorted(root.rglob("*"))
        if path.is_file()
    }


def test_codex_deployment_apply_verify_rollback_is_byte_exact(tmp_path: Path) -> None:
    codex_home = tmp_path / "codex-home"
    codex_home.mkdir()
    original_config = b'model = "existing-model"\n'
    original_hooks = json.dumps(
        {"description": "existing", "hooks": {"Stop": []}},
        ensure_ascii=False,
        separators=(",", ":"),
    ).encode()
    (codex_home / "config.toml").write_bytes(original_config)
    (codex_home / "hooks.json").write_bytes(original_hooks)
    before = _tree_digest(codex_home)

    deployment = CodexDeployment()
    detected = deployment.detect(codex_home)
    assert detected["classification"] == "transformable"
    assert detected["host_source_mutation_required"] is False

    receipt = deployment.apply(
        codex_home,
        db_path=tmp_path / "runtime" / "pcltm.db",
        memfs_root=tmp_path / "runtime" / "memfs",
    )
    assert isinstance(receipt, CodexDeploymentReceipt)
    assert deployment.verify(codex_home)
    assert "[mcp_servers.soullink]" in (codex_home / "config.toml").read_text(encoding="utf-8")
    hooks = json.loads((codex_home / "hooks.json").read_text(encoding="utf-8"))
    assert {"SessionStart", "UserPromptSubmit", "PreCompact", "PostCompact", "Stop"} <= set(hooks["hooks"])

    assert deployment.rollback(receipt)
    assert not receipt.receipt_path.exists()
    assert _tree_digest(codex_home) == before
    assert (codex_home / "config.toml").read_bytes() == original_config
    assert (codex_home / "hooks.json").read_bytes() == original_hooks


def test_codex_deployment_refuses_foreign_soullink_mcp_table(tmp_path: Path) -> None:
    codex_home = tmp_path / "codex-home"
    codex_home.mkdir()
    (codex_home / "config.toml").write_text(
        '[mcp_servers.soullink]\ncommand = "foreign"\n', encoding="utf-8"
    )

    state = CodexDeployment().detect(codex_home)

    assert state["classification"] == "incompatible"
    assert "foreign_mcp_table" in state["blockers"]


def test_detect_rejects_invalid_existing_toml(tmp_path: Path) -> None:
    codex_home = tmp_path / "codex-home"
    codex_home.mkdir()
    (codex_home / "config.toml").write_text("broken = [", encoding="utf-8")
    state = CodexDeployment().detect(codex_home)
    assert state["classification"] == "incompatible"
    assert "invalid_config_toml" in state["blockers"]


@pytest.mark.parametrize("value", ['"bad"', "[]"])
def test_detect_rejects_invalid_mcp_server_shape(tmp_path: Path, value: str) -> None:
    codex_home = tmp_path / "codex-home"
    codex_home.mkdir()
    (codex_home / "config.toml").write_text(f"mcp_servers = {value}\n", encoding="utf-8")
    state = CodexDeployment().detect(codex_home)
    assert state["classification"] == "incompatible"
    assert "invalid_config_shape" in state["blockers"]


def test_detect_rejects_invalid_soullink_server_shape(tmp_path: Path) -> None:
    codex_home = tmp_path / "codex-home"
    codex_home.mkdir()
    (codex_home / "config.toml").write_text('[mcp_servers]\nsoullink = "bad"\n', encoding="utf-8")
    state = CodexDeployment().detect(codex_home)
    assert state["classification"] == "incompatible"
    assert "invalid_config_shape" in state["blockers"]


def test_detect_rejects_invalid_hook_event_group(tmp_path: Path) -> None:
    codex_home = tmp_path / "codex-home"
    deployment = CodexDeployment()
    deployment.apply(codex_home, db_path=tmp_path / "runtime/pcltm.db", memfs_root=tmp_path / "runtime/memfs")
    hooks_path = codex_home / "hooks.json"
    hooks = json.loads(hooks_path.read_text(encoding="utf-8"))
    hooks["hooks"]["SessionStart"] = "damaged"
    hooks_path.write_text(json.dumps(hooks), encoding="utf-8")
    state = deployment.detect(codex_home)
    assert state["classification"] == "incompatible"
    assert state["installed"] is False
    assert "invalid_hooks_shape" in state["blockers"]


@pytest.mark.parametrize("payload", [{}, {"hooks": {"Stop": [{}]}}])
def test_detect_rejects_missing_hook_schema_members(tmp_path: Path, payload: dict[str, object]) -> None:
    codex_home = tmp_path / "codex-home"
    codex_home.mkdir()
    (codex_home / "hooks.json").write_text(json.dumps(payload), encoding="utf-8")
    state = CodexDeployment().detect(codex_home)
    assert state["classification"] == "incompatible"
    assert "invalid_hooks_shape" in state["blockers"]


def test_apply_rejects_symlinked_managed_directory_without_external_write(tmp_path: Path) -> None:
    codex_home = tmp_path / "codex-home"
    outside = tmp_path / "outside"
    codex_home.mkdir()
    outside.mkdir()
    try:
        (codex_home / "soullink").symlink_to(outside, target_is_directory=True)
    except OSError as exc:
        pytest.skip(f"directory symlinks unavailable: {exc}")

    with pytest.raises(RuntimeError, match="unsafe_managed_path:soullink"):
        CodexDeployment().apply(
            codex_home,
            db_path=tmp_path / "runtime" / "pcltm.db",
            memfs_root=tmp_path / "runtime" / "memfs",
        )
    assert not (outside / "adapter.json").exists()


def test_verify_rejects_empty_managed_hook_arrays(tmp_path: Path) -> None:
    codex_home = tmp_path / "codex-home"
    deployment = CodexDeployment()
    deployment.apply(
        codex_home,
        db_path=tmp_path / "runtime" / "pcltm.db",
        memfs_root=tmp_path / "runtime" / "memfs",
    )
    hooks_path = codex_home / "hooks.json"
    hooks = json.loads(hooks_path.read_text(encoding="utf-8"))
    hooks["hooks"] = {event: [] for event in hooks["hooks"]}
    hooks_path.write_text(json.dumps(hooks), encoding="utf-8")
    assert deployment.verify(codex_home) is False


def test_verify_rejects_tampered_managed_hook_command(tmp_path: Path) -> None:
    codex_home = tmp_path / "codex-home"
    deployment = CodexDeployment()
    deployment.apply(
        codex_home,
        db_path=tmp_path / "runtime" / "pcltm.db",
        memfs_root=tmp_path / "runtime" / "memfs",
    )
    hooks_path = codex_home / "hooks.json"
    hooks = json.loads(hooks_path.read_text(encoding="utf-8"))
    hooks["hooks"]["SessionStart"][0]["hooks"][0]["commandWindows"] = "foreign-command"
    hooks_path.write_text(json.dumps(hooks), encoding="utf-8")
    assert deployment.verify(codex_home) is False


def test_apply_rejects_symlinked_managed_file_without_mutation(tmp_path: Path) -> None:
    codex_home = tmp_path / "codex-home"
    outside = tmp_path / "outside-config.toml"
    codex_home.mkdir()
    outside.write_bytes(b'operator = "external"\n')
    try:
        (codex_home / "config.toml").symlink_to(outside)
    except OSError as exc:
        pytest.skip(f"file symlinks unavailable: {exc}")
    with pytest.raises(RuntimeError, match="unsafe_managed_path:config.toml"):
        CodexDeployment().apply(
            codex_home,
            db_path=tmp_path / "runtime" / "pcltm.db",
            memfs_root=tmp_path / "runtime" / "memfs",
        )
    assert outside.read_bytes() == b'operator = "external"\n'


def test_verify_rejects_tampered_mcp_command(tmp_path: Path) -> None:
    codex_home = tmp_path / "codex-home"
    deployment = CodexDeployment()
    deployment.apply(
        codex_home,
        db_path=tmp_path / "runtime" / "pcltm.db",
        memfs_root=tmp_path / "runtime" / "memfs",
    )
    config = codex_home / "config.toml"
    text = config.read_text(encoding="utf-8")
    start = text.index('command = "') + len('command = "')
    end = text.index('"', start)
    config.write_text(text[:start] + "foreign-python" + text[end:], encoding="utf-8")
    assert deployment.verify(codex_home) is False


def test_reapply_preserves_user_config_after_managed_block(tmp_path: Path) -> None:
    codex_home = tmp_path / "codex-home"
    deployment = CodexDeployment()
    receipt = deployment.apply(
        codex_home,
        db_path=tmp_path / "runtime" / "pcltm.db",
        memfs_root=tmp_path / "runtime" / "memfs",
    )
    assert receipt is not None
    deployment.rollback(receipt)
    receipt = deployment.apply(
        codex_home,
        db_path=tmp_path / "runtime" / "pcltm.db",
        memfs_root=tmp_path / "runtime" / "memfs",
    )
    assert receipt is not None
    config = codex_home / "config.toml"
    config.write_text(config.read_text(encoding="utf-8") + '\n[profiles.local]\nmodel = "custom"\n', encoding="utf-8")
    # Force the upgrade path while retaining a valid managed block, and use a
    # distinct receipt so the prior deployment remains independently roll-backable.
    (codex_home / "soullink" / "adapter.json").unlink()
    deployment.apply(
        codex_home,
        db_path=tmp_path / "runtime" / "pcltm.db",
        memfs_root=tmp_path / "runtime" / "memfs",
        receipt_path=tmp_path / "upgrade-receipt.json",
    )
    updated = config.read_text(encoding="utf-8")
    assert '[profiles.local]\nmodel = "custom"' in updated
    assert updated.count(BEGIN) == 1


def test_receipt_body_cannot_redirect_rollback_unlink(tmp_path: Path) -> None:
    codex_home = tmp_path / "codex-home"
    outside = tmp_path / "must-survive.txt"
    outside.write_text("safe", encoding="utf-8")
    receipt_path = tmp_path / "receipt.json"
    receipt = CodexDeployment().apply(
        codex_home,
        db_path=tmp_path / "runtime" / "pcltm.db",
        memfs_root=tmp_path / "runtime" / "memfs",
        receipt_path=receipt_path,
    )
    assert receipt is not None
    payload = json.loads(receipt_path.read_text(encoding="utf-8"))
    payload["receipt_path"] = str(outside)
    receipt_path.write_text(json.dumps(payload), encoding="utf-8")
    loaded = CodexDeploymentReceipt.load(receipt_path)
    CodexDeployment().rollback(loaded)
    assert outside.read_text(encoding="utf-8") == "safe"
    assert not receipt_path.exists()


def test_apply_refuses_to_overwrite_existing_receipt(tmp_path: Path) -> None:
    receipt_path = tmp_path / "receipt.json"
    receipt_path.write_bytes(b"operator-owned")
    with pytest.raises(RuntimeError, match="already exists"):
        CodexDeployment().apply(
            tmp_path / "codex-home",
            db_path=tmp_path / "runtime" / "pcltm.db",
            memfs_root=tmp_path / "runtime" / "memfs",
            receipt_path=receipt_path,
        )
    assert receipt_path.read_bytes() == b"operator-owned"
    assert not (tmp_path / "codex-home").exists()


def test_receipt_replace_failure_removes_temp_and_restores(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    codex_home = tmp_path / "codex-home"
    receipt_path = tmp_path / "receipt.json"
    before = _tree_digest(codex_home)

    def fail_replace(source: Path, destination: Path) -> None:
        raise OSError("synthetic replace failure")

    monkeypatch.setattr("soul_link.codex_deploy.os.replace", fail_replace)
    with pytest.raises(OSError, match="synthetic replace failure"):
        CodexDeployment().apply(
            codex_home,
            db_path=tmp_path / "runtime" / "pcltm.db",
            memfs_root=tmp_path / "runtime" / "memfs",
            receipt_path=receipt_path,
        )
    assert _tree_digest(codex_home) == before
    assert not receipt_path.exists()
    assert not receipt_path.with_name("receipt.json.tmp").exists()


def test_receipt_write_failure_restores_original_bytes(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    codex_home = tmp_path / "codex-home"
    codex_home.mkdir()
    (codex_home / "config.toml").write_bytes(b'approval_policy = "on-request"\n')
    before = _tree_digest(codex_home)
    deployment = CodexDeployment()

    def fail_write(self: CodexDeploymentReceipt, path: Path) -> None:
        raise OSError("synthetic receipt failure")

    monkeypatch.setattr(CodexDeploymentReceipt, "write", fail_write)

    with pytest.raises(OSError, match="synthetic receipt failure"):
        deployment.apply(
            codex_home,
            db_path=tmp_path / "runtime" / "pcltm.db",
            memfs_root=tmp_path / "runtime" / "memfs",
            receipt_path=tmp_path / "receipt.json",
        )

    assert _tree_digest(codex_home) == before
