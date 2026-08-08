from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest

from soul_link.zcode_deploy import BEGIN, ZCodeDeployment, ZCodeDeploymentReceipt


def _tree_digest(root: Path) -> dict[str, str]:
    if not root.exists():
        return {}
    return {
        path.relative_to(root).as_posix(): hashlib.sha256(path.read_bytes()).hexdigest()
        for path in sorted(root.rglob("*"))
        if path.is_file()
    }


def test_zcode_deployment_apply_verify_rollback_is_byte_exact(tmp_path: Path) -> None:
    zcode_root = tmp_path / "zcode-cli"
    zcode_root.mkdir()
    original_config = json.dumps(
        {"hooks": {"enabled": False}, "mcp": {"servers": {}}, "custom": "keep"},
        ensure_ascii=False,
    ).encode()
    (zcode_root / "config.json").write_bytes(original_config)
    before = _tree_digest(zcode_root)

    deployment = ZCodeDeployment()
    detected = deployment.detect(zcode_root)
    assert detected["classification"] == "transformable"
    assert detected["host_source_mutation_required"] is False

    receipt = deployment.apply(
        zcode_root,
        db_path=tmp_path / "runtime" / "pcltm.db",
        memfs_root=tmp_path / "runtime" / "memfs",
    )
    assert isinstance(receipt, ZCodeDeploymentReceipt)
    assert deployment.verify(zcode_root)
    config = json.loads((zcode_root / "config.json").read_text(encoding="utf-8"))
    assert config["custom"] == "keep"
    assert config["hooks"]["enabled"] is True
    assert set(config["hooks"]["events"]) == {
        "SessionStart", "UserPromptSubmit", "PreToolUse", "PermissionRequest",
        "PostToolUse", "PostToolUseFailure", "Stop",
    }
    assert config["mcp"]["servers"]["soullink"]["command"] is not None

    assert deployment.rollback(receipt)
    assert not receipt.receipt_path.exists()
    assert _tree_digest(zcode_root) == before
    assert (zcode_root / "config.json").read_bytes() == original_config


def test_zcode_deployment_refuses_foreign_soullink_mcp_server(tmp_path: Path) -> None:
    zcode_root = tmp_path / "zcode-cli"
    zcode_root.mkdir()
    (zcode_root / "config.json").write_text(json.dumps(
        {"mcp": {"servers": {"soullink": {"command": "foreign"}}}}
    ), encoding="utf-8")

    state = ZCodeDeployment().detect(zcode_root)

    assert state["classification"] == "incompatible"
    assert "foreign_mcp_server" in state["blockers"]


def test_detect_rejects_invalid_existing_json(tmp_path: Path) -> None:
    zcode_root = tmp_path / "zcode-cli"
    zcode_root.mkdir()
    (zcode_root / "config.json").write_text("{broken", encoding="utf-8")
    state = ZCodeDeployment().detect(zcode_root)
    assert state["classification"] == "incompatible"
    assert "invalid_config_json" in state["blockers"]


def test_detect_rejects_invalid_config_shape(tmp_path: Path) -> None:
    zcode_root = tmp_path / "zcode-cli"
    zcode_root.mkdir()
    (zcode_root / "config.json").write_text(json.dumps({"mcp": {"servers": "bad"}}), encoding="utf-8")
    state = ZCodeDeployment().detect(zcode_root)
    assert state["classification"] == "incompatible"
    assert "invalid_config_shape" in state["blockers"]


def test_detect_accepts_hooks_without_events_as_transformable(tmp_path: Path) -> None:
    zcode_root = tmp_path / "zcode-cli"
    zcode_root.mkdir()
    (zcode_root / "config.json").write_text(json.dumps({"hooks": {"enabled": False}}), encoding="utf-8")
    state = ZCodeDeployment().detect(zcode_root)
    assert state["classification"] == "transformable"


def test_apply_rejects_symlinked_managed_directory_without_external_write(tmp_path: Path) -> None:
    zcode_root = tmp_path / "zcode-cli"
    outside = tmp_path / "outside"
    zcode_root.mkdir()
    outside.mkdir()
    try:
        (zcode_root / "soullink").symlink_to(outside, target_is_directory=True)
    except OSError as exc:
        pytest.skip(f"directory symlinks unavailable: {exc}")

    with pytest.raises(RuntimeError, match="unsafe_managed_path:soullink"):
        ZCodeDeployment().apply(
            zcode_root,
            db_path=tmp_path / "runtime" / "pcltm.db",
            memfs_root=tmp_path / "runtime" / "memfs",
        )
    assert not (outside / "adapter.json").exists()


def test_apply_rejects_symlinked_managed_file_without_mutation(tmp_path: Path) -> None:
    zcode_root = tmp_path / "zcode-cli"
    outside = tmp_path / "outside-config.json"
    zcode_root.mkdir()
    outside.write_bytes(b'{"operator": "external"}')
    try:
        (zcode_root / "config.json").symlink_to(outside)
    except OSError as exc:
        pytest.skip(f"file symlinks unavailable: {exc}")
    with pytest.raises(RuntimeError, match="unsafe_managed_path:config.json"):
        ZCodeDeployment().apply(
            zcode_root,
            db_path=tmp_path / "runtime" / "pcltm.db",
            memfs_root=tmp_path / "runtime" / "memfs",
        )
    assert outside.read_bytes() == b'{"operator": "external"}'


def test_verify_rejects_tampered_mcp_server(tmp_path: Path) -> None:
    zcode_root = tmp_path / "zcode-cli"
    deployment = ZCodeDeployment()
    deployment.apply(
        zcode_root,
        db_path=tmp_path / "runtime" / "pcltm.db",
        memfs_root=tmp_path / "runtime" / "memfs",
    )
    config_path = zcode_root / "config.json"
    config = json.loads(config_path.read_text(encoding="utf-8"))
    config["mcp"]["servers"]["soullink"]["command"] = "foreign-python"
    config_path.write_text(json.dumps(config), encoding="utf-8")
    assert deployment.verify(zcode_root) is False


def test_verify_rejects_tampered_hook_command(tmp_path: Path) -> None:
    zcode_root = tmp_path / "zcode-cli"
    deployment = ZCodeDeployment()
    deployment.apply(
        zcode_root,
        db_path=tmp_path / "runtime" / "pcltm.db",
        memfs_root=tmp_path / "runtime" / "memfs",
    )
    config_path = zcode_root / "config.json"
    config = json.loads(config_path.read_text(encoding="utf-8"))
    config["hooks"]["events"]["SessionStart"][0]["hooks"][0]["command"] = "foreign-command"
    config_path.write_text(json.dumps(config), encoding="utf-8")
    assert deployment.verify(zcode_root) is False


def test_verify_rejects_disabled_hooks(tmp_path: Path) -> None:
    zcode_root = tmp_path / "zcode-cli"
    deployment = ZCodeDeployment()
    deployment.apply(
        zcode_root,
        db_path=tmp_path / "runtime" / "pcltm.db",
        memfs_root=tmp_path / "runtime" / "memfs",
    )
    config_path = zcode_root / "config.json"
    config = json.loads(config_path.read_text(encoding="utf-8"))
    config["hooks"]["enabled"] = False
    config_path.write_text(json.dumps(config), encoding="utf-8")
    assert deployment.verify(zcode_root) is False


def test_verify_rejects_removed_hook_events(tmp_path: Path) -> None:
    zcode_root = tmp_path / "zcode-cli"
    deployment = ZCodeDeployment()
    deployment.apply(
        zcode_root,
        db_path=tmp_path / "runtime" / "pcltm.db",
        memfs_root=tmp_path / "runtime" / "memfs",
    )
    config_path = zcode_root / "config.json"
    config = json.loads(config_path.read_text(encoding="utf-8"))
    del config["hooks"]["events"]["PostToolUse"]
    config_path.write_text(json.dumps(config), encoding="utf-8")
    assert deployment.verify(zcode_root) is False


def test_reapply_preserves_user_config_after_managed_sections(tmp_path: Path) -> None:
    zcode_root = tmp_path / "zcode-cli"
    deployment = ZCodeDeployment()
    receipt = deployment.apply(
        zcode_root,
        db_path=tmp_path / "runtime" / "pcltm.db",
        memfs_root=tmp_path / "runtime" / "memfs",
    )
    assert receipt is not None
    deployment.rollback(receipt)
    receipt = deployment.apply(
        zcode_root,
        db_path=tmp_path / "runtime" / "pcltm.db",
        memfs_root=tmp_path / "runtime" / "memfs",
    )
    assert receipt is not None
    config_path = zcode_root / "config.json"
    config = json.loads(config_path.read_text(encoding="utf-8"))
    config["custom"] = "preserved"
    config_path.write_text(json.dumps(config), encoding="utf-8")
    # Force the upgrade path while retaining the managed sections.
    (zcode_root / "soullink" / "adapter.json").unlink()
    deployment.apply(
        zcode_root,
        db_path=tmp_path / "runtime" / "pcltm.db",
        memfs_root=tmp_path / "runtime" / "memfs",
        receipt_path=tmp_path / "upgrade-receipt.json",
    )
    updated = json.loads(config_path.read_text(encoding="utf-8"))
    assert updated["custom"] == "preserved"
    assert len(updated["hooks"]["events"]["Stop"]) == 1


def test_receipt_body_cannot_redirect_rollback_unlink(tmp_path: Path) -> None:
    zcode_root = tmp_path / "zcode-cli"
    outside = tmp_path / "must-survive.txt"
    outside.write_text("safe", encoding="utf-8")
    receipt_path = tmp_path / "receipt.json"
    receipt = ZCodeDeployment().apply(
        zcode_root,
        db_path=tmp_path / "runtime" / "pcltm.db",
        memfs_root=tmp_path / "runtime" / "memfs",
        receipt_path=receipt_path,
    )
    assert receipt is not None
    payload = json.loads(receipt_path.read_text(encoding="utf-8"))
    payload["receipt_path"] = str(outside)
    receipt_path.write_text(json.dumps(payload), encoding="utf-8")
    loaded = ZCodeDeploymentReceipt.load(receipt_path)
    ZCodeDeployment().rollback(loaded)
    assert outside.read_text(encoding="utf-8") == "safe"
    assert not receipt_path.exists()


def test_apply_refuses_to_overwrite_existing_receipt(tmp_path: Path) -> None:
    receipt_path = tmp_path / "receipt.json"
    receipt_path.write_bytes(b"operator-owned")
    with pytest.raises(RuntimeError, match="already exists"):
        ZCodeDeployment().apply(
            tmp_path / "zcode-cli",
            db_path=tmp_path / "runtime" / "pcltm.db",
            memfs_root=tmp_path / "runtime" / "memfs",
            receipt_path=receipt_path,
        )
    assert receipt_path.read_bytes() == b"operator-owned"
    assert not (tmp_path / "zcode-cli").exists()


def test_receipt_replace_failure_removes_temp_and_restores(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    zcode_root = tmp_path / "zcode-cli"
    receipt_path = tmp_path / "receipt.json"
    before = _tree_digest(zcode_root)

    def fail_replace(source: Path, destination: Path) -> None:
        raise OSError("synthetic replace failure")

    monkeypatch.setattr("soul_link.zcode_deploy.os.replace", fail_replace)
    with pytest.raises(OSError, match="synthetic replace failure"):
        ZCodeDeployment().apply(
            zcode_root,
            db_path=tmp_path / "runtime" / "pcltm.db",
            memfs_root=tmp_path / "runtime" / "memfs",
            receipt_path=receipt_path,
        )
    assert _tree_digest(zcode_root) == before
    assert not receipt_path.exists()
    assert not receipt_path.with_name("receipt.json.tmp").exists()


def test_receipt_write_failure_restores_original_bytes(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    zcode_root = tmp_path / "zcode-cli"
    zcode_root.mkdir()
    (zcode_root / "config.json").write_bytes(b'{"custom": true}')
    before = _tree_digest(zcode_root)

    def fail_write(self: ZCodeDeploymentReceipt, path: Path) -> None:
        raise OSError("synthetic receipt failure")

    monkeypatch.setattr(ZCodeDeploymentReceipt, "write", fail_write)
    with pytest.raises(OSError, match="synthetic receipt failure"):
        ZCodeDeployment().apply(
            zcode_root,
            db_path=tmp_path / "runtime" / "pcltm.db",
            memfs_root=tmp_path / "runtime" / "memfs",
            receipt_path=tmp_path / "receipt.json",
        )
    assert _tree_digest(zcode_root) == before


def test_manage_agents_roundtrip_preserves_user_instructions(tmp_path: Path) -> None:
    zcode_root = tmp_path / "zcode-cli"
    zcode_root.mkdir()
    (zcode_root / "AGENTS.md").write_text("# 现有指令\n保留我\n", encoding="utf-8")

    deployment = ZCodeDeployment(manage_agents=True)
    receipt = deployment.apply(
        zcode_root,
        db_path=tmp_path / "runtime" / "pcltm.db",
        memfs_root=tmp_path / "runtime" / "memfs",
    )
    assert deployment.verify(zcode_root)
    text = (zcode_root / "AGENTS.md").read_text(encoding="utf-8")
    assert BEGIN in text
    assert "保留我" in text

    assert ZCodeDeployment(manage_agents=True).rollback(receipt)
    assert (zcode_root / "AGENTS.md").read_text(encoding="utf-8") == "# 现有指令\n保留我\n"


def test_rollback_rejects_manage_agents_mismatch(tmp_path: Path) -> None:
    zcode_root = tmp_path / "zcode-cli"
    receipt = ZCodeDeployment(manage_agents=True).apply(
        zcode_root,
        db_path=tmp_path / "runtime" / "pcltm.db",
        memfs_root=tmp_path / "runtime" / "memfs",
        receipt_path=tmp_path / "receipt.json",
    )
    assert receipt is not None
    with pytest.raises(RuntimeError, match="manage_agents"):
        ZCodeDeployment(manage_agents=False).rollback(receipt)
