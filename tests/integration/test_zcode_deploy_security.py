from __future__ import annotations

import json
import os
import subprocess
from pathlib import Path

import pytest

from soul_link.zcode_deploy import ZCodeDeployment, ZCodeDeploymentReceipt, main


def _junction(link, target) -> None:
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


def _apply(tmp_path, *, manage_agents: bool = False, receipt_path=None) -> tuple[ZCodeDeployment, ZCodeDeploymentReceipt]:
    deployment = ZCodeDeployment(manage_agents=manage_agents)
    receipt = deployment.apply(
        tmp_path / "zcode-cli",
        db_path=tmp_path / "runtime/pcltm.db",
        memfs_root=tmp_path / "runtime/memfs",
        receipt_path=receipt_path or tmp_path / "receipt.json",
    )
    assert receipt is not None
    return deployment, receipt


def test_apply_rejects_symlinked_zcode_root_without_external_write(tmp_path: Path) -> None:
    outside = tmp_path / "outside-home"
    outside.mkdir()
    linked_root = tmp_path / "zcode-cli"
    try:
        linked_root.symlink_to(outside, target_is_directory=True)
    except OSError as exc:
        pytest.skip(f"directory symlinks unavailable: {exc}")
    with pytest.raises(RuntimeError, match="symlink|reparse|unsafe_managed_path"):
        ZCodeDeployment().apply(
            linked_root, db_path=tmp_path / "runtime/pcltm.db", memfs_root=tmp_path / "runtime/memfs"
        )
    assert not (outside / "config.json").exists()


@pytest.mark.parametrize("mutation", ["env", "timeout", "approval", "command"])
def test_verify_rejects_tampered_mcp_policy(tmp_path: Path, mutation: str) -> None:
    deployment, _ = _apply(tmp_path)
    config_path = tmp_path / "zcode-cli" / "config.json"
    config = json.loads(config_path.read_text(encoding="utf-8"))
    server = config["mcp"]["servers"]["soullink"]
    if mutation == "env":
        server["env"]["HERMES_PCLTM_DB"] = "FOREIGN_DB"
    elif mutation == "timeout":
        server["args"] = ["-m", "soul_link.other_mcp"]
    elif mutation == "approval":
        server["type"] = "http"
    else:
        server["command"] = "foreign-python"
    config_path.write_text(json.dumps(config), encoding="utf-8")
    assert deployment.verify(tmp_path / "zcode-cli") is False


def test_verify_returns_false_for_malformed_hook_schema(tmp_path: Path) -> None:
    deployment, _ = _apply(tmp_path)
    config_path = tmp_path / "zcode-cli" / "config.json"
    config = json.loads(config_path.read_text(encoding="utf-8"))
    config["hooks"]["events"]["SessionStart"] = ["invalid"]
    config_path.write_text(json.dumps(config), encoding="utf-8")
    assert deployment.verify(tmp_path / "zcode-cli") is False


def test_receipt_symlink_cannot_delete_external_target(tmp_path: Path) -> None:
    _, receipt = _apply(tmp_path)
    external = tmp_path / "external-receipt.json"
    receipt.receipt_path.replace(external)
    try:
        receipt.receipt_path.symlink_to(external)
    except OSError as exc:
        pytest.skip(f"file symlinks unavailable: {exc}")
    with pytest.raises(RuntimeError, match="symlink|reparse"):
        ZCodeDeploymentReceipt.load(receipt.receipt_path)
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
            "--zcode-root", str(tmp_path / "zcode-cli"),
            "--db", str(tmp_path / "runtime/pcltm.db"),
            "--memfs", str(tmp_path / "runtime/memfs"),
            "--receipt", str(receipt_path),
        ])
    assert outside.read_bytes() == b"operator-owned"
    assert not (tmp_path / "zcode-cli").exists()


def test_rollback_rejects_incomplete_managed_set(tmp_path: Path) -> None:
    deployment, receipt = _apply(tmp_path)
    marker_path = receipt.backup_path / ".soullink-zcode-deploy.json"
    marker = json.loads(marker_path.read_text(encoding="utf-8"))
    marker["entries"] = {}
    marker["fingerprints"] = {}
    marker_path.write_text(json.dumps(marker), encoding="utf-8")
    forged = ZCodeDeploymentReceipt(
        receipt.zcode_root, receipt.backup_path, receipt.receipt_path,
        receipt.adapter_version, receipt.manage_agents, {}, {},
    )
    with pytest.raises(RuntimeError, match="managed entries"):
        deployment.rollback(forged)


@pytest.mark.parametrize("relative", ["config.json", "soullink/adapter.json"])
def test_apply_rejects_receipt_overlapping_managed_runtime(tmp_path: Path, relative: str) -> None:
    zcode_root = tmp_path / "zcode-cli"
    with pytest.raises(RuntimeError, match="overlaps managed runtime"):
        ZCodeDeployment().apply(
            zcode_root,
            db_path=tmp_path / "runtime/pcltm.db",
            memfs_root=tmp_path / "runtime/memfs",
            receipt_path=zcode_root / relative,
        )


def test_receipt_cannot_redirect_rollback_to_another_deployment(tmp_path: Path) -> None:
    first_path = tmp_path / "first-receipt.json"
    second_path = tmp_path / "second-receipt.json"
    first = ZCodeDeployment().apply(
        tmp_path / "first", db_path=tmp_path / "runtime/first.db",
        memfs_root=tmp_path / "runtime/first", receipt_path=first_path,
    )
    second = ZCodeDeployment().apply(
        tmp_path / "second", db_path=tmp_path / "runtime/second.db",
        memfs_root=tmp_path / "runtime/second", receipt_path=second_path,
    )
    assert first is not None and second is not None
    first_path.write_bytes(second_path.read_bytes())
    redirected = ZCodeDeploymentReceipt.load(first_path)
    with pytest.raises(RuntimeError, match="receipt path does not match backup"):
        ZCodeDeployment().rollback(redirected)
    assert ZCodeDeployment().verify(tmp_path / "first")
    assert ZCodeDeployment().verify(tmp_path / "second")


@pytest.mark.parametrize("layer", ["home", "ancestor", "managed_child", "receipt_parent"])
def test_apply_rejects_native_windows_junctions(tmp_path: Path, layer: str) -> None:
    outside = tmp_path / f"outside-{layer}"
    real_home = tmp_path / "real-home"
    zcode_root = real_home
    receipt_path = tmp_path / "receipt.json"

    if layer == "home":
        zcode_root = tmp_path / "linked-home"
        _junction(zcode_root, outside)
    elif layer == "ancestor":
        ancestor = tmp_path / "linked-ancestor"
        _junction(ancestor, outside)
        zcode_root = ancestor / "zcode-cli"
    elif layer == "managed_child":
        real_home.mkdir()
        _junction(real_home / "soullink", outside)
    else:
        receipt_parent = tmp_path / "linked-receipts"
        _junction(receipt_parent, outside)
        receipt_path = receipt_parent / "receipt.json"

    with pytest.raises(RuntimeError, match="symlink|reparse|unsafe_managed_path"):
        ZCodeDeployment().apply(
            zcode_root,
            db_path=tmp_path / "runtime/pcltm.db",
            memfs_root=tmp_path / "runtime/memfs",
            receipt_path=receipt_path,
        )
    assert not (outside / "adapter.json").exists()
    assert not (outside / "receipt.json").exists()


def test_cli_detect_emits_json_without_mutation(tmp_path: Path, capsys) -> None:
    zcode_root = tmp_path / "zcode-cli"
    before = sorted(p.name for p in zcode_root.rglob("*")) if zcode_root.exists() else []
    assert main(["detect", "--zcode-root", str(zcode_root)]) == 0
    payload = json.loads(capsys.readouterr().out)
    assert payload["classification"] == "transformable"
    after = sorted(p.name for p in zcode_root.rglob("*")) if zcode_root.exists() else []
    assert after == before


def test_apply_verify_rollback_across_cli_processes(tmp_path: Path) -> None:
    zcode_root = tmp_path / "zcode-cli"
    zcode_root.mkdir()
    original = b'{"custom": true}'
    (zcode_root / "config.json").write_bytes(original)
    receipt_path = tmp_path / "receipt.json"

    import sys

    env = dict(os.environ)
    env["PYTHONPATH"] = os.pathsep.join([".", "packages"])
    applied = subprocess.run(
        [sys.executable, "-m", "soul_link.zcode_deploy", "apply",
         "--zcode-root", str(zcode_root), "--db", str(tmp_path / "runtime/pcltm.db"),
         "--memfs", str(tmp_path / "runtime/memfs"), "--receipt", str(receipt_path)],
        capture_output=True, text=True, env=env, cwd=str(Path(__file__).resolve().parents[2]),
    )
    assert applied.returncode == 0, applied.stderr
    verified = subprocess.run(
        [sys.executable, "-m", "soul_link.zcode_deploy", "verify", "--zcode-root", str(zcode_root)],
        capture_output=True, text=True, env=env, cwd=str(Path(__file__).resolve().parents[2]),
    )
    assert json.loads(verified.stdout)["verified"] is True
    rolled = subprocess.run(
        [sys.executable, "-m", "soul_link.zcode_deploy", "rollback",
         "--zcode-root", str(zcode_root), "--receipt", str(receipt_path)],
        capture_output=True, text=True, env=env, cwd=str(Path(__file__).resolve().parents[2]),
    )
    assert rolled.returncode == 0, rolled.stderr
    assert (zcode_root / "config.json").read_bytes() == original
    assert not receipt_path.exists()
