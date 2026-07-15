from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path


def _run(*args: str) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [sys.executable, "-m", "soul_link.host_adaptation", *args],
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        check=False,
    )


def _write_fixture(tmp_path: Path) -> tuple[Path, Path, Path, Path]:
    host = tmp_path / "host"
    target = host / "agent/context_engine.py"
    target.parent.mkdir(parents=True)
    target.write_text("base\n", encoding="utf-8")
    patch = tmp_path / "adapter.patch"
    patch.write_text(
        "diff --git a/agent/context_engine.py b/agent/context_engine.py\n"
        "--- a/agent/context_engine.py\n"
        "+++ b/agent/context_engine.py\n"
        "@@ -1 +1 @@\n"
        "-base\n"
        "+applied\n",
        encoding="utf-8",
    )
    manifest = tmp_path / "compatibility.yaml"
    manifest.write_text(
        "host: fake\n"
        "adapter_version: '1'\n"
        "patch: adapter.patch\n"
        "required_paths:\n"
        "  - agent/context_engine.py\n"
        "verify_commands:\n"
        "  - [python, -c, \"from pathlib import Path; raise SystemExit(0 if Path('agent/context_engine.py').read_text().strip() == 'applied' else 1)\"]\n",
        encoding="utf-8",
    )
    return host, target, manifest, tmp_path / "receipt.json"


def test_host_adaptation_cli_detect_emits_json_without_mutation(tmp_path: Path) -> None:
    host = tmp_path / "host"
    target = host / "agent/context_engine.py"
    target.parent.mkdir(parents=True)
    target.write_text("base", encoding="utf-8")
    patch = tmp_path / "adapter.patch"
    patch.write_text("not applicable", encoding="utf-8")
    manifest = tmp_path / "compatibility.yaml"
    manifest.write_text(
        "host: fake\nadapter_version: '1'\npatch: adapter.patch\nrequired_paths:\n  - agent/context_engine.py\n",
        encoding="utf-8",
    )

    proc = _run("detect", "--manifest", str(manifest), "--host-root", str(host))

    assert proc.returncode == 2
    payload = json.loads(proc.stdout)
    assert payload["classification"] == "incompatible"
    assert payload["patch_state"] == "mismatch"
    assert target.read_text(encoding="utf-8") == "base"


def test_host_adaptation_cli_receipt_write_failure_rolls_back_host(tmp_path: Path) -> None:
    host, target, manifest, receipt = _write_fixture(tmp_path)
    receipt.mkdir()

    applied = _run(
        "apply", "--manifest", str(manifest), "--host-root", str(host), "--receipt", str(receipt)
    )

    assert applied.returncode != 0
    assert json.loads(applied.stdout)["success"] is False
    assert target.read_text(encoding="utf-8") == "base\n"
    assert not list(host.glob(".soullink-adapter-backup-*"))
    assert not receipt.with_name(receipt.name + ".tmp").exists()


def test_host_adaptation_cli_apply_verify_and_rollback_across_processes(tmp_path: Path) -> None:
    host, target, manifest, receipt = _write_fixture(tmp_path)

    applied = _run(
        "apply", "--manifest", str(manifest), "--host-root", str(host), "--receipt", str(receipt)
    )
    assert applied.returncode == 0, applied.stderr
    assert json.loads(applied.stdout)["classification"] == "supported"
    assert target.read_text(encoding="utf-8") == "applied\n"
    assert receipt.is_file()

    verified = _run("verify", "--manifest", str(manifest), "--host-root", str(host))
    assert verified.returncode == 0, verified.stderr
    assert json.loads(verified.stdout)["verified"] is True

    rolled_back = _run("rollback", "--manifest", str(manifest), "--receipt", str(receipt))
    assert rolled_back.returncode == 0, rolled_back.stderr
    assert json.loads(rolled_back.stdout)["rolled_back"] is True
    assert target.read_text(encoding="utf-8") == "base\n"
    assert not receipt.exists()
