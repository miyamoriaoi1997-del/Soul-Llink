from __future__ import annotations

import os
import shutil
import subprocess
import sys
from pathlib import Path

import pytest

from soul_link.host_adaptation import CompatibilityManifest

REPO_ROOT = Path(__file__).resolve().parents[1]
HOST_ROOT_VALUE = os.environ.get("HERMES_HOST_ROOT")
HOST_ROOT = Path(HOST_ROOT_VALUE) if HOST_ROOT_VALUE else None
MANIFEST_PATH = REPO_ROOT / "adapters/hermes/compatibility-memory-authority.yaml"


def _run(command: list[str], *, cwd: Path, timeout: int = 180) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        command,
        cwd=cwd,
        check=False,
        text=True,
        capture_output=True,
        timeout=timeout,
    )


def _materialize_host(tmp_path: Path, manifest: CompatibilityManifest) -> Path:
    host = tmp_path / "host"
    clone = _run(
        ["git", "clone", "--quiet", "--shared", "--no-checkout", str(HOST_ROOT), str(host)],
        cwd=tmp_path,
    )
    assert clone.returncode == 0, clone.stdout + clone.stderr

    # The adapter import path traverses ordinary host packages. A file-pattern
    # sparse checkout is unreliable with the real shallow Windows checkout, so
    # materialize the full tracked host tree in the isolated clone instead.
    checkout = _run(["git", "checkout", "--quiet", "HEAD"], cwd=host)
    assert checkout.returncode == 0, checkout.stdout + checkout.stderr
    for created_path in manifest.created_paths:
        candidate = host / created_path
        if candidate.is_file():
            candidate.unlink()
    return host


def test_real_host_patch_apply_and_memory_authority_contract(tmp_path: Path) -> None:
    if HOST_ROOT is None:
        pytest.skip("set HERMES_HOST_ROOT to run real-host adapter tests")
    if not HOST_ROOT.is_dir():
        pytest.fail(f"real Hermes host checkout missing: {HOST_ROOT}")

    real_host_status_before = _run(
        ["git", "status", "--short", "--untracked-files=all"],
        cwd=HOST_ROOT,
    )
    assert real_host_status_before.returncode == 0, (
        real_host_status_before.stdout + real_host_status_before.stderr
    )

    manifest = CompatibilityManifest.load(MANIFEST_PATH)
    assert manifest.adapter_version == "memory-authority-v2-generic-exclusive-provider"
    assert manifest.patch_path.is_file()
    assert set(manifest.required_paths) == {
        "model_tools.py",
        "agent/agent_init.py",
        "agent/tool_executor.py",
        "agent/agent_runtime_helpers.py",
        "hermes_cli/memory_setup.py",
        "hermes_cli/subcommands/memory.py",
    }
    assert set(manifest.created_paths) == {
        "agent/memory_authority.py",
        "tests/agent/test_exclusive_memory_authority_contract.py",
        "tests/agent/test_soullink_memory_authority_adapter.py",
    }

    host = _materialize_host(tmp_path, manifest)
    patch = str(manifest.patch_path)

    check = _run(["git", "apply", "--check", patch], cwd=host)
    assert check.returncode == 0, check.stdout + check.stderr
    apply_result = _run(["git", "apply", patch], cwd=host)
    assert apply_result.returncode == 0, apply_result.stdout + apply_result.stderr

    test_env = dict(os.environ)
    test_env["PYTHONPATH"] = str(host)
    test_result = subprocess.run(
        [sys.executable, "-m", "pytest", "tests/agent/test_soullink_memory_authority_adapter.py", "-q"],
        cwd=host,
        env=test_env,
        check=False,
        text=True,
        capture_output=True,
        timeout=300,
    )
    assert test_result.returncode == 0, test_result.stdout + test_result.stderr

    reverse_check = _run(["git", "apply", "--check", "--reverse", patch], cwd=host)
    assert reverse_check.returncode == 0, reverse_check.stdout + reverse_check.stderr

    changed = _run(["git", "status", "--short", "--untracked-files=all"], cwd=host)
    assert changed.returncode == 0, changed.stdout + changed.stderr
    changed_paths = {
        line[3:].strip()
        for line in changed.stdout.splitlines()
        if line and not line[3:].strip().endswith((".pyc", ".pyo"))
        and "__pycache__/" not in line[3:].replace("\\", "/")
        and ".pytest_cache/" not in line[3:].replace("\\", "/")
    }
    assert changed_paths == set(manifest.required_paths) | set(manifest.created_paths)

    real_host_status_after = _run(
        ["git", "status", "--short", "--untracked-files=all"],
        cwd=HOST_ROOT,
    )
    assert real_host_status_after.returncode == 0, (
        real_host_status_after.stdout + real_host_status_after.stderr
    )
    assert real_host_status_after.stdout == real_host_status_before.stdout
