from __future__ import annotations

from pathlib import Path
import re

import pytest

from soul_link.host_adaptation import CompatibilityManifest, HostAdapterController


def _manifest(tmp_path: Path) -> CompatibilityManifest:
    patch = tmp_path / "adapter.patch"
    patch.write_text("fake patch", encoding="utf-8")
    return CompatibilityManifest(
        host="hermes",
        adapter_version="1",
        required_paths=("agent/context_engine.py",),
        patch_path=patch,
    )


def test_detect_classifies_reverse_applicable_host_as_supported(tmp_path: Path) -> None:
    host = tmp_path / "host"
    (host / "agent").mkdir(parents=True)
    (host / "agent/context_engine.py").write_text("applied", encoding="utf-8")
    calls = []

    def run(command, cwd):
        calls.append((tuple(command), cwd))
        return 0 if "--reverse" in command else 1

    result = HostAdapterController(_manifest(tmp_path), command_runner=run).detect(host)

    assert result.classification == "supported"
    assert result.patch_state == "applied"
    assert result.missing_paths == ()
    assert calls[0][0][:4] == ("git", "apply", "--check", "--reverse")


def test_detect_classifies_forward_applicable_host_as_transformable(tmp_path: Path) -> None:
    host = tmp_path / "host"
    (host / "agent").mkdir(parents=True)
    (host / "agent/context_engine.py").write_text("base", encoding="utf-8")

    def run(command, cwd):
        return 1 if "--reverse" in command else 0

    result = HostAdapterController(_manifest(tmp_path), command_runner=run).detect(host)

    assert result.classification == "transformable"
    assert result.patch_state == "applicable"


def test_detect_fails_closed_for_missing_or_drifted_host(tmp_path: Path) -> None:
    host = tmp_path / "host"
    host.mkdir()
    controller = HostAdapterController(_manifest(tmp_path), command_runner=lambda *_: 1)

    missing = controller.detect(host)
    assert missing.classification == "incompatible"
    assert missing.missing_paths == ("agent/context_engine.py",)
    assert missing.patch_state == "not_checked"

    (host / "agent").mkdir()
    (host / "agent/context_engine.py").write_text("drifted", encoding="utf-8")
    drifted = controller.detect(host)
    assert drifted.classification == "incompatible"
    assert drifted.patch_state == "mismatch"


def test_verify_expands_python_placeholder_to_current_interpreter(tmp_path: Path) -> None:
    import sys

    host = tmp_path / "host"
    target = host / "agent/context_engine.py"
    target.parent.mkdir(parents=True)
    target.write_text("applied", encoding="utf-8")
    base = _manifest(tmp_path)
    manifest = CompatibilityManifest(
        host=base.host,
        adapter_version=base.adapter_version,
        required_paths=base.required_paths,
        patch_path=base.patch_path,
        verify_commands=(("{python}", "-m", "pytest", "-q"),),
    )
    calls = []

    def run(command, cwd):
        calls.append(tuple(command))
        return 0

    assert HostAdapterController(manifest, command_runner=run).verify(host) is True
    assert calls[-1][0] == sys.executable


def test_apply_backs_up_and_verify_failure_rolls_host_back(tmp_path: Path) -> None:
    host = tmp_path / "host"
    target = host / "agent/context_engine.py"
    target.parent.mkdir(parents=True)
    target.write_text("base", encoding="utf-8")
    commands = []

    def run(command, cwd):
        commands.append(tuple(command))
        if "--reverse" in command and "--check" in command:
            return 0 if target.read_text(encoding="utf-8") == "applied" else 1
        if tuple(command[:3]) == ("git", "apply", "--check"):
            return 0
        if tuple(command[:2]) == ("git", "apply"):
            target.write_text("applied", encoding="utf-8")
            return 0
        return 1

    controller = HostAdapterController(_manifest(tmp_path), command_runner=run)
    with pytest.raises(RuntimeError, match="verification failed"):
        controller.apply(host, verifier=lambda _: False)

    assert target.read_text(encoding="utf-8") == "base"
    assert not list(host.glob(".soullink-adapter-backup-*"))


    result, receipt = controller.apply(host, verifier=lambda _: True)

    assert result.classification == "supported"
    assert receipt.host_root == host.resolve()
    assert receipt.backup_path.is_dir()
    assert target.read_text(encoding="utf-8") == "applied"

    rolled_back = controller.rollback(receipt)

    assert rolled_back is True
    assert target.read_text(encoding="utf-8") == "base"
    assert not receipt.backup_path.exists()


def test_apply_rejects_incompatible_host_without_mutation(tmp_path: Path) -> None:
    host = tmp_path / "host"
    host.mkdir()
    controller = HostAdapterController(_manifest(tmp_path), command_runner=lambda *_: 1)

    with pytest.raises(RuntimeError, match="incompatible"):
        controller.apply(host, verifier=lambda _: True)

    assert list(host.iterdir()) == []


def test_rollback_removes_files_created_by_patch(tmp_path: Path) -> None:
    host = tmp_path / "host"
    target = host / "agent/context_engine.py"
    target.parent.mkdir(parents=True)
    target.write_text("applied", encoding="utf-8")
    created = host / "tests/new_adapter_test.py"
    created.parent.mkdir(parents=True)
    created.write_text("new", encoding="utf-8")
    backup = host / ".soullink-adapter-backup-test"
    saved = backup / "agent/context_engine.py"
    saved.parent.mkdir(parents=True)
    saved.write_text("base", encoding="utf-8")
    (backup / ".soullink-backup.json").write_text(
        '{"host_root": "' + str(host.resolve()).replace("\\", "\\\\") + '", "adapter_version": "1"}',
        encoding="utf-8",
    )
    from soul_link.host_adaptation import AdaptationReceipt

    base_manifest = _manifest(tmp_path)
    manifest = CompatibilityManifest(
        host=base_manifest.host,
        adapter_version=base_manifest.adapter_version,
        required_paths=base_manifest.required_paths,
        patch_path=base_manifest.patch_path,
        created_paths=("tests/new_adapter_test.py",),
    )
    controller = HostAdapterController(manifest, command_runner=lambda *_: 0)
    receipt = AdaptationReceipt(host.resolve(), backup.resolve(), "1")

    assert controller.rollback(receipt) is True
    assert target.read_text(encoding="utf-8") == "base"
    assert not created.exists()


def test_rollback_rejects_incomplete_backup_without_mutation_or_deletion(tmp_path: Path) -> None:
    host = tmp_path / "host"
    first = host / "agent/context_engine.py"
    second = host / "agent/other.py"
    first.parent.mkdir(parents=True)
    first.write_text("applied-first", encoding="utf-8")
    second.write_text("applied-second", encoding="utf-8")
    backup = host / ".soullink-adapter-backup-test"
    saved_first = backup / "agent/context_engine.py"
    saved_first.parent.mkdir(parents=True)
    saved_first.write_text("base-first", encoding="utf-8")
    (backup / ".soullink-backup.json").write_text(
        '{"host_root": "' + str(host.resolve()).replace("\\", "\\\\") + '", "adapter_version": "1"}',
        encoding="utf-8",
    )
    from soul_link.host_adaptation import AdaptationReceipt

    base = _manifest(tmp_path)
    manifest = CompatibilityManifest(
        host=base.host,
        adapter_version=base.adapter_version,
        required_paths=("agent/context_engine.py", "agent/other.py"),
        patch_path=base.patch_path,
    )
    receipt = AdaptationReceipt(host.resolve(), backup.resolve(), "1")

    with pytest.raises(RuntimeError, match="incomplete"):
        HostAdapterController(manifest, command_runner=lambda *_: 0).rollback(receipt)

    assert first.read_text(encoding="utf-8") == "applied-first"
    assert second.read_text(encoding="utf-8") == "applied-second"
    assert backup.is_dir()
    assert saved_first.read_text(encoding="utf-8") == "base-first"


def test_rollback_rejects_receipt_for_different_adapter_version(tmp_path: Path) -> None:
    host = tmp_path / "host"
    target = host / "agent/context_engine.py"
    target.parent.mkdir(parents=True)
    target.write_text("applied", encoding="utf-8")
    backup = host / ".soullink-adapter-backup-test"
    saved = backup / "agent/context_engine.py"
    saved.parent.mkdir(parents=True)
    saved.write_text("base", encoding="utf-8")
    from soul_link.host_adaptation import AdaptationReceipt

    receipt = AdaptationReceipt(host.resolve(), backup.resolve(), "other-version")
    controller = HostAdapterController(_manifest(tmp_path), command_runner=lambda *_: 0)

    with pytest.raises(RuntimeError, match="version"):
        controller.rollback(receipt)

    assert target.read_text(encoding="utf-8") == "applied"
    assert backup.is_dir()


def test_repository_hermes_manifest_loads_patch_and_all_patch_targets() -> None:
    root = Path(__file__).resolve().parents[2]
    manifest = CompatibilityManifest.load(root / "adapters/hermes/compatibility.yaml")

    assert manifest.host == "hermes-agent"
    assert manifest.patch_path.is_file()
    patch_text = manifest.patch_path.read_text(encoding="utf-8")
    assert manifest.required_paths
    patch_targets = manifest.required_paths + manifest.created_paths
    assert all(f"diff --git a/{path} b/{path}" in patch_text for path in patch_targets)
    assert manifest.created_paths == (
        "tests/agent/test_context_request_budget.py",
        "tests/agent/test_pcltm_context_budget.py",
    )


def test_repository_soullink_runtime_manifest_owns_every_live_host_delta() -> None:
    root = Path(__file__).resolve().parents[2]
    manifest = CompatibilityManifest.load(
        root / "adapters/hermes/compatibility-soullink-runtime.yaml"
    )

    assert manifest.host == "hermes-agent"
    assert manifest.adapter_version == "soullink-runtime-v4-upstream-01a1037-final-forward"
    patch_text = manifest.patch_path.read_text(encoding="utf-8")
    patch_targets = tuple(re.findall(r"^\+\+\+ b/(.+)$", patch_text, re.MULTILINE))
    assert set(patch_targets) == set(manifest.required_paths + manifest.created_paths)
    assert "agent/memory_authority.py" in manifest.created_paths
    assert "tests/agent/test_exclusive_memory_authority_contract.py" in manifest.created_paths
    assert "tests/agent/test_soullink_memory_authority_adapter.py" in manifest.created_paths
    assert "agent/memory_manager.py" in manifest.required_paths
    assert "agent/conversation_loop.py" in manifest.required_paths
    assert "tests/agent/test_memory_final_forward_observer.py" in manifest.created_paths
    assert "_final_forward_messages" in patch_text
    assert "agent/verification_stop.py" in manifest.required_paths
    assert "hermes_cli/subcommands/memory.py" in manifest.required_paths


def test_repository_context_budget_propagation_manifest_is_bounded() -> None:

    root = Path(__file__).resolve().parents[2]
    manifest = CompatibilityManifest.load(
        root / "adapters/hermes/compatibility-context-budget-propagation.yaml"
    )

    assert manifest.host == "hermes-agent"
    assert manifest.required_paths == (
        "agent/agent_init.py",
        "tests/run_agent/test_plugin_context_engine_init.py",
    )
    assert manifest.created_paths == ()
    patch_text = manifest.patch_path.read_text(encoding="utf-8")
    assert all(
        f"diff --git a/{path} b/{path}" in patch_text
        for path in manifest.required_paths
    )
    assert "_configure_context_engine(_ctx_cfg)" in patch_text
    assert "test_plugin_engine_gets_context_config_before_model_metadata" in patch_text


def test_repository_model_router_manifest_owns_detached_turn_overrides() -> None:
    root = Path(__file__).resolve().parents[2]
    manifest = CompatibilityManifest.load(
        root / "adapters/hermes/compatibility-model-router-v3.yaml"
    )

    assert manifest.host == "hermes-agent"
    assert "tests/run_agent/test_run_agent.py" in manifest.required_paths
    patch_text = manifest.patch_path.read_text(encoding="utf-8")
    patch_targets = tuple(re.findall(r"^\+\+\+ b/(.+)$", patch_text, re.MULTILINE))
    assert set(patch_targets) == set(manifest.required_paths + manifest.created_paths)
    assert "copy.deepcopy(agent.request_overrides or {})" in patch_text
    assert "test_current_turn_request_overrides_do_not_mutate_agent_config" in patch_text


def test_repository_verification_guidance_manifest_is_bounded() -> None:
    root = Path(__file__).resolve().parents[2]
    manifest = CompatibilityManifest.load(
        root / "adapters/hermes/compatibility-verification-guidance.yaml"
    )

    assert manifest.host == "hermes-agent"
    assert manifest.required_paths == (
        "agent/verification_evidence.py",
        "agent/verification_stop.py",
        "tests/agent/test_verification_evidence.py",
        "tests/agent/test_verification_stop.py",
    )
    assert manifest.created_paths == ()
    patch_text = manifest.patch_path.read_text(encoding="utf-8")
    assert all(
        f"diff --git a/{path} b/{path}" in patch_text
        for path in manifest.required_paths
    )
    assert "_verification_scope_instruction" in patch_text
    assert "test_quoted_windows_temp_script_records_ad_hoc_evidence" in patch_text


def test_detect_rejects_required_path_through_symlink_escape(tmp_path: Path) -> None:
    host = tmp_path / "host"
    outside = tmp_path / "outside"
    host.mkdir()
    outside.mkdir()
    (outside / "context_engine.py").write_text("external", encoding="utf-8")
    try:
        (host / "agent").symlink_to(outside, target_is_directory=True)
    except OSError:
        pytest.skip("directory symlinks unavailable")

    controller = HostAdapterController(_manifest(tmp_path), command_runner=lambda *_: 0)

    with pytest.raises(RuntimeError, match="escapes host root"):
        controller.detect(host)
    assert (outside / "context_engine.py").read_text(encoding="utf-8") == "external"


def test_rollback_rejects_forged_sibling_backup_directory(tmp_path: Path) -> None:
    from soul_link.host_adaptation import AdaptationReceipt

    host = tmp_path / "host"
    target = host / "agent/context_engine.py"
    target.parent.mkdir(parents=True)
    target.write_text("applied", encoding="utf-8")
    forged = host / "forged"
    saved = forged / "agent/context_engine.py"
    saved.parent.mkdir(parents=True)
    saved.write_text("attacker", encoding="utf-8")
    receipt = AdaptationReceipt(host.resolve(), forged.resolve(), "1")

    with pytest.raises(RuntimeError, match="backup"):
        HostAdapterController(_manifest(tmp_path), command_runner=lambda *_: 0).rollback(receipt)

    assert target.read_text(encoding="utf-8") == "applied"


def test_manifest_rejects_unsafe_or_overlapping_host_paths(tmp_path: Path) -> None:
    patch = (tmp_path / "x.patch").resolve()
    patch.write_text("patch", encoding="utf-8")

    for unsafe in ("../outside.py", "/absolute.py", "C:/absolute.py", ""):
        with pytest.raises(ValueError, match="host path"):
            CompatibilityManifest(
                host="hermes",
                adapter_version="1",
                required_paths=(unsafe,),
                patch_path=patch,
            )

    with pytest.raises(ValueError, match="overlap"):
        CompatibilityManifest(
            host="hermes",
            adapter_version="1",
            required_paths=("agent/context.py",),
            created_paths=("agent/context.py",),
            patch_path=patch,
        )


def test_manifest_rejects_relative_patch_path() -> None:
    with pytest.raises(ValueError, match="absolute"):
        CompatibilityManifest(host="hermes", adapter_version="1", required_paths=(), patch_path=Path("x.patch"))
