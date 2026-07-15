from persona_orchestrator import (
    MODE_CONFLICT,
    MODE_CREATIVE,
    MODE_DAILY,
    MODE_INTIMACY,
    MODE_SEX,
    MODE_SEX_CANDIDATE,
    MODE_SYSTEM_MAINTENANCE,
    MODE_WORK,
)
from persona_orchestrator.memory_selector import MemorySelector


def test_daily_selects_core_relationship_with_memory():
    selection = MemorySelector().select(MODE_DAILY)

    assert selection.profile == "core_relationship"
    assert selection.candidate_files == ["MEMORY.md", "USER.md", "STATE.md"]
    legacy_domain = ''.join(['MO', 'MENTS', '.md'])
    assert legacy_domain not in selection.candidate_files


def test_work_selects_technical_plus_core_relationship_with_memory():
    selection = MemorySelector().select(MODE_WORK)

    assert selection.profile == "technical_plus_core_relationship"
    assert selection.candidate_files == ["MEMORY.md", "USER.md", "STATE.md"]
    legacy_domain = ''.join(['MO', 'MENTS', '.md'])
    assert legacy_domain not in selection.candidate_files


def test_sex_selects_desire_gated_relationship_profile_with_memory():
    selection = MemorySelector().select(MODE_SEX)

    assert selection.profile == "relationship_preferences_desire_gated"
    assert selection.candidate_files == ["MEMORY.md", "USER.md", "STATE.md"]
    legacy_domain = ''.join(['MO', 'MENTS', '.md'])
    assert legacy_domain not in selection.candidate_files


def test_legacy_aliases_normalize_to_three_state_memory_profiles():
    selector = MemorySelector()

    daily_aliases = [MODE_CONFLICT, MODE_INTIMACY]
    work_aliases = [MODE_CREATIVE, MODE_SYSTEM_MAINTENANCE]
    sex_aliases = [MODE_SEX_CANDIDATE]

    for mode in daily_aliases:
        selection = selector.select(mode)
        assert selection.profile == "core_relationship"
        assert selection.candidate_files == ["MEMORY.md", "USER.md", "STATE.md"]

    for mode in work_aliases:
        selection = selector.select(mode)
        assert selection.profile == "technical_plus_core_relationship"
        assert selection.candidate_files == ["MEMORY.md", "USER.md", "STATE.md"]

    for mode in sex_aliases:
        selection = selector.select(mode)
        assert selection.profile == "relationship_preferences_desire_gated"
        assert selection.candidate_files == ["MEMORY.md", "USER.md", "STATE.md"]


def test_unknown_mode_falls_back_to_daily_memory_profile_with_flag():
    selection = MemorySelector().select("unknown", ["crisis_guard"])

    assert selection.profile == "core_relationship"
    assert selection.candidate_files == ["MEMORY.md", "USER.md", "STATE.md"]
    assert selection.safety_flags == ["crisis_guard", "unknown_mode_fallback"]


def test_work_mode_has_all_layers():
    selection = MemorySelector().select(MODE_WORK)

    assert selection.layers == ["system", "pinned", "episodic", "transient"]


def test_work_mode_has_runtime_buckets():
    selection = MemorySelector().select(MODE_WORK)

    assert "runtime_boundary" in selection.buckets
    assert "project_path" in selection.buckets


def test_sex_mode_excludes_runtime_buckets():
    selection = MemorySelector().select(MODE_SEX)

    assert "runtime_boundary" not in selection.buckets
    assert "project_path" not in selection.buckets
    assert "current_task" not in selection.buckets


def test_sex_mode_excludes_transient():
    selection = MemorySelector().select(MODE_SEX)

    assert "transient" not in selection.layers


def test_daily_mode_has_relationship_buckets():
    selection = MemorySelector().select(MODE_DAILY)

    assert "relationship" in selection.buckets
    assert "user_preference" in selection.buckets


def test_daily_mode_excludes_investment():
    selection = MemorySelector().select(MODE_DAILY)

    assert "investment" not in selection.buckets


def test_unknown_mode_falls_back_to_daily():
    selection = MemorySelector().select("invalid-mode")

    assert selection.profile == "core_relationship"
    assert selection.layers == ["system", "pinned", "episodic"]
    assert selection.buckets == ["relationship", "user_preference", "emotion_boundary"]


def test_backward_compat_candidate_files_still_set():
    selector = MemorySelector()

    assert selector.select(MODE_DAILY).candidate_files == ["MEMORY.md", "USER.md", "STATE.md"]
    assert selector.select(MODE_WORK).candidate_files == ["MEMORY.md", "USER.md", "STATE.md"]
    assert selector.select(MODE_SEX).candidate_files == ["MEMORY.md", "USER.md", "STATE.md"]


def test_stateful_active_and_archival_contracts():
    work = MemorySelector().select(MODE_WORK)
    daily = MemorySelector().select(MODE_DAILY)
    sex = MemorySelector().select(MODE_SEX)

    assert work.active_layer_contract() == ["system", "pinned", "transient"]
    assert work.archival_layer_contract() == ["episodic"]
    assert "episodic" in work.contract_summary()["reference_only_layers"]
    assert daily.active_layer_contract() == ["system", "pinned"]
    assert daily.archival_layer_contract() == ["episodic"]
    assert sex.active_layer_contract() == ["system", "pinned"]
    assert sex.archival_layer_contract() == []
