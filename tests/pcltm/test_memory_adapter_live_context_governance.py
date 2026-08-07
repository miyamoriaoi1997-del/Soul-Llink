from __future__ import annotations

from pcltm import memory_adapter


def test_retired_direct_prompt_path_never_injects_legacy_records() -> None:
    rendered = memory_adapter.load_prompt_context(
        mode="work",
        query="PCLTM context budget diagnostics",
        memory_limit=20000,
        user_limit=2000,
    )
    assert rendered == ""
    telemetry = memory_adapter.last_live_context_telemetry()
    assert telemetry == {
        "status": "retired",
        "reason": "legacy_direct_db_prompt_retired",
    }


def test_retired_direct_prompt_observation_is_typed_and_bodyless() -> None:
    memory_adapter.load_prompt_context(mode="daily", query="private legacy body")
    observation = memory_adapter.last_memory_selection_observation()
    assert observation == {
        "status": "retired",
        "authority": "pcltm.memory_current",
        "reason": "legacy_direct_db_prompt_retired",
        "selected_count": 0,
        "selected_records": [],
    }
    assert "private legacy body" not in str(observation)
