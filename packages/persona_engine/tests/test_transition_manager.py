from persona_orchestrator import (
    MODE_DAILY,
    MODE_SEX,
    MODE_WORK,
    ModeDecision,
    StateOrchestrator,
)
from persona_orchestrator.transition_manager import TransitionManager


def decision(mode, confidence=0.9, flags=None, signals=None, submode=""):
    return ModeDecision(mode=mode, submode=submode, confidence=confidence, reason="test", safety_flags=flags or [], signals=signals or {})


def test_start_transition_accepts_requested_mode():
    result = TransitionManager().transition(None, decision(MODE_DAILY))

    assert result.active_mode == MODE_DAILY
    assert result.transition == "start:daily"


def test_daily_to_work_accepted():
    result = TransitionManager().transition(MODE_DAILY, decision(MODE_WORK, 0.88))

    assert result.active_mode == MODE_WORK
    assert result.transition == "daily->work"


def test_daily_to_work_from_relationship_alias_context_accepted():
    result = TransitionManager().transition(MODE_DAILY, decision(MODE_WORK, 0.88))

    assert result.active_mode == MODE_WORK


def test_low_confidence_holds_previous_mode():
    result = TransitionManager().transition(MODE_DAILY, decision(MODE_WORK, 0.6))

    assert result.active_mode == MODE_DAILY
    assert result.transition == "hold_low_confidence"


def test_sex_uninhibited_can_enter_active_sex_mode():
    result = TransitionManager().transition(
        MODE_DAILY,
        decision(MODE_SEX, 0.9),
        desire_tier="uninhibited",
        enable_active_sex=True,
    )

    assert result.active_mode == MODE_SEX
    assert result.transition == "daily->sex"
    assert "sex_shadow_only" not in result.safety_flags


def test_sex_hint_is_held_without_explicit_continuation_signal():
    result = TransitionManager().transition(
        MODE_DAILY,
        decision(MODE_SEX, 0.82, signals={}, submode="hint_progression"),
        desire_tier="uninhibited",
        enable_active_sex=True,
    )

    assert result.active_mode == MODE_DAILY
    assert result.transition == "hold_sex_candidate"
    assert "sex_transition_confirmation_pending" in result.safety_flags


def test_sex_restrained_blocks_to_daily():
    result = TransitionManager().transition(
        MODE_DAILY,
        decision(MODE_SEX, 0.9),
        desire_tier="restrained",
        enable_active_sex=True,
    )

    assert result.active_mode == MODE_DAILY
    assert "blocked:sex_desire_gate_restrained" in result.reason
    assert "sex_desire_gate_restrained" in result.safety_flags


def test_sex_with_crisis_guard_becomes_daily():
    result = TransitionManager().transition(
        MODE_DAILY,
        decision(MODE_SEX, 0.9, ["crisis_guard"]),
        safety_flags=["crisis_guard"],
    )

    assert result.active_mode == MODE_DAILY
    assert "crisis_guard" in result.safety_flags


def test_relationship_conflict_is_daily_with_conflict_flag_not_mode():
    result = TransitionManager().transition(MODE_DAILY, decision(MODE_DAILY, 0.8, flags=["relationship_conflict"]))

    assert result.active_mode == MODE_DAILY
    assert result.transition == "stay:daily"
    assert "relationship_conflict" in result.safety_flags


def test_repair_is_daily_with_repair_flag_not_mode():
    result = TransitionManager().transition(MODE_DAILY, decision(MODE_DAILY, 0.82, flags=["repair_guard"]))

    assert result.active_mode == MODE_DAILY
    assert result.transition == "stay:daily"
    assert "repair_guard" in result.safety_flags


def test_work_does_not_hold_low_confidence_daily():
    """A configured work-exit boundary applies consistently to daily candidates."""
    result = TransitionManager().transition(
        MODE_WORK,
        decision(MODE_DAILY, confidence=0.55),
    )

    assert result.active_mode == MODE_WORK
    assert result.transition == "hold_work_context"


def test_work_holds_low_confidence_daily_candidate_to_prevent_reversal():
    result = TransitionManager().transition(
        MODE_WORK,
        decision(MODE_DAILY, confidence=0.60),
    )

    assert result.active_mode == MODE_WORK
    assert result.transition == "hold_work_context"


def test_work_exits_on_configured_confidence_boundary():
    result = TransitionManager().transition(
        MODE_WORK,
        decision(MODE_DAILY, confidence=0.90, submode="relationship_closeness"),
    )

    assert result.active_mode == MODE_DAILY
    assert result.transition == "work->daily"


def test_work_does_not_hold_low_confidence_relationship_daily():
    """A configured work-exit boundary applies consistently to daily candidates."""
    result = TransitionManager().transition(
        MODE_WORK,
        decision(MODE_DAILY, confidence=0.6),
    )

    assert result.active_mode == MODE_WORK
    assert result.transition == "hold_work_context"


def test_work_exits_immediately_on_existing_boundary_signal():
    result = TransitionManager().transition(
        MODE_WORK,
        decision(MODE_DAILY, confidence=0.55, signals={"sex_scene_close": True}),
    )

    assert result.active_mode == MODE_DAILY
    assert result.transition == "work->daily"


def test_explicit_task_request_is_immediate_even_below_confidence_gate():
    result = TransitionManager().transition(
        MODE_DAILY,
        decision(MODE_WORK, confidence=0.55, signals={"explicit_task_request": True}),
    )

    assert result.active_mode == MODE_WORK
    assert result.transition == "daily->work"


def test_sex_holds_on_daily_without_explicit_exit_signal():
    result = TransitionManager().transition(
        MODE_SEX,
        decision(MODE_DAILY, confidence=0.78),
        desire_tier="uninhibited",
        enable_active_sex=True,
    )

    assert result.active_mode == MODE_SEX
    assert result.transition == "hold_sex_continuation"


def test_sex_continuation_signal_uses_specific_hold_transition():
    result = TransitionManager().transition(
        MODE_SEX,
        decision(MODE_DAILY, confidence=0.78, signals={"sex_scene_continue": True}),
        desire_tier="uninhibited",
        enable_active_sex=True,
    )

    assert result.active_mode == MODE_SEX
    assert result.transition == "hold_sex_continuation"


def test_sex_close_to_daily_enters_aftercare_daily():
    result = TransitionManager().transition(
        MODE_SEX,
        decision(MODE_DAILY, confidence=0.55, signals={"sex_scene_close": True}),
        desire_tier="uninhibited",
        enable_active_sex=True,
    )

    assert result.active_mode == MODE_DAILY
    assert result.transition == "sex->daily"


def test_sex_crisis_guard_exits_to_daily():
    result = TransitionManager().transition(
        MODE_SEX,
        decision(MODE_SEX, confidence=0.9, flags=["crisis_guard", "repair_guard"]),
        desire_tier="uninhibited",
        enable_active_sex=True,
    )

    assert result.active_mode == MODE_DAILY
    assert result.transition == "sex->daily"


def test_sex_exits_on_explicit_work_request():
    result = TransitionManager().transition(
        MODE_SEX,
        decision(MODE_WORK, confidence=0.88),
        desire_tier="uninhibited",
        enable_active_sex=True,
    )

    assert result.active_mode == MODE_WORK
    assert result.transition == "sex->work"


def test_sex_exits_on_explicit_system_request_as_work():
    result = TransitionManager().transition(
        MODE_SEX,
        decision(MODE_WORK, confidence=0.9),
        desire_tier="uninhibited",
        enable_active_sex=True,
    )

    assert result.active_mode == MODE_WORK
    assert result.transition == "sex->work"


def test_sex_exits_on_explicit_closing_language():
    result = TransitionManager().transition(
        MODE_SEX,
        decision(MODE_DAILY, confidence=0.55, signals={"sex_scene_close": True, "normalized_text": "先到这里，收一下，晚点再继续"}),
        desire_tier="uninhibited",
        enable_active_sex=True,
    )

    assert result.active_mode == MODE_DAILY
    assert result.transition == "sex->daily"


def test_existing_sex_continuation_does_not_reenter_desire_gate():
    """Once active, explicit continuation must not be downgraded by the entry gate."""
    result = TransitionManager().transition(
        MODE_SEX,
        decision(
            MODE_SEX,
            confidence=0.9,
            flags=["sex_requires_gate"],
            signals={"normalized_text": "继续，不要停"},
        ),
        desire_tier="restrained",
        enable_active_sex=True,
    )

    assert result.active_mode == MODE_SEX
    assert result.transition in {"stay:sex", "hold_sex_continue", "hold_sex_continuation"}
    assert "sex_desire_gate_restrained" not in result.safety_flags


def test_existing_sex_continuation_keeps_sex_layer_under_low_desire(tmp_path):
    """A continued active scene should not route to daily just because current score is low."""
    orchestrator = StateOrchestrator(tmp_path, core_source="host_core")

    selected_layers = orchestrator._selected_layers(
        active_mode=MODE_SEX,
        safety_flags=[],
        previous_mode=MODE_SEX,
    )

    assert "sex" in selected_layers
    assert "daily" not in selected_layers
