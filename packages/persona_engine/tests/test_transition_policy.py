"""Tests for Phase 5: Pair-specific transition policy and decision table.

Task 5.1: Extract existing transition_manager.py behavior to legacy table
Task 5.2: Introduce pair-specific policy configuration
Task 5.3: Dual-execution shadow comparison
"""

import pytest

from packages.persona_engine.persona_orchestrator.transition_policy import (
    ContinuationTarget,
    ForceEvent,
    GateResult,
    TransitionCondition,
    TransitionRule,
    TransitionTable,
    build_legacy_transition_table,
    apply_legacy_conditions,
)


class TestTransitionCondition:
    """Test transition condition matching logic."""

    def test_exact_match(self):
        condition = TransitionCondition(
            previous_mode="work",
            nominated_mode="daily",
            force_event=ForceEvent.NONE,
        )
        assert condition.matches("work", "daily", ForceEvent.NONE, ContinuationTarget.NONE, GateResult.NONE)

    def test_any_previous_matches_all(self):
        condition = TransitionCondition(
            previous_mode="any",
            nominated_mode="daily",
        )
        assert condition.matches("work", "daily", ForceEvent.NONE, ContinuationTarget.NONE, GateResult.NONE)
        assert condition.matches("sex", "daily", ForceEvent.NONE, ContinuationTarget.NONE, GateResult.NONE)
        assert condition.matches(None, "daily", ForceEvent.NONE, ContinuationTarget.NONE, GateResult.NONE)

    def test_force_event_must_match(self):
        condition = TransitionCondition(
            previous_mode="any",
            nominated_mode="any",
            force_event=ForceEvent.EXPLICIT_TASK,
        )
        assert condition.matches("sex", "work", ForceEvent.EXPLICIT_TASK, ContinuationTarget.NONE, GateResult.NONE)
        assert not condition.matches("sex", "work", ForceEvent.NONE, ContinuationTarget.NONE, GateResult.NONE)

    def test_continuation_target_filtering(self):
        condition = TransitionCondition(
            previous_mode="work",
            nominated_mode="daily",
            continuation_target=ContinuationTarget.WORK,
        )
        assert condition.matches("work", "daily", ForceEvent.NONE, ContinuationTarget.WORK, GateResult.NONE)
        assert not condition.matches("work", "daily", ForceEvent.NONE, ContinuationTarget.RELATIONSHIP, GateResult.NONE)

    def test_gate_result_filtering(self):
        condition = TransitionCondition(
            previous_mode="any",
            nominated_mode="sex",
            gate_result=GateResult.PASS,
        )
        assert condition.matches("work", "sex", ForceEvent.NONE, ContinuationTarget.NONE, GateResult.PASS)
        assert not condition.matches("work", "sex", ForceEvent.NONE, ContinuationTarget.NONE, GateResult.FAIL)

    def test_submode_filtering(self):
        condition = TransitionCondition(
            previous_mode="work",
            nominated_mode="sex",
            submode="hint_progression",
        )
        assert condition.matches("work", "sex", ForceEvent.NONE, ContinuationTarget.NONE, GateResult.NONE, "hint_progression")
        assert not condition.matches("work", "sex", ForceEvent.NONE, ContinuationTarget.NONE, GateResult.NONE, "explicit_progression")


class TestTransitionRule:
    """Test transition rule validation."""

    def test_valid_rule_creation(self):
        rule = TransitionRule(
            rule_id="TEST.RULE.001",
            condition=TransitionCondition("work", "daily"),
            active_mode="daily",
            transition_label="work->daily",
            reason="test reason",
            priority=100,
        )
        assert rule.rule_id == "TEST.RULE.001"
        assert rule.priority == 100

    def test_invalid_active_mode_rejected(self):
        with pytest.raises(ValueError, match="active_mode must be daily/work/sex"):
            TransitionRule(
                rule_id="BAD.MODE",
                condition=TransitionCondition("any", "any"),
                active_mode="invalid",
                transition_label="test",
                reason="test",
                priority=100,
            )

    def test_negative_priority_rejected(self):
        with pytest.raises(ValueError, match="priority must be non-negative"):
            TransitionRule(
                rule_id="BAD.PRIORITY",
                condition=TransitionCondition("any", "any"),
                active_mode="daily",
                transition_label="test",
                reason="test",
                priority=-1,
            )

    def test_invalid_stale_hold_rejected(self):
        with pytest.raises(ValueError, match="max_stale_hold_turns must be positive"):
            TransitionRule(
                rule_id="BAD.STALE",
                condition=TransitionCondition("any", "any"),
                active_mode="daily",
                transition_label="test",
                reason="test",
                priority=100,
                max_stale_hold_turns=0,
            )


class TestTransitionTable:
    """Test transition table decision logic."""

    def test_empty_table_returns_none(self):
        table = TransitionTable([])
        result = table.decide("work", "daily", ForceEvent.NONE, ContinuationTarget.NONE, GateResult.NONE)
        assert result is None

    def test_single_matching_rule(self):
        rule = TransitionRule(
            rule_id="TEST.001",
            condition=TransitionCondition("work", "daily"),
            active_mode="daily",
            transition_label="work->daily",
            reason="test",
            priority=100,
        )
        table = TransitionTable([rule])
        result = table.decide("work", "daily", ForceEvent.NONE, ContinuationTarget.NONE, GateResult.NONE)
        assert result is not None
        assert result.rule_id == "TEST.001"

    def test_priority_ordering(self):
        low_priority = TransitionRule(
            rule_id="LOW",
            condition=TransitionCondition("work", "daily"),
            active_mode="work",
            transition_label="hold",
            reason="low priority hold",
            priority=100,
        )
        high_priority = TransitionRule(
            rule_id="HIGH",
            condition=TransitionCondition("work", "daily"),
            active_mode="daily",
            transition_label="switch",
            reason="high priority switch",
            priority=200,
        )
        table = TransitionTable([low_priority, high_priority])
        result = table.decide("work", "daily", ForceEvent.NONE, ContinuationTarget.NONE, GateResult.NONE)
        assert result.rule_id == "HIGH"

    def test_no_match_returns_none(self):
        rule = TransitionRule(
            rule_id="SPECIFIC",
            condition=TransitionCondition("work", "daily", force_event=ForceEvent.EXPLICIT_TASK),
            active_mode="work",
            transition_label="hold",
            reason="specific condition",
            priority=100,
        )
        table = TransitionTable([rule])
        # Query without the force event
        result = table.decide("work", "daily", ForceEvent.NONE, ContinuationTarget.NONE, GateResult.NONE)
        assert result is None


class TestLegacyTransitionTable:
    """Test that legacy table extraction preserves existing behavior."""

    def test_table_builds_without_error(self):
        table = build_legacy_transition_table()
        assert table is not None
        assert len(table.rules) > 0

    def test_crisis_guard_highest_priority(self):
        table = build_legacy_transition_table()
        result = table.decide("work", "work", ForceEvent.CRISIS, ContinuationTarget.NONE, GateResult.NONE)
        assert result is not None
        assert result.active_mode == "daily"
        assert "crisis" in result.reason.lower()

    def test_sex_gate_pass_allows_entry(self):
        table = build_legacy_transition_table()
        result = table.decide("work", "sex", ForceEvent.NONE, ContinuationTarget.NONE, GateResult.PASS)
        assert result is not None
        assert result.active_mode == "sex"

    def test_sex_gate_fail_blocks_entry(self):
        table = build_legacy_transition_table()
        result = table.decide("work", "sex", ForceEvent.NONE, ContinuationTarget.NONE, GateResult.FAIL)
        assert result is not None
        assert result.active_mode == "daily"
        assert "blocked" in result.transition_label

    def test_explicit_task_exits_sex(self):
        table = build_legacy_transition_table()
        result = table.decide("sex", "work", ForceEvent.EXPLICIT_TASK, ContinuationTarget.NONE, GateResult.NONE)
        assert result is not None
        assert result.active_mode == "work"
        assert result.condition.previous_mode == "sex"

    def test_sex_scene_close_exits_to_daily(self):
        table = build_legacy_transition_table()
        result = table.decide("sex", "daily", ForceEvent.SCENE_CLOSE, ContinuationTarget.NONE, GateResult.NONE)
        assert result is not None
        assert result.active_mode == "daily"

    def test_sex_holds_on_daily_nomination(self):
        table = build_legacy_transition_table()
        result = table.decide("sex", "daily", ForceEvent.NONE, ContinuationTarget.NONE, GateResult.NONE)
        assert result is not None
        assert result.active_mode == "sex"
        assert "hold" in result.transition_label

    def test_work_continuation_holds_work(self):
        table = build_legacy_transition_table()
        result = table.decide("work", "daily", ForceEvent.NONE, ContinuationTarget.WORK, GateResult.NONE)
        assert result is not None
        # The work continuation rule should match when continuation target is WORK
        assert result.rule_id == "WORK.HOLD.CONTEXT_ACTION"
        assert result.active_mode == "work"

    def test_start_state_accepts_nomination(self):
        table = build_legacy_transition_table()
        result = table.decide(None, "work", ForceEvent.NONE, ContinuationTarget.NONE, GateResult.NONE)
        assert result is not None
        # Start rule exists
        assert result.rule_id == "START.ANY"


class TestLegacyConditionApplication:
    """Test that apply_legacy_conditions correctly gates rules by confidence/message analysis."""

    def test_start_rule_uses_nominated_mode(self):
        start_rule = TransitionRule(
            rule_id="START.ANY",
            condition=TransitionCondition("none", "any"),
            active_mode="daily",  # placeholder
            transition_label="start:nominated",
            reason="first turn",
            priority=800,
        )

        result = apply_legacy_conditions(
            start_rule, None, "work", 0.8, False, False, False, False
        )
        assert result is not None
        assert result.active_mode == "work"
        assert result.transition_label == "start:work"

    def test_short_hold_requires_conditions(self):
        short_hold_rule = TransitionRule(
            rule_id="ANTIFLAP.SHORT_HOLD.NOT_SEX",
            condition=TransitionCondition("any", "daily"),
            active_mode="work",
            transition_label="hold_short_message",
            reason="short hold",
            priority=600,
        )

        # Should apply when short message, low confidence, not from sex
        result = apply_legacy_conditions(
            short_hold_rule, "work", "daily", 0.5, True, False, False, False
        )
        assert result is not None
        assert result.active_mode == "work"

        # Should not apply when confidence high
        result = apply_legacy_conditions(
            short_hold_rule, "work", "daily", 0.8, True, False, False, False
        )
        assert result is None

        # Should not apply when previous is sex
        result = apply_legacy_conditions(
            short_hold_rule, "sex", "daily", 0.5, True, False, False, False
        )
        assert result is None

    def test_work_hold_requires_low_confidence_and_continuation(self):
        work_hold_rule = TransitionRule(
            rule_id="WORK.HOLD.CONTEXT_ACTION",
            condition=TransitionCondition("work", "daily", continuation_target=ContinuationTarget.WORK),
            active_mode="work",
            transition_label="hold_context_action",
            reason="context hold",
            priority=500,
        )

        # Should apply with low confidence + continuation
        result = apply_legacy_conditions(
            work_hold_rule, "work", "daily", 0.5, False, False, False, True
        )
        assert result is not None

        # Should not apply with high confidence
        result = apply_legacy_conditions(
            work_hold_rule, "work", "daily", 0.8, False, False, False, True
        )
        assert result is None

        # Should not apply without continuation
        result = apply_legacy_conditions(
            work_hold_rule, "work", "daily", 0.5, False, False, False, False
        )
        assert result is None

    def test_high_confidence_work_requires_threshold(self):
        work_enter_rule = TransitionRule(
            rule_id="WORK.ENTER.HIGH_CONFIDENCE",
            condition=TransitionCondition("any", "work"),
            active_mode="work",
            transition_label="any->work",
            reason="high confidence",
            priority=400,
        )

        # Should apply with confidence >= 0.75
        result = apply_legacy_conditions(
            work_enter_rule, "daily", "work", 0.75, False, False, False, False
        )
        assert result is not None

        # Should not apply with lower confidence
        result = apply_legacy_conditions(
            work_enter_rule, "daily", "work", 0.74, False, False, False, False
        )
        assert result is None

    def test_machine_status_hold_requires_detection(self):
        machine_rule = TransitionRule(
            rule_id="WORK.HOLD.MACHINE_STATUS",
            condition=TransitionCondition("work", "daily"),
            active_mode="work",
            transition_label="hold_low_confidence",
            reason="machine status",
            priority=300,
        )

        # Should apply when machine status detected
        result = apply_legacy_conditions(
            machine_rule, "work", "daily", 0.5, False, False, True, False
        )
        assert result is not None

        # Should not apply without machine status
        result = apply_legacy_conditions(
            machine_rule, "work", "daily", 0.5, False, False, False, False
        )
        assert result is None

    def test_default_accept_uses_nominated_mode(self):
        default_rule = TransitionRule(
            rule_id="DEFAULT.ACCEPT_NOMINATION",
            condition=TransitionCondition("any", "any"),
            active_mode="daily",
            transition_label="accept_nomination",
            reason="default",
            priority=0,
        )

        result = apply_legacy_conditions(
            default_rule, "work", "daily", 0.8, False, False, False, False
        )
        assert result is not None
        assert result.active_mode == "daily"
        assert result.transition_label == "work->daily"


class TestTransitionTableCoverage:
    """Test that the transition table provides complete coverage without conflicts."""

    def test_all_mode_pairs_have_default_coverage(self):
        """Every (previous, nominated) pair should match at least the default rule."""
        table = build_legacy_transition_table()

        modes = ["daily", "work", "sex", None]
        nominated_modes = ["daily", "work", "sex"]

        for prev in modes:
            for nom in nominated_modes:
                result = table.decide(prev, nom, ForceEvent.NONE, ContinuationTarget.NONE, GateResult.NONE)
                assert result is not None, f"No rule for ({prev}, {nom})"

    def test_hard_boundaries_override_defaults(self):
        """Crisis and explicit exits should override lower-priority rules."""
        table = build_legacy_transition_table()

        # Crisis always goes to daily
        for prev in ["daily", "work", "sex"]:
            for nom in ["daily", "work", "sex"]:
                result = table.decide(prev, nom, ForceEvent.CRISIS, ContinuationTarget.NONE, GateResult.NONE)
                assert result is not None
                assert result.active_mode == "daily"

    def test_protected_mode_gate_enforcement(self):
        """Sex mode must pass gate to enter."""
        table = build_legacy_transition_table()

        for prev in ["daily", "work"]:
            # Gate pass allows entry
            result_pass = table.decide(prev, "sex", ForceEvent.NONE, ContinuationTarget.NONE, GateResult.PASS)
            assert result_pass is not None
            assert result_pass.active_mode == "sex"

            # Gate fail blocks entry
            result_fail = table.decide(prev, "sex", ForceEvent.NONE, ContinuationTarget.NONE, GateResult.FAIL)
            assert result_fail is not None
            assert result_fail.active_mode == "daily"


class TestMaxStaleHoldEnforcement:
    """Test that max_stale_hold_turns is properly configured for hold rules."""

    def test_sex_hold_has_stale_limit(self):
        table = build_legacy_transition_table()
        result = table.decide("sex", "daily", ForceEvent.NONE, ContinuationTarget.NONE, GateResult.NONE)
        assert result is not None
        assert result.max_stale_hold_turns is not None
        assert result.max_stale_hold_turns > 0

    def test_work_continuation_hold_has_limit(self):
        table = build_legacy_transition_table()
        result = table.decide("work", "daily", ForceEvent.NONE, ContinuationTarget.WORK, GateResult.NONE)
        assert result is not None
        # This rule should have a stale hold limit
        matching_rule = next((r for r in table.rules if r.rule_id == "WORK.HOLD.CONTEXT_ACTION"), None)
        assert matching_rule is not None
        assert matching_rule.max_stale_hold_turns is not None

    def test_low_confidence_hold_has_limit(self):
        table = build_legacy_transition_table()
        # Find the low confidence hold rule
        rule = next((r for r in table.rules if r.rule_id == "ANTIFLAP.HOLD.LOW_CONFIDENCE"), None)
        assert rule is not None
        assert rule.max_stale_hold_turns is not None
        assert rule.max_stale_hold_turns == 5
