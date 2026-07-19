"""Tests for Phase 5 shadow comparison between legacy and new transition logic."""

import pytest

from packages.persona_engine.persona_orchestrator.transition_shadow import (
    TransitionComparison,
    TransitionShadowComparator,
)
from packages.persona_engine.persona_orchestrator.transition_policy import (
    ContinuationTarget,
    ForceEvent,
    GateResult,
    TransitionCondition,
    TransitionRule,
)
from packages.persona_engine.persona_orchestrator.types import TransitionDecision


class TestTransitionShadowComparator:
    """Test shadow comparison logic."""

    def test_records_agreement(self):
        comparator = TransitionShadowComparator(enable_logging=False)

        legacy = TransitionDecision(
            previous_mode="work",
            requested_mode="daily",
            active_mode="daily",
            transition="work->daily",
            confidence=0.8,
            reason="test",
            safety_flags=[],
        )

        new_rule = TransitionRule(
            rule_id="TEST.001",
            condition=TransitionCondition("work", "daily"),
            active_mode="daily",
            transition_label="work->daily",
            reason="test",
            priority=100,
        )

        comparison = comparator.compare("work", "daily", 0.8, legacy, new_rule)

        assert comparison.agreement is True
        assert comparison.disagreement_type is None
        assert comparator.get_agreement_rate() == 1.0

    def test_records_mode_disagreement(self):
        comparator = TransitionShadowComparator(enable_logging=False)

        legacy = TransitionDecision(
            previous_mode="work",
            requested_mode="daily",
            active_mode="work",  # Legacy holds work
            transition="hold_low_confidence",
            confidence=0.6,
            reason="anti-flap",
            safety_flags=[],
        )

        new_rule = TransitionRule(
            rule_id="TEST.002",
            condition=TransitionCondition("work", "daily"),
            active_mode="daily",  # New switches to daily
            transition_label="work->daily",
            reason="test",
            priority=100,
        )

        comparison = comparator.compare("work", "daily", 0.6, legacy, new_rule)

        assert comparison.agreement is False
        assert comparison.disagreement_type == "mode"
        assert comparison.legacy_active_mode == "work"
        assert comparison.new_active_mode == "daily"
        assert comparator.get_agreement_rate() == 0.0

    def test_records_no_match(self):
        comparator = TransitionShadowComparator(enable_logging=False)

        legacy = TransitionDecision(
            previous_mode="work",
            requested_mode="daily",
            active_mode="work",
            transition="hold",
            confidence=0.6,
            reason="test",
            safety_flags=[],
        )

        # New table returns None (no matching rule)
        comparison = comparator.compare("work", "daily", 0.6, legacy, None)

        assert comparison.agreement is False
        assert comparison.disagreement_type == "no_match"
        assert comparison.new_active_mode is None
        assert comparison.new_rule_id is None

    def test_accumulates_statistics(self):
        comparator = TransitionShadowComparator(enable_logging=False)

        legacy_agree = TransitionDecision(
            previous_mode=None, requested_mode="work", active_mode="work",
            transition="start:work", confidence=0.9, reason="test", safety_flags=[]
        )
        legacy_disagree = TransitionDecision(
            previous_mode="work", requested_mode="daily", active_mode="work",
            transition="hold", confidence=0.6, reason="test", safety_flags=[]
        )

        rule_agree = TransitionRule(
            rule_id="AGREE", condition=TransitionCondition("none", "work"),
            active_mode="work", transition_label="start:work", reason="test", priority=100
        )
        rule_disagree = TransitionRule(
            rule_id="DISAGREE", condition=TransitionCondition("work", "daily"),
            active_mode="daily", transition_label="switch", reason="test", priority=100
        )

        # 3 agreements, 2 disagreements
        comparator.compare(None, "work", 0.9, legacy_agree, rule_agree)
        comparator.compare(None, "work", 0.9, legacy_agree, rule_agree)
        comparator.compare("work", "daily", 0.6, legacy_disagree, rule_disagree)
        comparator.compare(None, "work", 0.9, legacy_agree, rule_agree)
        comparator.compare("work", "daily", 0.6, legacy_disagree, None)

        assert comparator.get_agreement_rate() == 0.6  # 3/5

        summary = comparator.get_summary()
        assert summary["total_decisions"] == 5
        assert summary["agreements"] == 3
        assert summary["disagreements"] == 2
        assert summary["agreement_rate"] == 0.6

    def test_get_disagreements_filters_correctly(self):
        comparator = TransitionShadowComparator(enable_logging=False)

        legacy = TransitionDecision(
            previous_mode="work", requested_mode="daily", active_mode="work",
            transition="hold", confidence=0.6, reason="test", safety_flags=[]
        )

        agree_rule = TransitionRule(
            rule_id="AGREE", condition=TransitionCondition("work", "daily"),
            active_mode="work", transition_label="hold", reason="test", priority=100
        )
        disagree_rule = TransitionRule(
            rule_id="DISAGREE", condition=TransitionCondition("work", "daily"),
            active_mode="daily", transition_label="switch", reason="test", priority=100
        )

        comparator.compare("work", "daily", 0.6, legacy, agree_rule)
        comparator.compare("work", "daily", 0.6, legacy, disagree_rule)
        comparator.compare("work", "daily", 0.6, legacy, agree_rule)

        disagreements = comparator.get_disagreements()
        assert len(disagreements) == 1
        assert disagreements[0].new_rule_id == "DISAGREE"

    def test_reset_clears_state(self):
        comparator = TransitionShadowComparator(enable_logging=False)

        legacy = TransitionDecision(
            previous_mode=None, requested_mode="work", active_mode="work",
            transition="start", confidence=0.9, reason="test", safety_flags=[]
        )
        rule = TransitionRule(
            rule_id="TEST", condition=TransitionCondition("none", "work"),
            active_mode="work", transition_label="start", reason="test", priority=100
        )

        comparator.compare(None, "work", 0.9, legacy, rule)
        assert comparator.get_summary()["total_decisions"] == 1

        comparator.reset()
        assert comparator.get_summary()["total_decisions"] == 0
        assert comparator.get_agreement_rate() == 1.0  # Default when no data

    def test_case_context_hash_preserved(self):
        comparator = TransitionShadowComparator(enable_logging=False)

        legacy = TransitionDecision(
            previous_mode=None, requested_mode="work", active_mode="work",
            transition="start", confidence=0.9, reason="test", safety_flags=[]
        )
        rule = TransitionRule(
            rule_id="TEST", condition=TransitionCondition("none", "work"),
            active_mode="work", transition_label="start", reason="test", priority=100
        )

        comparison = comparator.compare(
            None, "work", 0.9, legacy, rule, case_context_hash="abc123"
        )

        assert comparison.case_context_hash == "abc123"

    def test_disagreement_by_type_breakdown(self):
        comparator = TransitionShadowComparator(enable_logging=False)

        legacy = TransitionDecision(
            previous_mode="work", requested_mode="daily", active_mode="work",
            transition="hold", confidence=0.6, reason="test", safety_flags=[]
        )

        mode_rule = TransitionRule(
            rule_id="MODE", condition=TransitionCondition("work", "daily"),
            active_mode="daily", transition_label="switch", reason="test", priority=100
        )

        # 2 mode disagreements, 1 no_match
        comparator.compare("work", "daily", 0.6, legacy, mode_rule)
        comparator.compare("work", "daily", 0.6, legacy, mode_rule)
        comparator.compare("work", "daily", 0.6, legacy, None)

        summary = comparator.get_summary()
        assert summary["disagreement_by_type"]["mode"] == 2
        assert summary["disagreement_by_type"]["no_match"] == 1


class TestTransitionComparisonDataclass:
    """Test TransitionComparison immutability and structure."""

    def test_frozen_immutable(self):
        comparison = TransitionComparison(
            previous_mode="work",
            nominated_mode="daily",
            confidence=0.8,
            legacy_active_mode="work",
            legacy_transition_label="hold",
            legacy_reason="test",
            new_active_mode="daily",
            new_transition_label="switch",
            new_reason="new",
            new_rule_id="TEST.001",
            agreement=False,
            disagreement_type="mode",
        )

        # Frozen dataclass should not allow mutation
        with pytest.raises(AttributeError):
            comparison.agreement = True  # type: ignore

    def test_all_fields_accessible(self):
        comparison = TransitionComparison(
            previous_mode="work",
            nominated_mode="daily",
            confidence=0.7,
            legacy_active_mode="work",
            legacy_transition_label="hold",
            legacy_reason="legacy",
            new_active_mode="daily",
            new_transition_label="switch",
            new_reason="new",
            new_rule_id="RULE.001",
            agreement=False,
            disagreement_type="mode",
            case_context_hash="hash123",
        )

        assert comparison.previous_mode == "work"
        assert comparison.nominated_mode == "daily"
        assert comparison.confidence == 0.7
        assert comparison.legacy_active_mode == "work"
        assert comparison.new_active_mode == "daily"
        assert comparison.new_rule_id == "RULE.001"
        assert comparison.agreement is False
        assert comparison.disagreement_type == "mode"
        assert comparison.case_context_hash == "hash123"
