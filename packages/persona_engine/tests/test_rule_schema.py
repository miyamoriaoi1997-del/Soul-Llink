"""Tests for typed rule schema validation and immutability."""
import pytest

from packages.persona_engine.persona_orchestrator.rule_schema import (
    CompiledRule,
    ConfidenceBand,
    ContinuationTarget,
    DiscourseMode,
    EvidenceSet,
    EvidenceStrength,
    FeatureClass,
    HardEvent,
    MatchEvidence,
    MatchScope,
    Mode,
    NegationPolicy,
    NominationDecision,
    QuotePolicy,
    RuleTerm,
    TransitionContext,
    TransitionDecision,
    validate_feature_class,
    validate_mode,
    validate_priority,
    validate_weight,
)


class TestEnums:
    """Test enum definitions."""

    def test_mode_values(self):
        assert Mode.DAILY.value == "daily"
        assert Mode.WORK.value == "work"
        assert Mode.SEX.value == "sex"

    def test_feature_class_values(self):
        assert FeatureClass.HARD_BOUNDARY.value == "hard_boundary"
        assert FeatureClass.EXPLICIT_TASK.value == "explicit_task"

    def test_confidence_band_values(self):
        assert ConfidenceBand.LOW.value == "low"
        assert ConfidenceBand.MEDIUM.value == "medium"
        assert ConfidenceBand.HIGH.value == "high"


class TestRuleTerm:
    """Test RuleTerm immutability and validation."""

    def test_basic_term(self):
        term = RuleTerm(term="test", weight=0.8)
        assert term.term == "test"
        assert term.weight == 0.8
        assert term.requires_context == ()
        assert term.forbids_context == ()

    def test_term_with_context(self):
        term = RuleTerm(
            term="test",
            weight=0.9,
            requires_context=("ctx1", "ctx2"),
            forbids_context=("neg1",)
        )
        assert term.requires_context == ("ctx1", "ctx2")
        assert term.forbids_context == ("neg1",)

    def test_frozen(self):
        term = RuleTerm(term="test")
        with pytest.raises(AttributeError):
            term.weight = 0.5  # type: ignore

    def test_empty_term_rejected(self):
        with pytest.raises(ValueError, match="non-empty string"):
            RuleTerm(term="")

    def test_weight_range_validation(self):
        with pytest.raises(ValueError, match="weight must be in"):
            RuleTerm(term="test", weight=1.5)
        with pytest.raises(ValueError, match="weight must be in"):
            RuleTerm(term="test", weight=-0.1)


class TestCompiledRule:
    """Test CompiledRule validation."""

    def test_minimal_valid_rule(self):
        rule = CompiledRule(
            rule_id="TEST.001",
            feature_class=FeatureClass.EXPLICIT_TASK,
            candidate_mode=Mode.WORK,
            terms=(RuleTerm(term="test"),),
            weight=0.85,
            priority=700,
            match_any=True
        )
        assert rule.rule_id == "TEST.001"
        assert rule.candidate_mode == Mode.WORK
        assert len(rule.terms) == 1

    def test_empty_rule_id_rejected(self):
        with pytest.raises(ValueError, match="rule_id must be non-empty"):
            CompiledRule(
                rule_id="",
                feature_class=FeatureClass.EXPLICIT_TASK,
                candidate_mode=Mode.WORK,
                terms=(RuleTerm(term="test"),),
                weight=0.85,
                priority=700,
                match_any=True
            )

    def test_no_terms_rejected(self):
        with pytest.raises(ValueError, match="must have at least one term"):
            CompiledRule(
                rule_id="TEST.001",
                feature_class=FeatureClass.EXPLICIT_TASK,
                candidate_mode=Mode.WORK,
                terms=(),
                weight=0.85,
                priority=700,
                match_any=True
            )

    def test_weight_range_validation(self):
        with pytest.raises(ValueError, match="weight must be in"):
            CompiledRule(
                rule_id="TEST.001",
                feature_class=FeatureClass.EXPLICIT_TASK,
                candidate_mode=Mode.WORK,
                terms=(RuleTerm(term="test"),),
                weight=1.5,
                priority=700,
                match_any=True
            )

    def test_priority_range_validation(self):
        with pytest.raises(ValueError, match="priority must be in"):
            CompiledRule(
                rule_id="TEST.001",
                feature_class=FeatureClass.EXPLICIT_TASK,
                candidate_mode=Mode.WORK,
                terms=(RuleTerm(term="test"),),
                weight=0.85,
                priority=1500,
                match_any=True
            )

    def test_stop_processing_only_for_hard_boundary(self):
        # Should fail for non-hard-boundary
        with pytest.raises(ValueError, match="stop_processing only allowed"):
            CompiledRule(
                rule_id="TEST.001",
                feature_class=FeatureClass.EXPLICIT_TASK,
                candidate_mode=Mode.WORK,
                terms=(RuleTerm(term="test"),),
                weight=0.85,
                priority=700,
                match_any=True,
                stop_processing=True
            )

        # Should succeed for hard_boundary
        rule = CompiledRule(
            rule_id="EXIT.001",
            feature_class=FeatureClass.HARD_BOUNDARY,
            candidate_mode=Mode.DAILY,
            terms=(RuleTerm(term="exit"),),
            weight=1.0,
            priority=1000,
            match_any=True,
            stop_processing=True
        )
        assert rule.stop_processing is True


class TestMatchEvidence:
    """Test MatchEvidence validation logic."""

    def test_valid_evidence(self):
        evidence = MatchEvidence(
            rule_id="TEST.001",
            feature_class=FeatureClass.EXPLICIT_TASK,
            candidate_mode=Mode.WORK,
            weight=0.85,
            priority=700,
            clause_index=0
        )
        assert evidence.is_valid() is True

    def test_evidence_with_unsatisfied_requires(self):
        evidence = MatchEvidence(
            rule_id="TEST.001",
            feature_class=FeatureClass.EXPLICIT_TASK,
            candidate_mode=Mode.WORK,
            weight=0.85,
            priority=700,
            clause_index=0,
            requires_satisfied=False
        )
        assert evidence.is_valid() is False

    def test_evidence_with_triggered_forbids(self):
        evidence = MatchEvidence(
            rule_id="TEST.001",
            feature_class=FeatureClass.EXPLICIT_TASK,
            candidate_mode=Mode.WORK,
            weight=0.85,
            priority=700,
            clause_index=0,
            forbids_triggered=True
        )
        assert evidence.is_valid() is False


class TestValidationUtilities:
    """Test standalone validation functions."""

    def test_validate_weight(self):
        validate_weight(0.0)
        validate_weight(0.5)
        validate_weight(1.0)

        with pytest.raises(ValueError):
            validate_weight(1.5)
        with pytest.raises(ValueError):
            validate_weight(-0.1)
        with pytest.raises(TypeError):
            validate_weight("0.5")  # type: ignore

    def test_validate_priority(self):
        validate_priority(0)
        validate_priority(500)
        validate_priority(1000)

        with pytest.raises(ValueError):
            validate_priority(1500)
        with pytest.raises(ValueError):
            validate_priority(-1)
        with pytest.raises(TypeError):
            validate_priority("500")  # type: ignore

    def test_validate_mode(self):
        assert validate_mode("daily") == Mode.DAILY
        assert validate_mode("WORK") == Mode.WORK
        assert validate_mode("Sex") == Mode.SEX

        with pytest.raises(ValueError):
            validate_mode("invalid")

    def test_validate_feature_class(self):
        assert validate_feature_class("hard_boundary") == FeatureClass.HARD_BOUNDARY
        assert validate_feature_class("EXPLICIT_TASK") == FeatureClass.EXPLICIT_TASK

        with pytest.raises(ValueError):
            validate_feature_class("invalid")


class TestImmutability:
    """Test that all dataclasses are truly frozen."""

    def test_rule_term_immutable(self):
        term = RuleTerm(term="test")
        with pytest.raises(AttributeError):
            term.term = "modified"  # type: ignore

    def test_compiled_rule_immutable(self):
        rule = CompiledRule(
            rule_id="TEST.001",
            feature_class=FeatureClass.EXPLICIT_TASK,
            candidate_mode=Mode.WORK,
            terms=(RuleTerm(term="test"),),
            weight=0.85,
            priority=700,
            match_any=True
        )
        with pytest.raises(AttributeError):
            rule.weight = 0.9  # type: ignore

    def test_match_evidence_immutable(self):
        evidence = MatchEvidence(
            rule_id="TEST.001",
            feature_class=FeatureClass.EXPLICIT_TASK,
            candidate_mode=Mode.WORK,
            weight=0.85,
            priority=700,
            clause_index=0
        )
        with pytest.raises(AttributeError):
            evidence.weight = 0.9  # type: ignore

    def test_nomination_decision_immutable(self):
        decision = NominationDecision(
            nominated_mode=Mode.WORK,
            nominated_submode="task",
            force_event=HardEvent.NONE,
            confidence_band=ConfidenceBand.HIGH,
            reason_codes=("explicit_task",)
        )
        with pytest.raises(AttributeError):
            decision.nominated_mode = Mode.DAILY  # type: ignore
