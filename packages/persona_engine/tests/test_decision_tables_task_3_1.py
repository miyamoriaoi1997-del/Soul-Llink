"""Tests for Phase 3 Task 3.1: Evidence to input equivalence classes."""
import pytest

from packages.persona_engine.persona_orchestrator.rule_schema import (
    ConfidenceBand,
    ContinuationTarget,
    DiscourseMode,
    EvidenceStrength,
    FeatureClass,
    HardEvent,
    MatchEvidence,
    Mode,
)


# Test will guide implementation
class TestEvidenceClassification:
    """Test evidence-to-band mapping for decision table inputs."""

    def test_import_decision_tables_module(self):
        """Module should exist and be importable."""
        from packages.persona_engine.persona_orchestrator import decision_tables
        assert decision_tables is not None

    def test_classify_hard_event_from_evidence(self):
        """Should extract hard_event from evidence collection."""
        from packages.persona_engine.persona_orchestrator.decision_tables import (
            classify_hard_event
        )

        # No boundary evidence
        evidences_none = []
        assert classify_hard_event(evidences_none) == HardEvent.NONE

        # Boundary evidence without veto
        evidences_normal = [
            MatchEvidence(
                rule_id="BOUND.001",
                feature_class=FeatureClass.HARD_BOUNDARY,
                candidate_mode=Mode.DAILY,
                weight=0.85,
                priority=1000,
                clause_index=0
            )
        ]
        # For now, classify based on presence
        result = classify_hard_event(evidences_normal)
        assert result in (HardEvent.BOUNDARY_EXIT, HardEvent.NONE)

    def test_classify_task_evidence_strength(self):
        """Should map task evidence to strength bands."""
        from packages.persona_engine.persona_orchestrator.decision_tables import (
            classify_task_evidence
        )

        # No task evidence
        evidences_none = []
        assert classify_task_evidence(evidences_none) == EvidenceStrength.NONE

        # Weak task evidence (low weight/priority)
        evidences_weak = [
            MatchEvidence(
                rule_id="TASK.LOW",
                feature_class=FeatureClass.EXPLICIT_TASK,
                candidate_mode=Mode.WORK,
                weight=0.50,
                priority=500,
                clause_index=0
            )
        ]
        result = classify_task_evidence(evidences_weak)
        assert result in (EvidenceStrength.WEAK, EvidenceStrength.STRONG)

        # Strong task evidence (high weight/priority)
        evidences_strong = [
            MatchEvidence(
                rule_id="TASK.HIGH",
                feature_class=FeatureClass.EXPLICIT_TASK,
                candidate_mode=Mode.WORK,
                weight=0.90,
                priority=800,
                clause_index=0
            )
        ]
        result = classify_task_evidence(evidences_strong)
        assert result in (EvidenceStrength.STRONG, EvidenceStrength.EXPLICIT)

    def test_classify_protected_evidence_strength(self):
        """Should map protected progression evidence."""
        from packages.persona_engine.persona_orchestrator.decision_tables import (
            classify_protected_evidence
        )

        evidences_none = []
        assert classify_protected_evidence(evidences_none) == EvidenceStrength.NONE

        evidences_hint = [
            MatchEvidence(
                rule_id="PROT.HINT",
                feature_class=FeatureClass.PROTECTED_PROGRESSION,
                candidate_mode=Mode.SEX,
                weight=0.60,
                priority=700,
                clause_index=0
            )
        ]
        result = classify_protected_evidence(evidences_hint)
        assert result in (EvidenceStrength.WEAK, EvidenceStrength.STRONG)

    def test_classify_relationship_evidence_strength(self):
        """Should map relationship evidence."""
        from packages.persona_engine.persona_orchestrator.decision_tables import (
            classify_relationship_evidence
        )

        evidences_none = []
        assert classify_relationship_evidence(evidences_none) == EvidenceStrength.NONE

        evidences_explicit = [
            MatchEvidence(
                rule_id="REL.HIGH",
                feature_class=FeatureClass.RELATIONSHIP,
                candidate_mode=Mode.DAILY,
                weight=0.85,
                priority=600,
                clause_index=0
            )
        ]
        result = classify_relationship_evidence(evidences_explicit)
        assert result in (EvidenceStrength.WEAK, EvidenceStrength.EXPLICIT)

    def test_classify_discourse_mode(self):
        """Should determine discourse mode from evidence flags."""
        from packages.persona_engine.persona_orchestrator.decision_tables import (
            classify_discourse
        )

        # Direct (no special flags)
        evidences_direct = [
            MatchEvidence(
                rule_id="TEST.001",
                feature_class=FeatureClass.EXPLICIT_TASK,
                candidate_mode=Mode.WORK,
                weight=0.85,
                priority=800,
                clause_index=0,
                quoted=False,
                negated=False,
                hypothetical=False,
                meta_context=False
            )
        ]
        assert classify_discourse(evidences_direct) == DiscourseMode.DIRECT

        # Quoted
        evidences_quoted = [
            MatchEvidence(
                rule_id="TEST.002",
                feature_class=FeatureClass.EXPLICIT_TASK,
                candidate_mode=Mode.WORK,
                weight=0.85,
                priority=800,
                clause_index=0,
                quoted=True
            )
        ]
        assert classify_discourse(evidences_quoted) == DiscourseMode.QUOTED

        # Meta discussion
        evidences_meta = [
            MatchEvidence(
                rule_id="META.001",
                feature_class=FeatureClass.EXPLICIT_META,
                candidate_mode=Mode.WORK,
                weight=0.90,
                priority=900,
                clause_index=0,
                meta_context=True
            )
        ]
        assert classify_discourse(evidences_meta) == DiscourseMode.META

    def test_classify_continuation_target(self):
        """Should determine continuation target from context."""
        from packages.persona_engine.persona_orchestrator.decision_tables import (
            classify_continuation_target
        )

        # No continuation evidence
        evidences_none = []
        assert classify_continuation_target(evidences_none) == ContinuationTarget.NONE

        # Work continuation
        evidences_work = [
            MatchEvidence(
                rule_id="CONT.WORK",
                feature_class=FeatureClass.CONTINUATION_BINDING,
                candidate_mode=Mode.WORK,
                weight=0.80,
                priority=500,
                clause_index=0,
                requires_satisfied=True
            )
        ]
        result = classify_continuation_target(evidences_work)
        assert result in (ContinuationTarget.WORK, ContinuationTarget.AMBIGUOUS)

    def test_classify_confidence_band(self):
        """Should map evidence quality to confidence bands."""
        from packages.persona_engine.persona_orchestrator.decision_tables import (
            classify_confidence
        )

        # High confidence: strong evidence, no conflicts
        evidences_high = [
            MatchEvidence(
                rule_id="STRONG.001",
                feature_class=FeatureClass.EXPLICIT_TASK,
                candidate_mode=Mode.WORK,
                weight=0.92,
                priority=800,
                clause_index=0
            )
        ]
        result = classify_confidence(evidences_high)
        assert result in (ConfidenceBand.MEDIUM, ConfidenceBand.HIGH)

        # Low confidence: weak evidence
        evidences_low = [
            MatchEvidence(
                rule_id="WEAK.001",
                feature_class=FeatureClass.FALLBACK,
                candidate_mode=Mode.DAILY,
                weight=0.40,
                priority=100,
                clause_index=0
            )
        ]
        result = classify_confidence(evidences_low)
        assert result in (ConfidenceBand.LOW, ConfidenceBand.MEDIUM)
