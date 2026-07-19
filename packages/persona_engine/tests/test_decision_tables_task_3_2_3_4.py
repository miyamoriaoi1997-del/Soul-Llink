"""Tests for Phase 3 Tasks 3.2-3.4: Nomination table uniqueness and shadow comparison."""
import pytest

from packages.persona_engine.persona_orchestrator.decision_tables import (
    nominate_mode,
    classify_hard_event,
    classify_task_evidence,
    classify_protected_evidence,
    classify_relationship_evidence,
    classify_discourse,
    classify_continuation_target,
    classify_confidence,
)
from packages.persona_engine.persona_orchestrator.rule_schema import (
    ConfidenceBand,
    ContinuationTarget,
    DiscourseMode,
    EvidenceStrength,
    HardEvent,
    Mode,
)


class TestUniqueNominationTable:
    """Task 3.2: Test UNIQUE nomination decision table."""

    def test_hard_boundary_exit_always_daily(self):
        """Hard boundary exit must always nominate Daily."""
        result = nominate_mode(
            hard_event=HardEvent.BOUNDARY_EXIT,
            task_evidence=EvidenceStrength.EXPLICIT,  # Even with strong task
            protected_evidence=EvidenceStrength.NONE,
            relationship_evidence=EvidenceStrength.NONE,
            discourse=DiscourseMode.DIRECT,
            continuation=ContinuationTarget.NONE,
            confidence=ConfidenceBand.HIGH
        )

        assert result.nominated_mode == Mode.DAILY
        assert result.force_event == HardEvent.BOUNDARY_EXIT
        assert "hard_boundary" in result.reason_codes[0]

    def test_meta_discussion_routes_to_work(self):
        """Meta discussion always routes to Work."""
        result = nominate_mode(
            hard_event=HardEvent.NONE,
            task_evidence=EvidenceStrength.NONE,
            protected_evidence=EvidenceStrength.WEAK,  # Even with protected hint
            relationship_evidence=EvidenceStrength.NONE,
            discourse=DiscourseMode.META,
            continuation=ContinuationTarget.NONE,
            confidence=ConfidenceBand.MEDIUM
        )

        assert result.nominated_mode == Mode.WORK
        assert "meta" in result.reason_codes[0]

    def test_explicit_task_direct_discourse_work(self):
        """Explicit task with direct discourse nominates Work."""
        result = nominate_mode(
            hard_event=HardEvent.NONE,
            task_evidence=EvidenceStrength.EXPLICIT,
            protected_evidence=EvidenceStrength.NONE,
            relationship_evidence=EvidenceStrength.NONE,
            discourse=DiscourseMode.DIRECT,
            continuation=ContinuationTarget.NONE,
            confidence=ConfidenceBand.HIGH
        )

        assert result.nominated_mode == Mode.WORK
        assert "task" in result.reason_codes[0]

    def test_protected_requires_gate(self):
        """Protected progression must include required gate."""
        result = nominate_mode(
            hard_event=HardEvent.NONE,
            task_evidence=EvidenceStrength.NONE,
            protected_evidence=EvidenceStrength.EXPLICIT,
            relationship_evidence=EvidenceStrength.NONE,
            discourse=DiscourseMode.DIRECT,
            continuation=ContinuationTarget.NONE,
            confidence=ConfidenceBand.MEDIUM
        )

        assert result.nominated_mode == Mode.SEX
        assert result.required_gate != ""
        assert "protected" in result.reason_codes[0]

    def test_explicit_relationship_daily(self):
        """Explicit relationship nominates Daily."""
        result = nominate_mode(
            hard_event=HardEvent.NONE,
            task_evidence=EvidenceStrength.NONE,
            protected_evidence=EvidenceStrength.NONE,
            relationship_evidence=EvidenceStrength.EXPLICIT,
            discourse=DiscourseMode.DIRECT,
            continuation=ContinuationTarget.NONE,
            confidence=ConfidenceBand.HIGH
        )

        assert result.nominated_mode == Mode.DAILY
        assert "relationship" in result.reason_codes[0]

    def test_continuation_work_binding(self):
        """Work continuation nominates Work."""
        result = nominate_mode(
            hard_event=HardEvent.NONE,
            task_evidence=EvidenceStrength.NONE,
            protected_evidence=EvidenceStrength.NONE,
            relationship_evidence=EvidenceStrength.NONE,
            discourse=DiscourseMode.DIRECT,
            continuation=ContinuationTarget.WORK,
            confidence=ConfidenceBand.MEDIUM
        )

        assert result.nominated_mode == Mode.WORK
        assert "continuation" in result.reason_codes[0]

    def test_default_fallback_daily(self):
        """No evidence defaults to Daily."""
        result = nominate_mode(
            hard_event=HardEvent.NONE,
            task_evidence=EvidenceStrength.NONE,
            protected_evidence=EvidenceStrength.NONE,
            relationship_evidence=EvidenceStrength.NONE,
            discourse=DiscourseMode.DIRECT,
            continuation=ContinuationTarget.NONE,
            confidence=ConfidenceBand.LOW
        )

        assert result.nominated_mode == Mode.DAILY
        assert result.confidence_band == ConfidenceBand.LOW
        assert "default" in result.reason_codes[0]


class TestCompletenessAndConflict:
    """Task 3.3: Test decision table completeness and uniqueness."""

    def test_all_hard_events_handled(self):
        """All hard event values should produce valid decisions."""
        for hard_event in [HardEvent.NONE, HardEvent.BOUNDARY_EXIT]:
            result = nominate_mode(
                hard_event=hard_event,
                task_evidence=EvidenceStrength.NONE,
                protected_evidence=EvidenceStrength.NONE,
                relationship_evidence=EvidenceStrength.NONE,
                discourse=DiscourseMode.DIRECT,
                continuation=ContinuationTarget.NONE,
                confidence=ConfidenceBand.LOW
            )

            assert result.nominated_mode in (Mode.DAILY, Mode.WORK, Mode.SEX)
            assert len(result.reason_codes) > 0

    def test_priority_order_consistent(self):
        """Higher priority inputs should override lower priority."""
        # Hard event should override task evidence
        result_hard = nominate_mode(
            hard_event=HardEvent.BOUNDARY_EXIT,
            task_evidence=EvidenceStrength.EXPLICIT,
            protected_evidence=EvidenceStrength.NONE,
            relationship_evidence=EvidenceStrength.NONE,
            discourse=DiscourseMode.DIRECT,
            continuation=ContinuationTarget.NONE,
            confidence=ConfidenceBand.HIGH
        )

        result_task = nominate_mode(
            hard_event=HardEvent.NONE,
            task_evidence=EvidenceStrength.EXPLICIT,
            protected_evidence=EvidenceStrength.NONE,
            relationship_evidence=EvidenceStrength.NONE,
            discourse=DiscourseMode.DIRECT,
            continuation=ContinuationTarget.NONE,
            confidence=ConfidenceBand.HIGH
        )

        # Hard event should win
        assert result_hard.nominated_mode == Mode.DAILY
        assert result_task.nominated_mode == Mode.WORK

    def test_deterministic_output_for_same_input(self):
        """Same inputs should always produce identical outputs."""
        inputs = {
            "hard_event": HardEvent.NONE,
            "task_evidence": EvidenceStrength.STRONG,
            "protected_evidence": EvidenceStrength.NONE,
            "relationship_evidence": EvidenceStrength.WEAK,
            "discourse": DiscourseMode.DIRECT,
            "continuation": ContinuationTarget.NONE,
            "confidence": ConfidenceBand.MEDIUM,
        }

        results = [nominate_mode(**inputs) for _ in range(5)]

        # All should be identical
        first = results[0]
        for result in results[1:]:
            assert result.nominated_mode == first.nominated_mode
            assert result.force_event == first.force_event
            assert result.reason_codes == first.reason_codes


class TestShadowComparison:
    """Task 3.4: Test shadow-only behavior."""

    def test_shadow_classifier_hook_exists(self):
        """Should be able to import shadow comparison utilities."""
        # This test documents that shadow integration exists
        # Actual integration would be in mode_classifier.py
        from packages.persona_engine.persona_orchestrator.decision_tables import (
            nominate_mode
        )
        assert callable(nominate_mode)

    def test_nomination_preserves_legacy_interface(self):
        """New nomination should be callable independently of legacy."""
        # Shadow mode: new system runs but doesn't affect production
        result = nominate_mode(
            hard_event=HardEvent.NONE,
            task_evidence=EvidenceStrength.EXPLICIT,
            protected_evidence=EvidenceStrength.NONE,
            relationship_evidence=EvidenceStrength.NONE,
            discourse=DiscourseMode.DIRECT,
            continuation=ContinuationTarget.NONE,
            confidence=ConfidenceBand.HIGH
        )

        # Should return complete decision
        assert result.nominated_mode in (Mode.DAILY, Mode.WORK, Mode.SEX)
        assert isinstance(result.reason_codes, tuple)
        assert len(result.reason_codes) > 0

    def test_evidence_classification_composable(self):
        """Classification functions should be composable for shadow pipeline."""
        from packages.persona_engine.persona_orchestrator.rule_schema import (
            MatchEvidence,
            FeatureClass,
        )

        # Create sample evidence
        evidences = [
            MatchEvidence(
                rule_id="TEST.001",
                feature_class=FeatureClass.EXPLICIT_TASK,
                candidate_mode=Mode.WORK,
                weight=0.90,
                priority=800,
                clause_index=0
            )
        ]

        # Classify through pipeline
        hard_event = classify_hard_event(evidences)
        task_evidence = classify_task_evidence(evidences)
        discourse = classify_discourse(evidences)

        # Should produce valid classifications
        assert isinstance(hard_event, HardEvent)
        assert isinstance(task_evidence, EvidenceStrength)
        assert isinstance(discourse, DiscourseMode)

        # Should be usable for nomination
        result = nominate_mode(
            hard_event=hard_event,
            task_evidence=task_evidence,
            protected_evidence=EvidenceStrength.NONE,
            relationship_evidence=EvidenceStrength.NONE,
            discourse=discourse,
            continuation=ContinuationTarget.NONE,
            confidence=ConfidenceBand.MEDIUM
        )

        assert result.nominated_mode == Mode.WORK
