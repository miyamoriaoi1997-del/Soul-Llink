"""Tests for Phase 4 Task 4.3: Comprehensive evaluation metrics."""

import pytest
from packages.persona_engine.persona_orchestrator.evaluation_metrics import (
    MetricsCalculator,
    TransitionResult,
    SequenceResult,
)


class TestMetricsCalculator:
    """Test comprehensive metrics calculation."""

    def test_final_mode_accuracy(self):
        """Calculate final mode accuracy correctly."""
        calc = MetricsCalculator()

        calc.add_result(TransitionResult(
            case_id="c1",
            expected_mode="work",
            actual_mode="work",
            expected_nomination=None,
            actual_nomination=None,
            expected_transition="*",
            actual_transition="enter",
            expected_layers=[],
            actual_layers=[],
            forbidden_layers=[],
            previous_mode=None,
        ))

        calc.add_result(TransitionResult(
            case_id="c2",
            expected_mode="daily",
            actual_mode="work",  # Wrong
            expected_nomination=None,
            actual_nomination=None,
            expected_transition="*",
            actual_transition="enter",
            expected_layers=[],
            actual_layers=[],
            forbidden_layers=[],
            previous_mode=None,
        ))

        report = calc.calculate()

        assert report.total_cases == 2
        assert report.final_mode_accuracy == 0.5  # 1 out of 2 correct

    def test_nomination_accuracy_separate_from_final_mode(self):
        """Nomination accuracy tracks classifier output vs final active mode."""
        calc = MetricsCalculator()

        # Case where nomination is correct but final mode differs
        calc.add_result(TransitionResult(
            case_id="c1",
            expected_mode="work",
            actual_mode="work",
            expected_nomination="work",
            actual_nomination="work",
            expected_transition="*",
            actual_transition="daily->work",
            expected_layers=[],
            actual_layers=[],
            forbidden_layers=[],
            previous_mode="daily",
        ))

        # Case where nomination is wrong
        calc.add_result(TransitionResult(
            case_id="c2",
            expected_mode="work",
            actual_mode="work",
            expected_nomination="work",
            actual_nomination="daily",  # Wrong nomination
            expected_transition="*",
            actual_transition="hold",
            expected_layers=[],
            actual_layers=[],
            forbidden_layers=[],
            previous_mode="work",
        ))

        report = calc.calculate()

        assert report.nomination_accuracy == 0.5  # 1 out of 2 nominations correct
        assert report.final_mode_accuracy == 1.0  # Both final modes correct

    def test_transition_accuracy(self):
        """Calculate transition accuracy for non-wildcard cases."""
        calc = MetricsCalculator()

        calc.add_result(TransitionResult(
            case_id="c1",
            expected_mode="work",
            actual_mode="work",
            expected_nomination=None,
            actual_nomination=None,
            expected_transition="daily->work",
            actual_transition="daily->work",
            expected_layers=[],
            actual_layers=[],
            forbidden_layers=[],
            previous_mode="daily",
        ))

        calc.add_result(TransitionResult(
            case_id="c2",
            expected_mode="daily",
            actual_mode="daily",
            expected_nomination=None,
            actual_nomination=None,
            expected_transition="hold",
            actual_transition="enter",  # Wrong
            expected_layers=[],
            actual_layers=[],
            forbidden_layers=[],
            previous_mode=None,
        ))

        calc.add_result(TransitionResult(
            case_id="c3",
            expected_mode="work",
            actual_mode="work",
            expected_nomination=None,
            actual_nomination=None,
            expected_transition="*",  # Wildcard - not counted
            actual_transition="anything",
            expected_layers=[],
            actual_layers=[],
            forbidden_layers=[],
            previous_mode=None,
        ))

        report = calc.calculate()

        assert report.transition_accuracy == 0.5  # 1 out of 2 non-wildcard correct

    def test_layer_accuracy(self):
        """Calculate layer accuracy including forbidden layer violations."""
        calc = MetricsCalculator()

        # Correct layers
        calc.add_result(TransitionResult(
            case_id="c1",
            expected_mode="work",
            actual_mode="work",
            expected_nomination=None,
            actual_nomination=None,
            expected_transition="*",
            actual_transition="enter",
            expected_layers=["work"],
            actual_layers=["work"],
            forbidden_layers=["sex"],
            previous_mode=None,
        ))

        # Missing expected layer
        calc.add_result(TransitionResult(
            case_id="c2",
            expected_mode="work",
            actual_mode="work",
            expected_nomination=None,
            actual_nomination=None,
            expected_transition="*",
            actual_transition="enter",
            expected_layers=["work", "technical"],
            actual_layers=["work"],  # Missing technical
            forbidden_layers=[],
            previous_mode=None,
        ))

        # Forbidden layer present
        calc.add_result(TransitionResult(
            case_id="c3",
            expected_mode="daily",
            actual_mode="daily",
            expected_nomination=None,
            actual_nomination=None,
            expected_transition="*",
            actual_transition="enter",
            expected_layers=["daily"],
            actual_layers=["daily", "sex"],  # Has forbidden
            forbidden_layers=["sex"],
            previous_mode=None,
        ))

        report = calc.calculate()

        assert report.layer_accuracy == pytest.approx(1/3)  # Only first is correct

    def test_same_turn_switch_rate(self):
        """Calculate same-turn switching rate."""
        calc = MetricsCalculator()

        # Same-turn switch: previous != actual
        calc.add_result(TransitionResult(
            case_id="c1",
            expected_mode="work",
            actual_mode="work",
            expected_nomination=None,
            actual_nomination=None,
            expected_transition="daily->work",
            actual_transition="daily->work",
            expected_layers=[],
            actual_layers=[],
            forbidden_layers=[],
            previous_mode="daily",  # Switched
        ))

        # Hold: previous == actual
        calc.add_result(TransitionResult(
            case_id="c2",
            expected_mode="work",
            actual_mode="work",
            expected_nomination=None,
            actual_nomination=None,
            expected_transition="hold",
            actual_transition="hold",
            expected_layers=[],
            actual_layers=[],
            forbidden_layers=[],
            previous_mode="work",  # Held
        ))

        # No previous mode (enter)
        calc.add_result(TransitionResult(
            case_id="c3",
            expected_mode="daily",
            actual_mode="daily",
            expected_nomination=None,
            actual_nomination=None,
            expected_transition="enter",
            actual_transition="enter",
            expected_layers=[],
            actual_layers=[],
            forbidden_layers=[],
            previous_mode=None,  # Not counted
        ))

        report = calc.calculate()

        assert report.same_turn_switch_rate == 0.5  # 1 switch out of 2 with previous

    def test_switch_delay_metrics(self):
        """Calculate switch delay P50, P95, max."""
        calc = MetricsCalculator()

        # Sequence with various delays
        seq1 = SequenceResult(
            sequence_id="seq1",
            turns=[],
            expected_switch_turn=2,
            actual_switch_turn=2,
            switch_delay=0,
            max_allowed_delay=3,
        )

        seq2 = SequenceResult(
            sequence_id="seq2",
            turns=[],
            expected_switch_turn=1,
            actual_switch_turn=3,
            switch_delay=2,
            max_allowed_delay=3,
        )

        seq3 = SequenceResult(
            sequence_id="seq3",
            turns=[],
            expected_switch_turn=0,
            actual_switch_turn=5,
            switch_delay=5,
            max_allowed_delay=3,
        )

        calc.add_sequence(seq1)
        calc.add_sequence(seq2)
        calc.add_sequence(seq3)

        report = calc.calculate()

        assert report.switch_delay_p50 == 2.0  # Median of [0, 2, 5]
        assert report.switch_delay_max == 5
        assert report.stale_hold_count == 1  # seq3 exceeds max_allowed_delay
        assert report.max_stale_duration == 5

    def test_confusion_matrices(self):
        """Build mode and transition confusion matrices."""
        calc = MetricsCalculator()

        calc.add_result(TransitionResult(
            case_id="c1",
            expected_mode="work",
            actual_mode="work",
            expected_nomination=None,
            actual_nomination=None,
            expected_transition="daily->work",
            actual_transition="daily->work",
            expected_layers=[],
            actual_layers=[],
            forbidden_layers=[],
            previous_mode="daily",
        ))

        calc.add_result(TransitionResult(
            case_id="c2",
            expected_mode="work",
            actual_mode="daily",  # Confusion
            expected_nomination=None,
            actual_nomination=None,
            expected_transition="hold",
            actual_transition="exit",
            expected_layers=[],
            actual_layers=[],
            forbidden_layers=[],
            previous_mode="work",
        ))

        report = calc.calculate()

        assert report.mode_confusion[("work", "work")] == 1
        assert report.mode_confusion[("work", "daily")] == 1
        assert report.transition_confusion[("daily->work", "daily->work")] == 1
        assert report.transition_confusion[("hold", "exit")] == 1

    def test_high_risk_error_counting(self):
        """Count false positives and negatives for high-risk transitions."""
        calc = MetricsCalculator()

        # False positive: entered sex when shouldn't
        calc.add_result(TransitionResult(
            case_id="c1",
            expected_mode="daily",
            actual_mode="sex",
            expected_nomination=None,
            actual_nomination=None,
            expected_transition="*",
            actual_transition="daily->sex",
            expected_layers=[],
            actual_layers=[],
            forbidden_layers=[],
            previous_mode="daily",
            risk_class="high",
        ))

        # False negative: should be sex but isn't
        calc.add_result(TransitionResult(
            case_id="c2",
            expected_mode="sex",
            actual_mode="daily",
            expected_nomination=None,
            actual_nomination=None,
            expected_transition="*",
            actual_transition="hold",
            expected_layers=[],
            actual_layers=[],
            forbidden_layers=[],
            previous_mode="daily",
            risk_class="critical",
        ))

        # Correct classification (not an error)
        calc.add_result(TransitionResult(
            case_id="c3",
            expected_mode="sex",
            actual_mode="sex",
            expected_nomination=None,
            actual_nomination=None,
            expected_transition="*",
            actual_transition="enter",
            expected_layers=[],
            actual_layers=[],
            forbidden_layers=[],
            previous_mode=None,
            risk_class="high",
        ))

        report = calc.calculate()

        assert report.high_risk_false_positive == 1
        assert report.high_risk_false_negative == 1

    def test_protected_mode_errors(self):
        """Count all sex/protected mode misclassifications."""
        calc = MetricsCalculator()

        # Error: expected sex, got daily
        calc.add_result(TransitionResult(
            case_id="c1",
            expected_mode="sex",
            actual_mode="daily",
            expected_nomination=None,
            actual_nomination=None,
            expected_transition="*",
            actual_transition="exit",
            expected_layers=[],
            actual_layers=[],
            forbidden_layers=[],
            previous_mode="sex",
        ))

        # Error: expected daily, got sex
        calc.add_result(TransitionResult(
            case_id="c2",
            expected_mode="daily",
            actual_mode="sex",
            expected_nomination=None,
            actual_nomination=None,
            expected_transition="*",
            actual_transition="enter",
            expected_layers=[],
            actual_layers=[],
            forbidden_layers=[],
            previous_mode=None,
        ))

        # Correct: both sex
        calc.add_result(TransitionResult(
            case_id="c3",
            expected_mode="sex",
            actual_mode="sex",
            expected_nomination=None,
            actual_nomination=None,
            expected_transition="*",
            actual_transition="hold",
            expected_layers=[],
            actual_layers=[],
            forbidden_layers=[],
            previous_mode="sex",
        ))

        report = calc.calculate()

        assert report.protected_mode_errors == 2

    def test_latency_metrics(self):
        """Calculate latency P50, P95, max."""
        calc = MetricsCalculator()

        for i, latency in enumerate([10.0, 20.0, 30.0, 40.0, 100.0]):
            calc.add_result(TransitionResult(
                case_id=f"c{i}",
                expected_mode="daily",
                actual_mode="daily",
                expected_nomination=None,
                actual_nomination=None,
                expected_transition="*",
                actual_transition="enter",
                expected_layers=[],
                actual_layers=[],
                forbidden_layers=[],
                previous_mode=None,
                latency_ms=latency,
            ))

        report = calc.calculate()

        assert report.latency_p50_ms == 30.0  # Median
        assert report.latency_max_ms == 100.0
        assert report.latency_p95_ms is not None

    def test_pair_specific_confusion(self):
        """Generate confusion by (previous, expected) pair."""
        calc = MetricsCalculator()

        calc.add_result(TransitionResult(
            case_id="c1",
            expected_mode="work",
            actual_mode="work",
            expected_nomination=None,
            actual_nomination=None,
            expected_transition="*",
            actual_transition="enter",
            expected_layers=[],
            actual_layers=[],
            forbidden_layers=[],
            previous_mode="daily",
        ))

        calc.add_result(TransitionResult(
            case_id="c2",
            expected_mode="work",
            actual_mode="daily",  # Error
            expected_nomination=None,
            actual_nomination=None,
            expected_transition="*",
            actual_transition="hold",
            expected_layers=[],
            actual_layers=[],
            forbidden_layers=[],
            previous_mode="daily",
        ))

        pair_confusion = calc.generate_pair_specific_confusion()

        assert "daily->work" in pair_confusion
        assert pair_confusion["daily->work"][("work", "work")] == 1
        assert pair_confusion["daily->work"][("work", "daily")] == 1

    def test_risk_breakdown(self):
        """Break down accuracy by risk class."""
        calc = MetricsCalculator()

        calc.add_result(TransitionResult(
            case_id="c1",
            expected_mode="work",
            actual_mode="work",
            expected_nomination=None,
            actual_nomination=None,
            expected_transition="*",
            actual_transition="enter",
            expected_layers=[],
            actual_layers=[],
            forbidden_layers=[],
            previous_mode=None,
            risk_class="low",
        ))

        calc.add_result(TransitionResult(
            case_id="c2",
            expected_mode="sex",
            actual_mode="daily",  # Error
            expected_nomination=None,
            actual_nomination=None,
            expected_transition="*",
            actual_transition="exit",
            expected_layers=[],
            actual_layers=[],
            forbidden_layers=[],
            previous_mode="sex",
            risk_class="high",
        ))

        breakdown = calc.get_risk_breakdown()

        assert breakdown["low"]["total"] == 1
        assert breakdown["low"]["correct"] == 1
        assert breakdown["high"]["total"] == 1
        assert breakdown["high"]["correct"] == 0
