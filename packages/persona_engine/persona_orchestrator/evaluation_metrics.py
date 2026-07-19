"""Phase 4 Task 4.3: Comprehensive evaluation metrics.

Separate reporting for nomination accuracy, final mode accuracy, transition accuracy,
switch timing, pair-specific confusion, and high-risk transitions.
"""

from __future__ import annotations

from collections import Counter, defaultdict
from dataclasses import dataclass, field
from typing import Any, Optional
import statistics


@dataclass
class TransitionResult:
    """Single turn evaluation result."""
    case_id: str
    expected_mode: str
    actual_mode: str
    expected_nomination: Optional[str]
    actual_nomination: Optional[str]
    expected_transition: str
    actual_transition: str
    expected_layers: list[str]
    actual_layers: list[str]
    forbidden_layers: list[str]
    previous_mode: Optional[str]
    turn_index: Optional[int] = None
    sequence_id: Optional[str] = None
    risk_class: Optional[str] = None
    latency_ms: Optional[float] = None


@dataclass
class SequenceResult:
    """Multi-turn sequence evaluation result."""
    sequence_id: str
    turns: list[TransitionResult]
    expected_switch_turn: Optional[int]
    actual_switch_turn: Optional[int]
    switch_delay: Optional[int]  # Actual - expected
    max_allowed_delay: Optional[int]


@dataclass
class MetricsReport:
    """Comprehensive metrics report."""

    # Accuracy metrics
    nomination_accuracy: float = 0.0
    final_mode_accuracy: float = 0.0
    transition_accuracy: float = 0.0
    layer_accuracy: float = 0.0

    # Transition timing
    same_turn_switch_rate: float = 0.0
    switch_delay_p50: Optional[float] = None
    switch_delay_p95: Optional[float] = None
    switch_delay_max: Optional[int] = None

    # Stale hold analysis
    stale_hold_count: int = 0
    avg_recovery_turns: Optional[float] = None
    max_stale_duration: Optional[int] = None

    # Rapid reversal (mode changes too quickly)
    rapid_reversal_count: int = 0

    # Confusion matrices
    mode_confusion: dict[tuple[str, str], int] = field(default_factory=dict)
    transition_confusion: dict[tuple[str, str], int] = field(default_factory=dict)

    # High-risk metrics
    high_risk_false_positive: int = 0
    high_risk_false_negative: int = 0
    protected_mode_errors: int = 0

    # Layer/model agreement
    layer_correctness: float = 0.0
    model_agreement: float = 0.0

    # Decision table coverage
    no_hit_count: int = 0
    multi_hit_count: int = 0
    abstain_count: int = 0

    # Rule coverage
    rules_hit: set[str] = field(default_factory=set)
    dead_rules: set[str] = field(default_factory=set)
    conflict_count: int = 0

    # Latency
    latency_p50_ms: Optional[float] = None
    latency_p95_ms: Optional[float] = None
    latency_max_ms: Optional[float] = None

    # Sample counts
    total_cases: int = 0
    total_sequences: int = 0
    total_turns: int = 0


class MetricsCalculator:
    """Calculate comprehensive evaluation metrics."""

    def __init__(self):
        self.results: list[TransitionResult] = []
        self.sequences: list[SequenceResult] = []

    def add_result(self, result: TransitionResult):
        """Add a single turn result."""
        self.results.append(result)

    def add_sequence(self, sequence: SequenceResult):
        """Add a multi-turn sequence result."""
        self.sequences.append(sequence)
        self.results.extend(sequence.turns)

    def calculate(self) -> MetricsReport:
        """Calculate all metrics from accumulated results."""
        report = MetricsReport()

        report.total_cases = len(self.results)
        report.total_sequences = len(self.sequences)
        report.total_turns = len(self.results)

        # Early return only if no data at all
        if not self.results and not self.sequences:
            return report

        # Accuracy metrics (only if we have results)
        if self.results:
            mode_correct = sum(1 for r in self.results if r.actual_mode == r.expected_mode)
            report.final_mode_accuracy = mode_correct / len(self.results)

        # Nomination accuracy (only for cases with expected_nomination)
        nomination_cases = [r for r in self.results if r.expected_nomination]
        if nomination_cases:
            nomination_correct = sum(
                1 for r in nomination_cases
                if r.actual_nomination == r.expected_nomination
            )
            report.nomination_accuracy = nomination_correct / len(nomination_cases)

        # Transition accuracy
        transition_cases = [r for r in self.results if r.expected_transition != "*"]
        if transition_cases:
            transition_correct = sum(
                1 for r in transition_cases
                if r.actual_transition == r.expected_transition
            )
            report.transition_accuracy = transition_correct / len(transition_cases)

        # Layer accuracy
        layer_cases = [r for r in self.results if r.expected_layers]
        if layer_cases:
            layer_correct = sum(
                1 for r in layer_cases
                if set(r.actual_layers) == set(r.expected_layers)
                and not any(f in r.actual_layers for f in r.forbidden_layers)
            )
            report.layer_accuracy = layer_correct / len(layer_cases)

        # Confusion matrices
        for r in self.results:
            report.mode_confusion[(r.expected_mode, r.actual_mode)] = \
                report.mode_confusion.get((r.expected_mode, r.actual_mode), 0) + 1

            if r.expected_transition != "*":
                report.transition_confusion[(r.expected_transition, r.actual_transition)] = \
                    report.transition_confusion.get((r.expected_transition, r.actual_transition), 0) + 1

        # Same-turn switching (transition happens in same turn)
        same_turn_switches = sum(
            1 for r in self.results
            if r.previous_mode and r.actual_mode != r.previous_mode
        )
        cases_with_previous = sum(1 for r in self.results if r.previous_mode)
        if cases_with_previous > 0:
            report.same_turn_switch_rate = same_turn_switches / cases_with_previous

        # Sequence-based timing metrics
        if self.sequences:
            switch_delays = [
                seq.switch_delay for seq in self.sequences
                if seq.switch_delay is not None and seq.switch_delay >= 0
            ]
            if switch_delays:
                report.switch_delay_p50 = statistics.median(switch_delays)
                report.switch_delay_p95 = statistics.quantiles(switch_delays, n=20)[18] if len(switch_delays) > 1 else switch_delays[0]
                report.switch_delay_max = max(switch_delays)

            # Stale hold: delay exceeds max_allowed_delay
            stale_seqs = [
                seq for seq in self.sequences
                if seq.switch_delay is not None
                and seq.max_allowed_delay is not None
                and seq.switch_delay > seq.max_allowed_delay
            ]
            report.stale_hold_count = len(stale_seqs)
            if stale_seqs:
                report.max_stale_duration = max(seq.switch_delay for seq in stale_seqs)

        # High-risk error counting
        high_risk_cases = [r for r in self.results if r.risk_class in ("high", "critical")]
        for r in high_risk_cases:
            if r.actual_mode != r.expected_mode:
                # False positive: entered when shouldn't
                if r.actual_mode == "sex" and r.expected_mode != "sex":
                    report.high_risk_false_positive += 1
                # False negative: didn't enter when should
                elif r.expected_mode == "sex" and r.actual_mode != "sex":
                    report.high_risk_false_negative += 1

        # Protected mode errors (sex/protected misclassification)
        protected_cases = [
            r for r in self.results
            if "sex" in [r.expected_mode, r.actual_mode]
        ]
        report.protected_mode_errors = sum(
            1 for r in protected_cases
            if r.actual_mode != r.expected_mode
        )

        # Latency metrics
        latencies = [r.latency_ms for r in self.results if r.latency_ms is not None]
        if latencies:
            report.latency_p50_ms = statistics.median(latencies)
            report.latency_p95_ms = statistics.quantiles(latencies, n=20)[18] if len(latencies) > 1 else latencies[0]
            report.latency_max_ms = max(latencies)

        return report

    def generate_pair_specific_confusion(self) -> dict[str, dict[tuple[str, str], int]]:
        """Generate confusion matrix for each (previous_mode, expected_mode) pair."""
        pair_confusion: dict[str, dict[tuple[str, str], int]] = defaultdict(lambda: defaultdict(int))

        for r in self.results:
            if r.previous_mode:
                pair_key = f"{r.previous_mode}->{r.expected_mode}"
                pair_confusion[pair_key][(r.expected_mode, r.actual_mode)] += 1

        return dict(pair_confusion)

    def get_risk_breakdown(self) -> dict[str, dict[str, int]]:
        """Break down accuracy by risk class."""
        breakdown: dict[str, dict[str, int]] = defaultdict(lambda: {"correct": 0, "total": 0})

        for r in self.results:
            risk = r.risk_class or "unclassified"
            breakdown[risk]["total"] += 1
            if r.actual_mode == r.expected_mode:
                breakdown[risk]["correct"] += 1

        return dict(breakdown)
