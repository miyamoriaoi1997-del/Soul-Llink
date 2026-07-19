"""Phase 4 Task 4.4: Combinatorial test generation.

Generate test cases using NIST pairwise combinatorial testing to cover
parameter interactions efficiently. For critical paths (hard-boundary,
protected transitions), use 3-way coverage.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from itertools import product
from typing import Any, Iterator


class PreviousMode(Enum):
    """Previous mode values."""
    NONE = "none"
    DAILY = "daily"
    WORK = "work"
    SEX = "sex"


class NominationMode(Enum):
    """Nominated mode values."""
    DAILY = "daily"
    WORK = "work"
    SEX = "sex"


class ExplicitTask(Enum):
    """Explicit task evidence."""
    NO = "no"
    YES = "yes"


class HardExit(Enum):
    """Hard boundary exit."""
    NO = "no"
    YES = "yes"


class ContinuationTarget(Enum):
    """Continuation binding target."""
    NONE = "none"
    WORK = "work"
    RELATIONSHIP = "relationship"
    AMBIGUOUS = "ambiguous"


class DiscourseMode(Enum):
    """Discourse context."""
    DIRECT = "direct"
    QUOTED = "quoted"
    NEGATED = "negated"
    HYPOTHETICAL = "hypothetical"
    META = "meta"


class GateStatus(Enum):
    """Protected mode gate status."""
    NONE = "none"
    PASS = "pass"
    FAIL = "fail"


class ConfidenceBand(Enum):
    """Confidence level."""
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"


@dataclass(frozen=True)
class CombinatorialCase:
    """Generated combinatorial test case."""
    case_id: str
    previous_mode: PreviousMode
    nomination: NominationMode
    explicit_task: ExplicitTask
    hard_exit: HardExit
    continuation_target: ContinuationTarget
    discourse: DiscourseMode
    gate_status: GateStatus
    confidence: ConfidenceBand

    # Derived expectations (computed based on decision table rules)
    expected_mode: str
    expected_transition_type: str  # "enter", "hold", "switch", "exit"
    invariant_checks: list[str]


class PairwiseGenerator:
    """Generate pairwise covering test cases.

    Simplified pairwise algorithm: ensure each pair of parameter values
    appears together in at least one test case.
    """

    @staticmethod
    def generate_pairwise(
        parameters: dict[str, list[Any]]
    ) -> list[dict[str, Any]]:
        """Generate pairwise covering test cases.

        This is a greedy algorithm that builds test cases by selecting
        parameter values that cover the most uncovered pairs.
        """
        param_names = list(parameters.keys())
        param_values = [parameters[name] for name in param_names]

        # Track all possible pairs
        uncovered_pairs: set[tuple[str, str, Any, Any]] = set()
        for i, name1 in enumerate(param_names):
            for j, name2 in enumerate(param_names):
                if i < j:
                    for val1 in param_values[i]:
                        for val2 in param_values[j]:
                            uncovered_pairs.add((name1, name2, val1, val2))

        test_cases = []
        max_iterations = 1000  # Safety limit
        iteration = 0

        while uncovered_pairs and iteration < max_iterations:
            iteration += 1

            # Build a test case that covers the most uncovered pairs
            best_case = None
            best_coverage = 0

            # Try different combinations
            for combo in product(*param_values):
                case = dict(zip(param_names, combo))
                coverage = PairwiseGenerator._count_coverage(case, uncovered_pairs)

                if coverage > best_coverage:
                    best_coverage = coverage
                    best_case = case

            if best_case:
                test_cases.append(best_case)
                # Remove covered pairs
                PairwiseGenerator._remove_covered_pairs(best_case, uncovered_pairs)
            else:
                break  # No more coverage possible

        return test_cases

    @staticmethod
    def _count_coverage(
        test_case: dict[str, Any],
        uncovered_pairs: set[tuple[str, str, Any, Any]]
    ) -> int:
        """Count how many uncovered pairs this test case would cover."""
        count = 0
        param_names = list(test_case.keys())

        for i, name1 in enumerate(param_names):
            for j, name2 in enumerate(param_names):
                if i < j:
                    val1 = test_case[name1]
                    val2 = test_case[name2]
                    if (name1, name2, val1, val2) in uncovered_pairs:
                        count += 1

        return count

    @staticmethod
    def _remove_covered_pairs(
        test_case: dict[str, Any],
        uncovered_pairs: set[tuple[str, str, Any, Any]]
    ):
        """Remove pairs covered by this test case."""
        param_names = list(test_case.keys())

        for i, name1 in enumerate(param_names):
            for j, name2 in enumerate(param_names):
                if i < j:
                    val1 = test_case[name1]
                    val2 = test_case[name2]
                    uncovered_pairs.discard((name1, name2, val1, val2))


class ThreeWayGenerator:
    """Generate 3-way covering test cases for critical paths."""

    @staticmethod
    def generate_full_combinations(
        parameters: dict[str, list[Any]]
    ) -> list[dict[str, Any]]:
        """Generate all combinations (for small critical parameter sets)."""
        param_names = list(parameters.keys())
        param_values = [parameters[name] for name in param_names]

        test_cases = []
        for combo in product(*param_values):
            test_cases.append(dict(zip(param_names, combo)))

        return test_cases


class CombinatorialTestSuite:
    """Generate full combinatorial test suite for state machine."""

    def generate_standard_suite(self) -> list[CombinatorialCase]:
        """Generate pairwise test suite for standard transitions."""
        parameters = {
            "previous_mode": [e.value for e in PreviousMode],
            "nomination": [e.value for e in NominationMode],
            "explicit_task": [e.value for e in ExplicitTask],
            "hard_exit": [e.value for e in HardExit],
            "continuation_target": [e.value for e in ContinuationTarget],
            "discourse": [e.value for e in DiscourseMode],
            "gate_status": [e.value for e in GateStatus],
            "confidence": [e.value for e in ConfidenceBand],
        }

        raw_cases = PairwiseGenerator.generate_pairwise(parameters)

        # Convert to CombinatorialCase with expectations
        cases = []
        for idx, raw in enumerate(raw_cases):
            case = self._build_case(f"pairwise_{idx:03d}", raw)
            cases.append(case)

        return cases

    def generate_critical_suite(self) -> list[CombinatorialCase]:
        """Generate 3-way suite for critical hard-boundary transitions."""
        # Focus on critical parameters only
        parameters = {
            "previous_mode": ["daily", "work", "sex"],
            "nomination": ["daily", "work", "sex"],
            "hard_exit": ["no", "yes"],
        }

        raw_cases = ThreeWayGenerator.generate_full_combinations(parameters)

        # Fill in other parameters with safe defaults
        cases = []
        for idx, raw in enumerate(raw_cases):
            raw["explicit_task"] = "no"
            raw["continuation_target"] = "none"
            raw["discourse"] = "direct"
            raw["gate_status"] = "none"
            raw["confidence"] = "medium"

            case = self._build_case(f"critical_{idx:03d}", raw)
            cases.append(case)

        return cases

    def _build_case(self, case_id: str, params: dict[str, str]) -> CombinatorialCase:
        """Build a CombinatorialCase with expected outcomes."""
        # Apply decision table logic to compute expectations
        expected_mode, transition_type, invariants = self._compute_expectations(params)

        return CombinatorialCase(
            case_id=case_id,
            previous_mode=PreviousMode(params["previous_mode"]),
            nomination=NominationMode(params["nomination"]),
            explicit_task=ExplicitTask(params["explicit_task"]),
            hard_exit=HardExit(params["hard_exit"]),
            continuation_target=ContinuationTarget(params["continuation_target"]),
            discourse=DiscourseMode(params["discourse"]),
            gate_status=GateStatus(params["gate_status"]),
            confidence=ConfidenceBand(params["confidence"]),
            expected_mode=expected_mode,
            expected_transition_type=transition_type,
            invariant_checks=invariants,
        )

    def _compute_expectations(
        self,
        params: dict[str, str]
    ) -> tuple[str, str, list[str]]:
        """Compute expected outcomes based on decision table rules.

        This encodes the nominal decision logic for validation.
        """
        invariants = []

        # Rule 1: Hard exit always goes to daily
        if params["hard_exit"] == "yes":
            return "daily", "exit", ["hard_exit_forces_daily"]

        # Rule 2: Explicit task with direct discourse goes to work
        if params["explicit_task"] == "yes" and params["discourse"] == "direct":
            return "work", "switch", ["explicit_task_enters_work"]

        # Rule 3: Protected nomination requires gate pass
        if params["nomination"] == "sex":
            if params["gate_status"] == "pass":
                return "sex", "switch", ["protected_gate_required"]
            else:
                # Gate failed or none - hold previous or default daily
                prev = params["previous_mode"]
                if prev != "none":
                    return prev, "hold", ["protected_gate_blocked"]
                else:
                    return "daily", "enter", ["protected_gate_blocked_default"]

        # Rule 4: Work nomination generally accepted
        if params["nomination"] == "work":
            return "work", "switch", ["work_nomination_accepted"]

        # Rule 5: Daily nomination or fallback
        if params["nomination"] == "daily":
            return "daily", "switch", ["daily_nomination"]

        # Default: stay in previous or enter daily
        prev = params["previous_mode"]
        if prev != "none":
            return prev, "hold", ["default_hold"]
        else:
            return "daily", "enter", ["default_enter"]
