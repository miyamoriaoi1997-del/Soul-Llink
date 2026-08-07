"""Tests for Phase 4 Task 4.4: Combinatorial test generation."""

import pytest
from persona_orchestrator.combinatorial_tests import (
    PairwiseGenerator,
    ThreeWayGenerator,
    CombinatorialTestSuite,
    PreviousMode,
    NominationMode,
    HardExit,
)


class TestPairwiseGenerator:
    """Test pairwise combinatorial coverage."""

    def test_pairwise_covers_all_pairs(self):
        """Pairwise generator covers all parameter value pairs."""
        parameters = {
            "param1": ["a", "b", "c"],
            "param2": ["x", "y"],
            "param3": ["1", "2"],
        }

        cases = PairwiseGenerator.generate_pairwise(parameters)

        # Check all pairs between param1 and param2 are covered
        pairs_12 = set()
        for case in cases:
            pairs_12.add((case["param1"], case["param2"]))

        expected_pairs_12 = {
            ("a", "x"), ("a", "y"),
            ("b", "x"), ("b", "y"),
            ("c", "x"), ("c", "y"),
        }
        assert pairs_12 == expected_pairs_12

        # Check all pairs between param2 and param3 are covered
        pairs_23 = set()
        for case in cases:
            pairs_23.add((case["param2"], case["param3"]))

        expected_pairs_23 = {
            ("x", "1"), ("x", "2"),
            ("y", "1"), ("y", "2"),
        }
        assert pairs_23 == expected_pairs_23

    def test_pairwise_fewer_cases_than_full_combination(self):
        """Pairwise generates fewer cases than exhaustive."""
        parameters = {
            "p1": ["a", "b", "c"],
            "p2": ["x", "y", "z"],
            "p3": ["1", "2", "3"],
        }

        pairwise_cases = PairwiseGenerator.generate_pairwise(parameters)
        full_combinations = 3 * 3 * 3  # 27

        # Pairwise should be significantly smaller
        assert len(pairwise_cases) < full_combinations
        assert len(pairwise_cases) <= 15  # Typical pairwise size

    def test_pairwise_handles_binary_parameters(self):
        """Pairwise works with binary (yes/no) parameters."""
        parameters = {
            "hard_exit": ["no", "yes"],
            "explicit_task": ["no", "yes"],
            "gate_pass": ["no", "yes"],
        }

        cases = PairwiseGenerator.generate_pairwise(parameters)

        # All 4 combinations of hard_exit x explicit_task must be present
        pairs = {(c["hard_exit"], c["explicit_task"]) for c in cases}
        assert len(pairs) == 4  # All combinations of 2 binary params


class TestThreeWayGenerator:
    """Test 3-way combinatorial generation for critical paths."""

    def test_full_combinations_exhaustive(self):
        """ThreeWay generates all combinations."""
        parameters = {
            "p1": ["a", "b"],
            "p2": ["x", "y"],
            "p3": ["1", "2"],
        }

        cases = ThreeWayGenerator.generate_full_combinations(parameters)

        assert len(cases) == 2 * 2 * 2  # 8 full combinations

        # Verify uniqueness
        case_tuples = {tuple(sorted(c.items())) for c in cases}
        assert len(case_tuples) == 8

    def test_critical_path_small_enough_for_full_coverage(self):
        """Critical path parameters small enough for exhaustive testing."""
        parameters = {
            "previous_mode": ["daily", "work", "sex"],
            "nomination": ["daily", "work", "sex"],
            "hard_exit": ["no", "yes"],
        }

        cases = ThreeWayGenerator.generate_full_combinations(parameters)

        # 3 * 3 * 2 = 18 combinations (manageable)
        assert len(cases) == 18


class TestCombinatorialTestSuite:
    """Test full test suite generation."""

    def test_generate_standard_suite(self):
        """Standard suite uses pairwise coverage."""
        suite = CombinatorialTestSuite()
        cases = suite.generate_standard_suite()

        # Should generate pairwise cases
        assert len(cases) > 0
        assert len(cases) < 1000  # Not exhaustive

        # Each case has all required fields
        for case in cases:
            assert case.case_id.startswith("pairwise_")
            assert case.expected_mode in ("daily", "work", "sex")
            assert case.expected_transition_type in ("enter", "hold", "switch", "exit")
            assert len(case.invariant_checks) > 0

    def test_generate_critical_suite(self):
        """Critical suite uses 3-way coverage for hard boundaries."""
        suite = CombinatorialTestSuite()
        cases = suite.generate_critical_suite()

        # Should have 3*3*2 = 18 cases
        assert len(cases) == 18

        # All combinations of critical params present
        critical_combos = set()
        for case in cases:
            critical_combos.add((
                case.previous_mode.value,
                case.nomination.value,
                case.hard_exit.value,
            ))

        assert len(critical_combos) == 18

    def test_hard_exit_invariant_always_forces_daily(self):
        """Hard exit must always result in daily mode."""
        suite = CombinatorialTestSuite()
        cases = suite.generate_standard_suite()

        hard_exit_cases = [c for c in cases if c.hard_exit == HardExit.YES]

        for case in hard_exit_cases:
            assert case.expected_mode == "daily"
            assert "hard_exit_forces_daily" in case.invariant_checks

    def test_protected_mode_requires_gate(self):
        """Sex/protected mode nomination requires gate pass."""
        suite = CombinatorialTestSuite()
        cases = suite.generate_standard_suite()

        sex_nomination_cases = [c for c in cases if c.nomination == NominationMode.SEX]

        for case in sex_nomination_cases:
            if case.gate_status.value == "pass":
                assert case.expected_mode == "sex"
                assert "protected_gate_required" in case.invariant_checks
            else:
                # Should not enter sex without gate
                assert case.expected_mode != "sex" or "protected_gate_blocked" in case.invariant_checks

    def test_explicit_task_with_direct_discourse_enters_work(self):
        """Explicit task + direct discourse should nominate work."""
        suite = CombinatorialTestSuite()

        # Build a specific case
        params = {
            "previous_mode": "daily",
            "nomination": "work",
            "explicit_task": "yes",
            "hard_exit": "no",
            "continuation_target": "none",
            "discourse": "direct",
            "gate_status": "none",
            "confidence": "high",
        }

        case = suite._build_case("test", params)

        assert case.expected_mode == "work"
        assert "explicit_task_enters_work" in case.invariant_checks

    def test_all_cases_have_valid_expectations(self):
        """All generated cases have valid expected outcomes."""
        suite = CombinatorialTestSuite()

        standard = suite.generate_standard_suite()
        critical = suite.generate_critical_suite()

        for case in standard + critical:
            # Expected mode must be valid
            assert case.expected_mode in ("daily", "work", "sex")

            # Transition type must be valid
            assert case.expected_transition_type in ("enter", "hold", "switch", "exit")

            # Must have at least one invariant
            assert len(case.invariant_checks) > 0

            # Invariants must be non-empty strings
            for inv in case.invariant_checks:
                assert isinstance(inv, str)
                assert len(inv) > 0
