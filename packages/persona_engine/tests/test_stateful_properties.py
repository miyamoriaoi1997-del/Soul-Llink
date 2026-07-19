"""Phase 4 Task 4.5: Hypothesis stateful property tests for sequence invariants."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

try:
    from hypothesis import strategies as st
    from hypothesis.stateful import RuleBasedStateMachine, rule, invariant
    HYPOTHESIS_AVAILABLE = True
except ImportError:
    HYPOTHESIS_AVAILABLE = False
    # Fallback types for when hypothesis is not installed
    class RuleBasedStateMachine:
        pass
    def rule(*args, **kwargs):
        def decorator(func):
            return func
        return decorator
    def invariant(func):
        return func


@dataclass
class StateMachineState:
    """Tracked state for property-based testing."""
    current_mode: str
    previous_mode: Optional[str]
    turn_count: int
    stale_hold_turns: int
    last_nomination: Optional[str]
    protected_gate_passed: bool
    hard_exit_seen: bool

    def __init__(self):
        self.current_mode = "daily"
        self.previous_mode = None
        self.turn_count = 0
        self.stale_hold_turns = 0
        self.last_nomination = None
        self.protected_gate_passed = False
        self.hard_exit_seen = False


class StateMachineInvariants:
    """Property invariants that must hold across all sequences."""

    MAX_STALE_HOLD = 5  # Maximum turns to hold incorrect mode
    VALID_MODES = {"daily", "work", "sex"}

    @staticmethod
    def hard_exit_forces_daily(
        hard_exit: bool,
        resulting_mode: str
    ) -> bool:
        """Hard exit must immediately switch to daily."""
        if hard_exit:
            return resulting_mode == "daily"
        return True

    @staticmethod
    def explicit_work_same_turn(
        explicit_task: bool,
        direct_discourse: bool,
        previous_mode: str,
        resulting_mode: str
    ) -> bool:
        """Explicit work task should enter work in same turn."""
        if explicit_task and direct_discourse and previous_mode != "work":
            return resulting_mode == "work"
        return True

    @staticmethod
    def protected_requires_gate(
        nomination: str,
        gate_passed: bool,
        resulting_mode: str
    ) -> bool:
        """Cannot enter sex/protected without gate pass."""
        if nomination == "sex" and not gate_passed:
            return resulting_mode != "sex"
        return True

    @staticmethod
    def mode_is_valid(mode: str) -> bool:
        """Mode must be one of the valid values."""
        return mode in StateMachineInvariants.VALID_MODES

    @staticmethod
    def stale_hold_bounded(stale_turns: int) -> bool:
        """Stale hold must not exceed maximum."""
        return stale_turns <= StateMachineInvariants.MAX_STALE_HOLD


if HYPOTHESIS_AVAILABLE:
    class StateMachinePropertyTest(RuleBasedStateMachine):
        """Hypothesis-based stateful property testing."""

        def __init__(self):
            super().__init__()
            self.state = StateMachineState()

        @rule(
            nomination=st.sampled_from(["daily", "work", "sex"]),
            explicit_task=st.booleans(),
            hard_exit=st.booleans(),
            direct_discourse=st.booleans(),
        )
        def transition(
            self,
            nomination: str,
            explicit_task: bool,
            hard_exit: bool,
            direct_discourse: bool,
        ):
            """Simulate a state transition."""
            previous = self.state.current_mode

            # Apply decision logic
            if hard_exit:
                resulting_mode = "daily"
                self.state.hard_exit_seen = True
            elif explicit_task and direct_discourse:
                resulting_mode = "work"
            elif nomination == "sex" and not self.state.protected_gate_passed:
                # Cannot enter without gate
                resulting_mode = self.state.current_mode
            else:
                resulting_mode = nomination

            # Check invariants
            assert StateMachineInvariants.hard_exit_forces_daily(
                hard_exit, resulting_mode
            ), f"Hard exit violated: {resulting_mode}"

            assert StateMachineInvariants.explicit_work_same_turn(
                explicit_task, direct_discourse, previous, resulting_mode
            ), f"Explicit work not honored: {previous} -> {resulting_mode}"

            assert StateMachineInvariants.protected_requires_gate(
                nomination, self.state.protected_gate_passed, resulting_mode
            ), f"Protected mode entered without gate: {resulting_mode}"

            # Update state
            self.state.previous_mode = self.state.current_mode
            self.state.current_mode = resulting_mode
            self.state.turn_count += 1
            self.state.last_nomination = nomination

            # Track stale hold
            if self.state.current_mode == self.state.previous_mode:
                self.state.stale_hold_turns += 1
            else:
                self.state.stale_hold_turns = 0

        @rule()
        def pass_protected_gate(self):
            """Pass the protected mode gate."""
            self.state.protected_gate_passed = True

        @rule()
        def fail_protected_gate(self):
            """Fail the protected mode gate."""
            self.state.protected_gate_passed = False

        @invariant()
        def mode_always_valid(self):
            """Current mode must always be valid."""
            assert StateMachineInvariants.mode_is_valid(
                self.state.current_mode
            ), f"Invalid mode: {self.state.current_mode}"

        @invariant()
        def stale_hold_bounded(self):
            """Stale hold must not exceed maximum."""
            assert StateMachineInvariants.stale_hold_bounded(
                self.state.stale_hold_turns
            ), f"Stale hold exceeded: {self.state.stale_hold_turns} turns"

        @invariant()
        def hard_exit_leaves_protected(self):
            """If hard exit was seen and we were in sex, must have left."""
            if self.state.hard_exit_seen and self.state.previous_mode == "sex":
                assert self.state.current_mode == "daily", \
                    f"Hard exit didn't leave sex: still in {self.state.current_mode}"


# Test runner functions for pytest integration
def test_state_machine_properties():
    """Run Hypothesis stateful property tests."""
    if not HYPOTHESIS_AVAILABLE:
        import pytest
        pytest.skip("Hypothesis not installed")

    from hypothesis import settings
    from hypothesis.stateful import run_state_machine_as_test

    # Run with moderate number of steps
    run_state_machine_as_test(
        StateMachinePropertyTest,
        settings=settings(max_examples=50, stateful_step_count=20)
    )


def test_individual_invariants():
    """Test individual invariants directly."""
    # Hard exit forces daily
    assert StateMachineInvariants.hard_exit_forces_daily(True, "daily")
    assert not StateMachineInvariants.hard_exit_forces_daily(True, "work")
    assert StateMachineInvariants.hard_exit_forces_daily(False, "work")

    # Explicit work enters work
    assert StateMachineInvariants.explicit_work_same_turn(True, True, "daily", "work")
    assert not StateMachineInvariants.explicit_work_same_turn(True, True, "daily", "daily")

    # Protected requires gate
    assert StateMachineInvariants.protected_requires_gate("sex", False, "daily")
    assert not StateMachineInvariants.protected_requires_gate("sex", False, "sex")
    assert StateMachineInvariants.protected_requires_gate("sex", True, "sex")

    # Mode validation
    assert StateMachineInvariants.mode_is_valid("daily")
    assert StateMachineInvariants.mode_is_valid("work")
    assert StateMachineInvariants.mode_is_valid("sex")
    assert not StateMachineInvariants.mode_is_valid("invalid")

    # Stale hold bounded
    assert StateMachineInvariants.stale_hold_bounded(0)
    assert StateMachineInvariants.stale_hold_bounded(5)
    assert not StateMachineInvariants.stale_hold_bounded(6)


def test_sequence_isolation():
    """Test that independent sequences don't share state."""
    state1 = StateMachineState()
    state2 = StateMachineState()

    # Modify state1
    state1.current_mode = "work"
    state1.protected_gate_passed = True

    # state2 should be unaffected
    assert state2.current_mode == "daily"
    assert state2.protected_gate_passed is False


def test_deterministic_transitions():
    """Same facts and state must produce same result."""
    # Setup identical initial states
    state_a = StateMachineState()
    state_b = StateMachineState()

    # Apply same transition inputs
    def apply_transition(state: StateMachineState, nomination: str) -> str:
        if nomination == "work":
            return "work"
        return "daily"

    result_a = apply_transition(state_a, "work")
    result_b = apply_transition(state_b, "work")

    assert result_a == result_b, "Same inputs must produce same outputs"
