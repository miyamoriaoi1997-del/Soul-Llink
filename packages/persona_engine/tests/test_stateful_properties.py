"""Stateful properties exercised through the production persona orchestrator."""

from __future__ import annotations

import tempfile
from pathlib import Path

from hypothesis.stateful import RuleBasedStateMachine, invariant, precondition, rule

from persona_engine.persona_orchestrator.state_orchestrator import StateOrchestrator


class PersonaRuntimeStateMachine(RuleBasedStateMachine):
    """Generate turn sequences against the real production routing pipeline."""

    VALID_MODES = {"daily", "work", "sex"}

    def __init__(self) -> None:
        super().__init__()
        self._temp_dir = tempfile.TemporaryDirectory()
        self.orchestrator = StateOrchestrator(
            ".",
            log_path=Path(self._temp_dir.name) / "stateful-orchestrator.jsonl",
        )
        self.current_mode = "daily"

    def teardown(self) -> None:
        self._temp_dir.cleanup()

    def _turn(self, message: str, score: float | None = None):
        emotion_state = {} if score is None else {"emotion_score": score}
        packet = self.orchestrator.analyze_turn(
            user_message=message,
            emotion_state=emotion_state,
            previous_mode=self.current_mode,
        )
        self.current_mode = packet.mode
        return packet

    @rule()
    def explicit_work_enters_work(self) -> None:
        packet = self._turn("帮我检查 gateway 日志", score=1.0)
        assert packet.mode == "work"
        assert "work" in packet.selected_layers

    @rule()
    def approved_protected_request_enters_sex(self) -> None:
        packet = self._turn("我们做爱", score=70.0)
        assert packet.mode == "sex"
        assert "sex" in packet.selected_layers

    @precondition(lambda self: self.current_mode != "sex")
    @rule()
    def restrained_protected_request_cannot_enter_sex(self) -> None:
        packet = self._turn("我们做爱", score=0.5)
        assert packet.mode != "sex"
        assert "sex" not in packet.selected_layers
        assert "sex_desire_gate_restrained" in packet.safety_flags

    @rule()
    def explicit_close_leaves_protected_mode(self) -> None:
        previous = self.current_mode
        packet = self._turn("先到这里，抱抱我", score=70.0)
        if previous == "sex":
            assert packet.mode == "daily"
            assert "sex" not in packet.selected_layers

    @rule()
    def short_continuation_preserves_a_valid_mode(self) -> None:
        packet = self._turn("继续")
        assert packet.mode in self.VALID_MODES

    @invariant()
    def mode_and_layers_remain_canonical(self) -> None:
        assert self.current_mode in self.VALID_MODES


def test_state_machine_properties() -> None:
    """Run Hypothesis defaults against the real orchestrator state machine."""
    from hypothesis.stateful import run_state_machine_as_test

    run_state_machine_as_test(PersonaRuntimeStateMachine)


def test_independent_runtime_instances_do_not_share_sequence_state(tmp_path) -> None:
    first = StateOrchestrator(".", log_path=tmp_path / "first.jsonl")
    second = StateOrchestrator(".", log_path=tmp_path / "second.jsonl")

    first_packet = first.analyze_turn(
        user_message="帮我检查 gateway 日志",
        emotion_state={"emotion_score": 1.0},
        previous_mode="daily",
    )
    second_packet = second.analyze_turn(
        user_message="继续",
        previous_mode="daily",
    )

    assert first_packet.mode == "work"
    assert second_packet.mode == "daily"


def test_same_turn_inputs_produce_same_runtime_decision(tmp_path) -> None:
    packets = [
        StateOrchestrator(".", log_path=tmp_path / f"run-{index}.jsonl").analyze_turn(
            user_message="帮我检查 gateway 日志",
            emotion_state={"emotion_score": 1.0},
            previous_mode="daily",
        )
        for index in range(2)
    ]

    assert packets[0].mode == packets[1].mode == "work"
    assert packets[0].transition == packets[1].transition
    assert packets[0].selected_layers == packets[1].selected_layers
