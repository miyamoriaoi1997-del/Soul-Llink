from __future__ import annotations

import pytest

from soul_link.integration import (
    HostCapabilities,
    ReferenceAgent,
    RequestBudget,
    SoulLinkRuntime,
    ToolCatalog,
    ToolSpec,
)


class _RecordingRuntime(SoulLinkRuntime):
    def __init__(self):
        self.events = []
        self.session_events = []
        super().__init__(
            link=_Link(),
            capabilities=HostCapabilities.full(),
            turn_sink=self.events.append,
            session_sink=self.session_events.append,
        )


class _Link:
    def compose_active_prompt(self, **kwargs):
        return type(
            "Resolution",
            (),
            {
                "mode": "work",
                "route_bucket": "task",
                "selected_layers": ["core", "work"],
                "prompt_candidate": {"prompt_text": "active prompt", "prompt_hash": "hash-1"},
                "shadow_packet": {},
                "audit_packet": {},
            },
        )()


def test_reference_agent_runs_prepare_model_complete_lifecycle() -> None:
    runtime = _RecordingRuntime()
    calls = []

    def model(prompt, messages, tools):
        calls.append((prompt, messages, tools))
        return "assistant reply"

    agent = ReferenceAgent(
        runtime=runtime,
        model=model,
        platform="reference-agent",
        host_system_prompt="host prompt",
        request_budget=RequestBudget(
            total_tokens=1000,
            system_prompt_tokens=100,
            tool_schema_tokens=0,
            memory_prompt_tokens=0,
            message_tokens=650,
            response_tokens=200,
            safety_margin_tokens=50,
        ),
    )

    reply = agent.run_turn("hello")

    assert reply == "assistant reply"
    assert calls == [("active prompt", [{"role": "user", "content": "hello"}], [])]
    assert len(runtime.events) == 1
    completed = runtime.events[0]
    assert completed.session_id == agent.session_id
    assert completed.turn_number == 1
    assert completed.user_content == "hello"
    assert completed.assistant_content == "assistant reply"
    assert agent.messages == [
        {"role": "user", "content": "hello"},
        {"role": "assistant", "content": "assistant reply"},
    ]


def test_reference_agent_rejects_response_over_budget_without_completing_turn() -> None:
    runtime = _RecordingRuntime()
    budget = RequestBudget(
        total_tokens=30,
        system_prompt_tokens=5,
        tool_schema_tokens=0,
        memory_prompt_tokens=0,
        message_tokens=10,
        response_tokens=1,
        safety_margin_tokens=14,
    )
    agent = ReferenceAgent(runtime=runtime, model=lambda *_: "response much longer than four characters", request_budget=budget)

    with pytest.raises(ValueError, match="response budget"):
        agent.run_turn("hi")

    assert agent.turn_number == 0
    assert agent.messages == []
    assert runtime.events == []


def test_reference_agent_rejects_messages_over_budget() -> None:
    runtime = _RecordingRuntime()
    budget = RequestBudget(
        total_tokens=20,
        system_prompt_tokens=5,
        tool_schema_tokens=0,
        memory_prompt_tokens=0,
        message_tokens=1,
        response_tokens=12,
        safety_margin_tokens=2,
    )
    agent = ReferenceAgent(runtime=runtime, model=lambda *_: "reply", request_budget=budget)

    with pytest.raises(ValueError, match="message budget"):
        agent.run_turn("this message is much longer than four characters")

    assert agent.turn_number == 0
    assert runtime.events == []


def test_reference_agent_rejects_tool_schemas_over_budget() -> None:
    runtime = _RecordingRuntime()
    budget = RequestBudget(
        total_tokens=20,
        system_prompt_tokens=5,
        tool_schema_tokens=1,
        memory_prompt_tokens=0,
        message_tokens=2,
        response_tokens=10,
        safety_margin_tokens=2,
    )
    tools = ToolCatalog(
        [ToolSpec(name="large", description="x" * 100, parameters={"type": "object"}, handler=lambda _: {})]
    )
    agent = ReferenceAgent(runtime=runtime, model=lambda *_: "reply", request_budget=budget, tools=tools)

    with pytest.raises(ValueError, match="tool schema budget"):
        agent.run_turn("hello")

    assert agent.turn_number == 0
    assert runtime.events == []


def test_reference_agent_failed_model_call_does_not_consume_turn() -> None:
    runtime = _RecordingRuntime()

    def fail_model(*_):
        raise RuntimeError("model unavailable")

    agent = ReferenceAgent(runtime=runtime, model=fail_model)

    with pytest.raises(RuntimeError, match="unavailable"):
        agent.run_turn("hello")

    assert agent.turn_number == 0
    assert agent.messages == []
    assert runtime.events == []


def test_reference_agent_failed_session_switch_preserves_local_scope() -> None:
    runtime = _RecordingRuntime()
    agent = ReferenceAgent(runtime=runtime, model=lambda *_: "reply")
    agent.run_turn("hello")
    old_session = agent.session_id
    old_messages = list(agent.messages)
    runtime._session_sink = lambda _: (_ for _ in ()).throw(RuntimeError("switch rejected"))

    with pytest.raises(RuntimeError, match="rejected"):
        agent.reset_session()

    assert agent.session_id == old_session
    assert agent.turn_number == 1
    assert agent.messages == old_messages


def test_reference_agent_session_reset_rotates_scope_and_clears_messages() -> None:
    runtime = _RecordingRuntime()
    agent = ReferenceAgent(runtime=runtime, model=lambda *_: "reply")
    old_session = agent.session_id
    agent.run_turn("hello")

    new_session = agent.reset_session()

    assert new_session != old_session
    assert agent.messages == []
    assert agent.turn_number == 0
    assert len(runtime.session_events) == 1
    event = runtime.session_events[0]
    assert event.previous_session_id == old_session
    assert event.session_id == new_session
    assert event.reset is True
    assert event.rewound is False
