from __future__ import annotations

import sys
from pathlib import Path

import pytest

from soul_link.integration import (
    HostCapabilities,
    RequestBudget,
    SoulLinkRuntime,
    TurnEnvelope,
)


class _FakeLink:
    def compose_active_prompt(self, **kwargs):
        assert kwargs["host_system_prompt"] == "host prompt"
        assert kwargs["user_message"] == "continue"
        assert kwargs["recent_context"] == [{"role": "assistant", "content": "prior"}]
        assert kwargs["platform"] == "reference-agent"
        return type(
            "Resolution",
            (),
            {
                "mode": "work",
                "route_bucket": "task",
                "selected_layers": ["core", "work"],
                "prompt_candidate": {"prompt_text": "composed", "prompt_hash": "abc123"},
                "shadow_packet": {"shadow": True},
                "audit_packet": {"audit": True},
            },
        )()


def test_prepare_turn_is_host_neutral_and_preserves_one_request_budget() -> None:
    forbidden_before = {name for name in sys.modules if name.split(".", 1)[0] in {"agent", "gateway", "hermes_cli"}}
    budget = RequestBudget(
        total_tokens=1000,
        system_prompt_tokens=100,
        tool_schema_tokens=50,
        memory_prompt_tokens=0,
        message_tokens=600,
        response_tokens=200,
        safety_margin_tokens=50,
    )
    envelope = TurnEnvelope(
        session_id="session-1",
        turn_number=2,
        platform="reference-agent",
        raw_message="continue",
        normalized_message="continue",
        recent_context=({"role": "assistant", "content": "prior"},),
        host_system_prompt="host prompt",
        request_budget=budget,
    )
    runtime = SoulLinkRuntime(
        link=_FakeLink(),
        capabilities=HostCapabilities.full(),
        turn_sink=lambda _: None,
        session_sink=lambda _: None,
    )

    prepared = runtime.prepare_turn(envelope)

    assert prepared.prompt_text == "composed"
    assert prepared.prompt_hash == "abc123"
    assert prepared.mode == "work"
    assert prepared.selected_layers == ("core", "work")
    assert prepared.request_budget is budget
    assert prepared.capability_status == "full"
    forbidden_after = {name for name in sys.modules if name.split(".", 1)[0] in {"agent", "gateway", "hermes_cli"}}
    assert forbidden_after == forbidden_before


def test_request_budget_rejects_double_counted_or_incomplete_buckets() -> None:
    with pytest.raises(ValueError, match="sum"):
        RequestBudget(
            total_tokens=1000,
            system_prompt_tokens=100,
            tool_schema_tokens=50,
            memory_prompt_tokens=0,
            message_tokens=700,
            response_tokens=200,
            safety_margin_tokens=50,
        )


def test_prepare_turn_rejects_prompt_that_exceeds_declared_prompt_buckets() -> None:
    budget = RequestBudget(
        total_tokens=20,
        system_prompt_tokens=0,
        tool_schema_tokens=1,
        memory_prompt_tokens=1,
        message_tokens=6,
        response_tokens=10,
        safety_margin_tokens=2,
    )
    runtime = SoulLinkRuntime(
        link=_FakeLink(),
        capabilities=HostCapabilities.full(),
        turn_sink=lambda _: None,
        session_sink=lambda _: None,
    )
    envelope = TurnEnvelope(
        session_id="session-1",
        turn_number=1,
        platform="reference-agent",
        raw_message="continue",
        normalized_message="continue",
        recent_context=({"role": "assistant", "content": "prior"},),
        host_system_prompt="host prompt",
        request_budget=budget,
    )

    with pytest.raises(ValueError, match="prompt budget"):
        runtime.prepare_turn(envelope)


def test_full_lifecycle_capabilities_require_explicit_sinks() -> None:
    with pytest.raises(ValueError, match="turn lifecycle sink"):
        SoulLinkRuntime(link=_FakeLink(), capabilities=HostCapabilities.full())


def test_missing_prompt_injection_capability_is_explicitly_degraded() -> None:
    runtime = SoulLinkRuntime(
        link=_FakeLink(),
        capabilities=HostCapabilities(prompt_injection=False),
        turn_sink=lambda _: None,
        session_sink=lambda _: None,
    )
    envelope = TurnEnvelope(
        session_id="session-1",
        turn_number=1,
        platform="reference-agent",
        raw_message="continue",
        normalized_message="continue",
        recent_context=({"role": "assistant", "content": "prior"},),
        host_system_prompt="host prompt",
    )

    prepared = runtime.prepare_turn(envelope)

    assert prepared.prompt_text == ""
    assert prepared.capability_status == "degraded"
    assert prepared.missing_capabilities == ("prompt_injection",)
