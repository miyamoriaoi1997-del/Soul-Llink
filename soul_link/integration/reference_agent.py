from __future__ import annotations

from collections.abc import Callable
import json
from typing import Any
from uuid import uuid4

from .contracts import CompletedTurn, RequestBudget, SessionEvent, TurnEnvelope
from .runtime import SoulLinkRuntime
from .tools import ToolCatalog

ModelCallable = Callable[[str, list[dict[str, Any]], list[dict[str, Any]]], str]


class ReferenceAgent:
    """Minimal host demonstrating the complete SoulLink turn contract."""

    def __init__(
        self,
        *,
        runtime: SoulLinkRuntime,
        model: ModelCallable,
        platform: str = "reference-agent",
        host_system_prompt: str = "",
        request_budget: RequestBudget | None = None,
        tools: ToolCatalog | None = None,
    ) -> None:
        self.runtime = runtime
        self.model = model
        self.platform = platform
        self.host_system_prompt = host_system_prompt
        self.request_budget = request_budget
        self.tools = tools or ToolCatalog()
        self.session_id = uuid4().hex
        self.turn_number = 0
        self.messages: list[dict[str, Any]] = []

    def run_turn(self, message: str) -> str:
        normalized = str(message).strip()
        if not normalized:
            raise ValueError("message is required")
        next_turn_number = self.turn_number + 1
        envelope = TurnEnvelope(
            session_id=self.session_id,
            turn_number=next_turn_number,
            platform=self.platform,
            raw_message=str(message),
            normalized_message=normalized,
            recent_context=tuple(self.messages),
            host_system_prompt=self.host_system_prompt,
            request_budget=self.request_budget,
        )
        prepared = self.runtime.prepare_turn(envelope)
        tool_schemas = self.tools.schemas()
        if self.request_budget is not None:
            serialized_tools = json.dumps(tool_schemas, ensure_ascii=False, separators=(",", ":"))
            tool_estimate = 0 if not tool_schemas else (len(serialized_tools) + 3) // 4
            if tool_estimate > self.request_budget.tool_schema_tokens:
                raise ValueError(
                    "tool schema budget exceeded: "
                    f"estimated {tool_estimate} tokens > {self.request_budget.tool_schema_tokens}"
                )
        request_messages = [*self.messages, {"role": "user", "content": normalized}]
        if self.request_budget is not None:
            serialized_messages = json.dumps(request_messages, ensure_ascii=False, separators=(",", ":"))
            message_estimate = (len(serialized_messages) + 3) // 4
            if message_estimate > self.request_budget.message_tokens:
                raise ValueError(
                    "message budget exceeded: "
                    f"estimated {message_estimate} tokens > {self.request_budget.message_tokens}"
                )
        reply = str(self.model(prepared.prompt_text, request_messages, tool_schemas))
        if self.request_budget is not None:
            response_estimate = (len(reply) + 3) // 4
            if response_estimate > self.request_budget.response_tokens:
                raise ValueError(
                    "response budget exceeded: "
                    f"estimated {response_estimate} tokens > {self.request_budget.response_tokens}"
                )
        self.runtime.complete_turn(
            CompletedTurn(
                session_id=self.session_id,
                turn_number=next_turn_number,
                platform=self.platform,
                user_content=normalized,
                assistant_content=reply,
            )
        )
        self.turn_number = next_turn_number
        self.messages.extend(
            (
                {"role": "user", "content": normalized},
                {"role": "assistant", "content": reply},
            )
        )
        return reply

    def reset_session(self) -> str:
        previous_session_id = self.session_id
        next_session_id = uuid4().hex
        self.runtime.switch_session(
            SessionEvent(
                session_id=next_session_id,
                previous_session_id=previous_session_id,
                reset=True,
            )
        )
        self.session_id = next_session_id
        self.turn_number = 0
        self.messages.clear()
        return self.session_id
