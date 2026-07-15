from __future__ import annotations

from collections.abc import Callable
from types import MappingProxyType
from typing import Any

from .contracts import HostCapabilities, PreparedTurn, TurnEnvelope


class SoulLinkRuntime:
    """Host-neutral turn preparation facade around the existing SoulLink API."""

    def __init__(
        self,
        *,
        link: Any,
        capabilities: HostCapabilities,
        turn_sink: Callable[[Any], None] | None = None,
        session_sink: Callable[[Any], None] | None = None,
    ) -> None:
        if capabilities.turn_lifecycle and turn_sink is None:
            raise ValueError("turn lifecycle sink is required when turn_lifecycle is enabled")
        if capabilities.session_lifecycle and session_sink is None:
            raise ValueError("session lifecycle sink is required when session_lifecycle is enabled")
        self._link = link
        self._capabilities = capabilities
        self._turn_sink = turn_sink
        self._session_sink = session_sink

    def complete_turn(self, completed: Any) -> None:
        if self._turn_sink is None:
            raise RuntimeError("turn lifecycle is unavailable")
        self._turn_sink(completed)

    def switch_session(self, event: Any) -> None:
        if self._session_sink is None:
            raise RuntimeError("session lifecycle is unavailable")
        self._session_sink(event)

    def prepare_turn(self, envelope: TurnEnvelope) -> PreparedTurn:
        missing = self._capabilities.missing()
        if not self._capabilities.prompt_injection:
            return PreparedTurn(
                session_id=envelope.session_id,
                turn_number=envelope.turn_number,
                mode=envelope.previous_mode or "daily",
                route_bucket="",
                selected_layers=(),
                prompt_text="",
                prompt_hash="",
                request_budget=envelope.request_budget,
                capability_status="degraded",
                missing_capabilities=missing,
                shadow_packet=MappingProxyType({}),
                audit_packet=MappingProxyType({}),
            )

        resolution = self._link.compose_active_prompt(
            host_system_prompt=envelope.host_system_prompt,
            user_message=envelope.normalized_message,
            recent_context=[dict(item) for item in envelope.recent_context],
            previous_mode=envelope.previous_mode,
            emotion_state=dict(envelope.emotion_state),
            emotion_modifier=envelope.emotion_modifier,
            platform=envelope.platform,
        )
        candidate = dict(resolution.prompt_candidate or {})
        prompt_text = str(candidate.get("prompt_text") or "")
        budget = envelope.request_budget
        if budget is not None:
            prompt_budget = budget.system_prompt_tokens + budget.memory_prompt_tokens
            prompt_estimate = (len(prompt_text) + 3) // 4
            if prompt_estimate > prompt_budget:
                raise ValueError(
                    f"prompt budget exceeded: estimated {prompt_estimate} tokens > {prompt_budget}"
                )
        return PreparedTurn(
            session_id=envelope.session_id,
            turn_number=envelope.turn_number,
            mode=str(resolution.mode),
            route_bucket=str(resolution.route_bucket),
            selected_layers=tuple(resolution.selected_layers),
            prompt_text=prompt_text,
            prompt_hash=str(candidate.get("prompt_hash") or ""),
            request_budget=envelope.request_budget,
            capability_status="full" if not missing else "degraded",
            missing_capabilities=missing,
            shadow_packet=MappingProxyType(dict(resolution.shadow_packet)),
            audit_packet=MappingProxyType(dict(resolution.audit_packet)),
        )
