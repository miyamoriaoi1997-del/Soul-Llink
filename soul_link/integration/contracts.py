from __future__ import annotations

from dataclasses import dataclass, field
from types import MappingProxyType
from typing import Any, Mapping


@dataclass(frozen=True, slots=True)
class RequestBudget:
    total_tokens: int
    system_prompt_tokens: int
    tool_schema_tokens: int
    memory_prompt_tokens: int
    message_tokens: int
    response_tokens: int
    safety_margin_tokens: int

    def __post_init__(self) -> None:
        values = (
            self.total_tokens,
            self.system_prompt_tokens,
            self.tool_schema_tokens,
            self.memory_prompt_tokens,
            self.message_tokens,
            self.response_tokens,
            self.safety_margin_tokens,
        )
        if any(type(value) is not int or value < 0 for value in values):
            raise ValueError("request budget values must be non-negative integers")
        buckets = sum(values[1:])
        if buckets != self.total_tokens:
            raise ValueError("request budget bucket sum must equal total_tokens")


@dataclass(frozen=True, slots=True)
class HostCapabilities:
    turn_lifecycle: bool = True
    session_lifecycle: bool = True
    prompt_injection: bool = True
    context_compaction: bool = True
    tool_registration: bool = True
    request_budget: bool = True
    usage_feedback: bool = True
    exact_capture: bool = True

    def __post_init__(self) -> None:
        for name in self.__dataclass_fields__:
            if type(getattr(self, name)) is not bool:
                raise TypeError(f"host capability {name} must be bool")

    @classmethod
    def full(cls) -> "HostCapabilities":
        return cls()

    def missing(self) -> tuple[str, ...]:
        return tuple(name for name in self.__dataclass_fields__ if not getattr(self, name))


@dataclass(frozen=True, slots=True)
class TurnEnvelope:
    session_id: str
    turn_number: int
    platform: str
    raw_message: str
    normalized_message: str
    recent_context: tuple[Mapping[str, Any], ...] = ()
    host_system_prompt: str = ""
    previous_mode: str | None = None
    emotion_state: Mapping[str, Any] = field(default_factory=dict)
    emotion_modifier: str = ""
    request_budget: RequestBudget | None = None

    def __post_init__(self) -> None:
        if not self.session_id.strip():
            raise ValueError("session_id is required")
        if type(self.turn_number) is not int or self.turn_number < 1:
            raise ValueError("turn_number must be a positive integer")
        if not self.platform.strip():
            raise ValueError("platform is required")
        if not self.normalized_message.strip():
            raise ValueError("normalized_message is required")
        context = tuple(MappingProxyType(dict(item)) for item in self.recent_context)
        object.__setattr__(self, "recent_context", context)
        object.__setattr__(self, "emotion_state", MappingProxyType(dict(self.emotion_state)))


@dataclass(frozen=True, slots=True)
class SessionEvent:
    session_id: str
    previous_session_id: str = ""
    reset: bool = False
    rewound: bool = False

    def __post_init__(self) -> None:
        if not self.session_id.strip():
            raise ValueError("session_id is required")
        if type(self.reset) is not bool or type(self.rewound) is not bool:
            raise TypeError("session event flags must be bool")


@dataclass(frozen=True, slots=True)
class CompletedTurn:
    session_id: str
    turn_number: int
    platform: str
    user_content: str
    assistant_content: str


@dataclass(frozen=True, slots=True)
class PreparedTurn:
    session_id: str
    turn_number: int
    mode: str
    route_bucket: str
    selected_layers: tuple[str, ...]
    prompt_text: str
    prompt_hash: str
    request_budget: RequestBudget | None
    capability_status: str
    missing_capabilities: tuple[str, ...]
    shadow_packet: Mapping[str, Any]
    audit_packet: Mapping[str, Any]
