from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping, Protocol, Sequence, cast, runtime_checkable

from .continuity import ConversationContinuitySnapshot, HostMessage, build_conversation_continuity_snapshot


@dataclass(frozen=True)
class HostConversationState:
    conversation_id: str
    agent_id: str = ""
    previous_conversation_id: str = ""
    summary: str = ""
    in_context_message_ids: tuple[str, ...] = ()
    session_key: str = ""
    resume_requested: bool = False

    def to_dict(self) -> dict[str, object]:
        return {
            "conversation_id": self.conversation_id,
            "agent_id": self.agent_id,
            "previous_conversation_id": self.previous_conversation_id,
            "summary": self.summary,
            "in_context_message_ids": list(self.in_context_message_ids),
            "session_key": self.session_key,
            "resume_requested": self.resume_requested,
        }


@runtime_checkable
class HostSessionProvider(Protocol):
    def get_previous_conversation(self, current: HostConversationState) -> HostConversationState | None:
        ...

    def get_messages(self, conversation_id: str) -> Sequence[HostMessage]:
        ...


@dataclass(frozen=True)
class FakeHostSessionProvider:
    # ``current`` is accepted for older tests/callers that model a complete
    # host session fixture. Runtime code passes the current state explicitly to
    # build_continuity_snapshot_from_provider(), so this field is informational.
    current: HostConversationState | None = None
    previous: HostConversationState | None = None
    messages_by_conversation: Mapping[str, Sequence[HostMessage]] | None = None

    def get_previous_conversation(self, current: HostConversationState) -> HostConversationState | None:
        if self.previous is None:
            return None
        if current.previous_conversation_id and self.previous.conversation_id != current.previous_conversation_id:
            return None
        return self.previous

    def current_conversation(self) -> HostConversationState | None:
        return self.current

    def get_messages(self, conversation_id: str) -> Sequence[HostMessage]:
        if self.messages_by_conversation is None:
            return ()
        return tuple(self.messages_by_conversation.get(conversation_id, ()))


def build_continuity_snapshot_from_provider(
    provider: HostSessionProvider,
    *,
    current: HostConversationState | None = None,
    latest_user_message: str = "",
) -> ConversationContinuitySnapshot:
    if current is None:
        current_getter = getattr(provider, "current_conversation", None)
        current_value = current_getter() if callable(current_getter) else current_getter
        current = cast(HostConversationState | None, current_value or getattr(provider, "current", None))
    if current is None:
        raise ValueError("current conversation state is required")
    previous = provider.get_previous_conversation(current)
    previous_session_id = previous.conversation_id if previous else ""
    previous_messages = provider.get_messages(previous_session_id) if previous else ()
    in_context_message_ids = previous.in_context_message_ids if previous else ()
    return build_conversation_continuity_snapshot(
        previous_session_id=previous_session_id,
        current_session_id=current.conversation_id,
        session_key=current.session_key or (previous.session_key if previous else ""),
        latest_user_message=latest_user_message,
        resume_requested=current.resume_requested,
        in_context_message_ids=in_context_message_ids,
        previous_messages=previous_messages,
    )
