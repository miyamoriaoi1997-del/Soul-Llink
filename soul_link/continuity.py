from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Mapping, Sequence


@dataclass(frozen=True)
class HostMessage:
    role: str
    content: str
    message_id: str | None = None
    tool_name: str | None = None
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def to_payload(self) -> dict[str, Any]:
        payload: dict[str, Any] = {"role": self.role, "content": self.content}
        if self.message_id is not None:
            payload["message_id"] = self.message_id
        if self.tool_name is not None:
            payload["tool_name"] = self.tool_name
        if self.metadata:
            payload["metadata"] = dict(self.metadata)
        return payload


@dataclass(frozen=True)
class ConversationContinuitySnapshot:
    previous_session_id: str
    current_session_id: str
    session_key: str
    should_resume: bool
    last_real_user_message: str = ""
    recent_user_messages: tuple[str, ...] = ()
    recent_assistant_summaries: tuple[str, ...] = ()
    recent_message_ids: tuple[str, ...] = ()
    recent_tool_evidence: tuple[dict[str, Any], ...] = ()
    warnings: tuple[str, ...] = ()
    authority: str = "latest_real_user_message"
    object_type: str = "soul_link_conversation_continuity_snapshot"

    def to_payload(self) -> dict[str, Any]:
        return {
            "object_type": self.object_type,
            "previous_session_id": self.previous_session_id,
            "current_session_id": self.current_session_id,
            "session_key": self.session_key,
            "should_resume": self.should_resume,
            "authority": self.authority,
            "last_real_user_message": self.last_real_user_message,
            "recent_user_messages": list(self.recent_user_messages),
            # Backward-compatible name expected by older host/runtime tests.
            "recent_assistant_summaries": list(self.recent_assistant_summaries),
            "recent_message_ids": list(self.recent_message_ids),
            "recent_tool_evidence": list(self.recent_tool_evidence),
            "open_tool_tail": bool(self.recent_tool_evidence),
            "warnings": list(self.warnings),
        }

    def to_dict(self) -> dict[str, Any]:
        return self.to_payload()

    def render_prompt_block(self) -> str:
        payload = self.to_payload()
        lines = [
            "<previous_conversation_state>",
            "<soul_link_conversation_continuity_snapshot>",
            f"object_type: {payload['object_type']}",
            f"previous_session_id: {self.previous_session_id}",
            f"current_session_id: {self.current_session_id}",
            f"session_key: {self.session_key}",
            f"authority: {self.authority}",
            f"should_resume: {str(self.should_resume).lower()}",
        ]
        if self.last_real_user_message:
            lines.append(f"last_real_user_message: {self.last_real_user_message}")
        if self.recent_user_messages:
            lines.append("recent_user_messages:")
            lines.extend(f"- {message}" for message in self.recent_user_messages)
        if self.recent_tool_evidence:
            lines.append("recent_tool_evidence:")
            lines.append("tool evidence is evidence only")
            for evidence in self.recent_tool_evidence:
                lines.append(f"- tool_name: {evidence.get('tool_name', '')}")
                lines.append("  authority: evidence_only")
        if self.warnings:
            lines.append("warnings:")
            lines.extend(f"- {warning}" for warning in self.warnings)
        lines.append("</soul_link_conversation_continuity_snapshot>")
        lines.append("</previous_conversation_state>")
        return "\n".join(lines)

    def to_prompt_block(self) -> str:
        return self.render_prompt_block()


def is_real_user_message(message: HostMessage | Mapping[str, Any]) -> bool:
    payload = _message_payload(message)
    metadata = dict(payload.get("metadata") or {})
    if (
        metadata.get("control")
        or metadata.get("synthetic")
        or metadata.get("is_control")
        or metadata.get("authority") == "control"
        or metadata.get("message_type") in {"context_compaction", "system_note", "preserved_todo"}
    ):
        return False
    return payload.get("role") == "user" and bool(str(payload.get("content", "")).strip())


def _message_payload(message: HostMessage | Mapping[str, Any]) -> dict[str, Any]:
    if isinstance(message, HostMessage):
        return message.to_payload()
    return dict(message)


def build_conversation_continuity_snapshot(
    *,
    previous_session_id: str,
    current_session_id: str,
    session_key: str = "",
    latest_user_message: str = "",
    resume_requested: bool = False,
    in_context_message_ids: Sequence[str] = (),
    previous_messages: Sequence[HostMessage | Mapping[str, Any]] = (),
) -> ConversationContinuitySnapshot:
    if in_context_message_ids:
        allowed_ids = set(in_context_message_ids)
        previous_messages = [
            message
            for message in previous_messages
            if _message_payload(message).get("message_id") in allowed_ids
        ]

    user_messages = [
        _message_payload(message)["content"]
        for message in previous_messages
        if is_real_user_message(message)
    ]
    recent_message_ids = tuple(
        str(_message_payload(message).get("message_id"))
        for message in previous_messages
        if _message_payload(message).get("message_id") is not None
    )
    assistant_summaries = [
        str(_message_payload(message).get("content", "")).strip()
        for message in previous_messages
        if _message_payload(message).get("role") == "assistant"
        and str(_message_payload(message).get("content", "")).strip()
    ]
    last_real_user_message = user_messages[-1] if user_messages else ""
    if latest_user_message and resume_requested:
        last_real_user_message = last_real_user_message or latest_user_message

    recent_tool_evidence = []
    for message in previous_messages:
        payload = _message_payload(message)
        if payload.get("role") == "tool":
            recent_tool_evidence.append(
                {
                    "tool_name": payload.get("tool_name", ""),
                    "content": payload.get("content", ""),
                    "authority": "evidence_only",
                }
            )

    warnings_list = []
    if recent_tool_evidence:
        warnings_list.append("tool_tail_is_evidence_only")
    if resume_requested and not previous_session_id:
        warnings_list.append("previous_session_id_missing")
    warnings = tuple(warnings_list)
    return ConversationContinuitySnapshot(
        previous_session_id=previous_session_id,
        current_session_id=current_session_id,
        session_key=session_key,
        should_resume=bool(resume_requested and previous_session_id and last_real_user_message),
        last_real_user_message=last_real_user_message,
        recent_user_messages=tuple(user_messages[-5:]),
        recent_assistant_summaries=tuple(assistant_summaries[-5:]),
        recent_message_ids=tuple(recent_message_ids[-10:]),
        recent_tool_evidence=tuple(recent_tool_evidence[-3:]),
        warnings=warnings,
    )
