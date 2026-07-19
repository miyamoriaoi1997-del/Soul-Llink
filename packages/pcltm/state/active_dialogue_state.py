"""Active Dialogue State (ADS) for runtime-local conversation continuity.

ADS is intentionally short-lived runtime state.  It records "where this
conversation is right now" so weak continuations such as "continue" or
"do the above" resolve against the active thread before any durable memory
or archival recall can influence the response.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Mapping


def _clean_text(value: Any, *, max_chars: int = 280) -> str:
    text = "" if value is None else str(value)
    text = " ".join(text.strip().split())
    if len(text) <= max_chars:
        return text
    return text[: max_chars - 1].rstrip() + "…"


def _clean_list(values: Any, *, max_items: int = 4, max_chars: int = 180) -> tuple[str, ...]:
    if values is None:
        return ()
    if isinstance(values, str):
        values = [values]
    cleaned: list[str] = []
    for value in values:
        item = _clean_text(value, max_chars=max_chars)
        if item and item not in cleaned:
            cleaned.append(item)
        if len(cleaned) >= max_items:
            break
    return tuple(cleaned)


@dataclass(frozen=True)
class ActiveDialogueState:
    """Small runtime state injected ahead of durable memory.

    This object is not a semantic memory record and must not be persisted as
    durable PCLTM/MemFS knowledge.  Hosts may carry it in session-local state
    or reconstruct it from the visible conversation tail after compaction.
    """

    conversation_goal: str = ""
    current_task: str = ""
    last_user_intent: str = ""
    last_assistant_commitment: str = ""
    open_threads: tuple[str, ...] = field(default_factory=tuple)
    pending_questions: tuple[str, ...] = field(default_factory=tuple)
    local_constraints: tuple[str, ...] = field(default_factory=tuple)
    response_mode: str = "work"
    continuation_hint: str = ""

    def __post_init__(self) -> None:
        object.__setattr__(self, "conversation_goal", _clean_text(self.conversation_goal))
        object.__setattr__(self, "current_task", _clean_text(self.current_task))
        object.__setattr__(self, "last_user_intent", _clean_text(self.last_user_intent))
        object.__setattr__(self, "last_assistant_commitment", _clean_text(self.last_assistant_commitment))
        object.__setattr__(self, "open_threads", _clean_list(self.open_threads))
        object.__setattr__(self, "pending_questions", _clean_list(self.pending_questions))
        object.__setattr__(self, "local_constraints", _clean_list(self.local_constraints))
        object.__setattr__(self, "response_mode", _clean_text(self.response_mode, max_chars=40) or "work")
        object.__setattr__(self, "continuation_hint", _clean_text(self.continuation_hint))

    @property
    def is_empty(self) -> bool:
        return not any(
            (
                self.conversation_goal,
                self.current_task,
                self.last_user_intent,
                self.last_assistant_commitment,
                self.open_threads,
                self.pending_questions,
                self.local_constraints,
                self.continuation_hint,
            )
        )

    def with_updates(self, **updates: Any) -> "ActiveDialogueState":
        data = self.to_dict()
        data.update(updates)
        return ActiveDialogueState.from_dict(data)

    def to_dict(self) -> dict[str, Any]:
        return {
            "conversation_goal": self.conversation_goal,
            "current_task": self.current_task,
            "last_user_intent": self.last_user_intent,
            "last_assistant_commitment": self.last_assistant_commitment,
            "open_threads": list(self.open_threads),
            "pending_questions": list(self.pending_questions),
            "local_constraints": list(self.local_constraints),
            "response_mode": self.response_mode,
            "continuation_hint": self.continuation_hint,
        }

    @classmethod
    def from_dict(cls, value: Mapping[str, Any] | None) -> "ActiveDialogueState":
        if not value:
            return cls()
        return cls(
            conversation_goal=value.get("conversation_goal", ""),
            current_task=value.get("current_task", ""),
            last_user_intent=value.get("last_user_intent", ""),
            last_assistant_commitment=value.get("last_assistant_commitment", ""),
            open_threads=_clean_list(value.get("open_threads")),
            pending_questions=_clean_list(value.get("pending_questions")),
            local_constraints=_clean_list(value.get("local_constraints")),
            response_mode=value.get("response_mode", "work"),
            continuation_hint=value.get("continuation_hint", ""),
        )

    def render(self) -> str:
        """Render a compact prompt block that outranks durable memory."""
        return self.render_sealed()

    def render_sealed(self, references: Mapping[str, str] | None = None) -> str:
        """Render short-term state as sealed metadata, never as dialogue turns.

        Active Dialogue State may bind weak continuations such as "继续" to a
        prior task, but it must not re-inject prior user text as an executable
        instruction.  Any state value not explicitly replaced by a typed
        reference is reduced to an opaque presence marker.
        """
        references = references or {}

        def redact(value: str) -> str:
            rendered = value
            ordered_references = sorted(
                references.items(), key=lambda pair: len(pair[0]), reverse=True
            )
            for raw, marker in ordered_references:
                if raw:
                    rendered = rendered.replace(raw, marker)
            return rendered

        def sealed_value(field: str, value: str) -> str:
            rendered = redact(value)
            if rendered != value and rendered.startswith("<") and rendered.endswith(">"):
                return rendered
            return f"<sealed_active_dialogue_{field}>"

        def sealed_list(field: str, values: tuple[str, ...]) -> str:
            return " | ".join(sealed_value(field, item) for item in values)

        lines = [
            "【active_dialogue_state】",
            "  scope: sealed_runtime_short_term_only",
            "  priority: before_durable_memory",
            "  active_turn: false",
            "  pending_answer: false",
            "  executable: false",
            "  policy: state_summary_not_chat_transcript_no_task_resurrection",
        ]
        if self.conversation_goal:
            lines.append(f"  conversation_goal: {sealed_value('conversation_goal', self.conversation_goal)}")
        if self.current_task:
            lines.append(f"  current_task: {sealed_value('current_task', self.current_task)}")
        if self.last_user_intent:
            lines.append(f"  last_user_intent: {sealed_value('last_user_intent', self.last_user_intent)}")
        if self.last_assistant_commitment:
            lines.append(
                f"  last_assistant_commitment: {sealed_value('last_assistant_commitment', self.last_assistant_commitment)}"
            )
        if self.open_threads:
            lines.append("  open_threads: " + sealed_list('open_threads', self.open_threads))
        if self.pending_questions:
            lines.append("  pending_questions: " + sealed_list('pending_questions', self.pending_questions))
        if self.local_constraints:
            lines.append("  local_constraints: " + sealed_list('local_constraints', self.local_constraints))
        if self.continuation_hint:
            lines.append(f"  continuation_hint: {sealed_value('continuation_hint', self.continuation_hint)}")
        return "\n".join(lines)
