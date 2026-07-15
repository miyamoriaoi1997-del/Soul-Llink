"""Update and inject Active Dialogue State (ADS).

The updater is deliberately heuristic and deterministic.  It is not a long-term
memory writer and does not perform semantic recall; it only keeps the active
runtime thread coherent across adjacent turns and compacted tails.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

from .active_dialogue_state import ActiveDialogueState


_CONTINUATION_MARKERS = (
    "继续",
    "接着",
    "刚才",
    "刚刚",
    "那个",
    "这个",
    "上面",
    "上述",
    "按上面的做",
    "照这个做",
    "do it",
    "continue",
    "go on",
    "that one",
    "the above",
)

_INTERRUPTION_MARKERS = (
    "顺便",
    "插一句",
    "先问个",
    "先打断",
    "另一个问题",
    "off-topic",
    "by the way",
)

_CONSTRAINT_MARKERS = (
    "不要",
    "禁止",
    "必须",
    "先",
    "只",
    "不能",
    "直接",
    "完成后",
    "do not",
    "must",
    "only",
)

_QUESTION_MARKERS = ("?", "？", "确认", "要不要", "是否", "吗", "么")

_COMMITMENT_MARKERS = (
    "我会",
    "我先",
    "接下来",
    "我将",
    "我来",
    "会先",
    "完成后",
    "will",
    "I'll",
    "I will",
)


def _norm(text: str) -> str:
    return " ".join(str(text or "").strip().split())


def is_continuation_request(text: str) -> bool:
    lowered = _norm(text).lower()
    return bool(lowered) and any(marker.lower() in lowered for marker in _CONTINUATION_MARKERS)


def is_side_interruption(text: str) -> bool:
    lowered = _norm(text).lower()
    return any(marker.lower() in lowered for marker in _INTERRUPTION_MARKERS)


def _first_sentence(text: str, *, max_chars: int = 220) -> str:
    text = _norm(text)
    if len(text) <= max_chars:
        return text
    return text[: max_chars - 1].rstrip() + "…"


def _append_unique(values: Iterable[str], value: str, *, max_items: int = 4) -> tuple[str, ...]:
    out: list[str] = []
    for item in values:
        item = _first_sentence(item, max_chars=180)
        if item and item not in out:
            out.append(item)
    value = _first_sentence(value, max_chars=180)
    if value and value not in out:
        out.append(value)
    return tuple(out[-max_items:])


def _extract_constraints(text: str, previous: tuple[str, ...]) -> tuple[str, ...]:
    lowered = text.lower()
    if not any(marker.lower() in lowered for marker in _CONSTRAINT_MARKERS):
        return previous
    return _append_unique(previous, text, max_items=4)


def _extract_pending_questions(user_text: str, assistant_text: str, previous: tuple[str, ...]) -> tuple[str, ...]:
    pending = list(previous)
    if any(marker in user_text for marker in _QUESTION_MARKERS):
        pending = list(_append_unique(pending, user_text, max_items=4))
    # If the assistant asked a confirmation question, keep it as pending for the
    # next turn; if it did not, a direct answer normally closes older questions.
    if any(marker in assistant_text for marker in _QUESTION_MARKERS):
        pending = list(_append_unique(pending, assistant_text, max_items=4))
    elif user_text and pending:
        pending = pending[-2:]
    return tuple(pending[-4:])


def _extract_commitment(assistant_text: str) -> str:
    text = _first_sentence(assistant_text)
    lowered = text.lower()
    if any(marker.lower() in lowered for marker in _COMMITMENT_MARKERS):
        return text
    return text[:220]


@dataclass(frozen=True)
class DialogueTurn:
    """One visible user/assistant exchange for ADS updates."""

    user: str = ""
    assistant: str = ""


def update_active_dialogue(
    previous: ActiveDialogueState | None,
    turn: DialogueTurn,
    *,
    response_mode: str | None = None,
) -> ActiveDialogueState:
    """Return the next runtime-local ADS after one conversation turn.

    Continuation turns update ``last_user_intent`` while preserving the current
    task.  Side interruptions are tracked as open threads without stealing the
    main task.  Substantive new user turns replace the current task.
    """

    state = previous or ActiveDialogueState(response_mode=response_mode or "work")
    user_text = _first_sentence(turn.user)
    assistant_text = _first_sentence(turn.assistant)
    mode = response_mode or state.response_mode

    if not user_text and not assistant_text:
        return state.with_updates(response_mode=mode)

    continuation = is_continuation_request(user_text)
    interruption = is_side_interruption(user_text)

    conversation_goal = state.conversation_goal
    current_task = state.current_task
    open_threads = state.open_threads

    if user_text:
        if continuation and current_task:
            # Keep the active task anchored; the weak phrase points back to it.
            pass
        elif interruption and current_task:
            open_threads = _append_unique(open_threads, user_text)
        else:
            current_task = user_text
            conversation_goal = conversation_goal or user_text
            open_threads = _append_unique(open_threads, user_text)

    local_constraints = _extract_constraints(user_text, state.local_constraints)
    pending_questions = _extract_pending_questions(user_text, assistant_text, state.pending_questions)
    commitment = _extract_commitment(assistant_text) or state.last_assistant_commitment

    if current_task:
        hint = f"继续围绕当前任务推进：{current_task}"
        if continuation:
            hint = f"将“{user_text}”解析为继续当前任务：{current_task}"
    else:
        hint = "等待用户给出当前任务。"

    return ActiveDialogueState(
        conversation_goal=conversation_goal,
        current_task=current_task,
        last_user_intent=user_text or state.last_user_intent,
        last_assistant_commitment=commitment,
        open_threads=open_threads,
        pending_questions=pending_questions,
        local_constraints=local_constraints,
        response_mode=mode,
        continuation_hint=hint,
    )


def update_from_turns(
    turns: Iterable[DialogueTurn],
    *,
    initial: ActiveDialogueState | None = None,
    response_mode: str = "work",
) -> ActiveDialogueState:
    state = initial or ActiveDialogueState(response_mode=response_mode)
    for turn in turns:
        state = update_active_dialogue(state, turn, response_mode=response_mode)
    return state


def inject_active_dialogue_state(prompt: str, state: ActiveDialogueState | None) -> str:
    """Inject ADS before the rest of a prompt/context packet.

    Empty ADS is a no-op.  Non-empty ADS is placed first so it outranks durable
    memory and archival recall in the assembled context.
    """

    if state is None or state.is_empty:
        return prompt
    if not prompt:
        return state.render()
    return f"{state.render()}\n{prompt}"
