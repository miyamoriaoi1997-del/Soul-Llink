"""Session Summary Chain for long current-session continuity.

This layer is a traceable chain over raw in-session turns. It is not recursive
compression: every segment is derived from raw user/assistant turns and keeps
raw refs so the original evidence can be recovered when needed.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Iterable, Mapping, Sequence

from pcltm.state.active_dialogue_state import ActiveDialogueState
from pcltm.state.update_active_dialogue import DialogueTurn, update_from_turns

from .segmenter import segment_turns


def _clean_text(value: Any, *, max_chars: int = 240) -> str:
    text = " ".join(str(value or "").strip().split())
    if len(text) > max_chars:
        return text[: max_chars - 1].rstrip() + "…"
    return text


def _append_unique(values: Iterable[str], value: str, *, max_items: int = 8) -> tuple[str, ...]:
    cleaned = _clean_text(value, max_chars=220)
    output = [v for v in values if v]
    if cleaned and cleaned not in output:
        output.append(cleaned)
    return tuple(output[-max_items:])


def _turn_text(turn: DialogueTurn) -> str:
    return "\n".join(part for part in (turn.user, turn.assistant) if part)


def _contains_any(text: str, needles: Sequence[str]) -> bool:
    lowered = text.lower()
    return any(needle.lower() in lowered for needle in needles)


_DONE_MARKERS = ("完成", "完工", "done", "passed", "提交", "committed", "已解决")
_PAUSE_MARKERS = ("暂停", "先停", "停停", "pause", "blocked", "等待")
_EXPLICIT_REVOKE_MARKERS = ("撤销", "回滚", "取消", "别继续", "revert", "cancel")
_DECISION_MARKERS = ("决定", "采用", "不采用", "边界", "必须", "不能", "优先", "注意")
_COMMITMENT_MARKERS = ("我会", "完成", "提交", "验证", "测试", "will", "done", "passed")
_MEMORY_MARKERS = ("记住", "偏好", "以后", "用户不喜欢", "preference", "remember")


@dataclass(frozen=True)
class SessionSegment:
    """One raw-turn-derived current-session segment."""

    segment_id: int
    time_range: tuple[int, int]
    raw_message_refs: tuple[str, ...]
    local_summary: str = ""
    decisions: tuple[str, ...] = field(default_factory=tuple)
    commitments: tuple[str, ...] = field(default_factory=tuple)
    unresolved_items: tuple[str, ...] = field(default_factory=tuple)
    emotional_delta: str = ""
    memory_candidates: tuple[str, ...] = field(default_factory=tuple)
    verification_notes: tuple[str, ...] = field(default_factory=tuple)
    completed_items: tuple[str, ...] = field(default_factory=tuple)
    paused_items: tuple[str, ...] = field(default_factory=tuple)
    revoked_items: tuple[str, ...] = field(default_factory=tuple)

    # Backward-compatible fields used by the first phase-2 tests and renderer.
    current_task: str = ""
    last_user_intent: str = ""
    open_threads: tuple[str, ...] = field(default_factory=tuple)
    pending_questions: tuple[str, ...] = field(default_factory=tuple)
    local_constraints: tuple[str, ...] = field(default_factory=tuple)
    evidence: tuple[str, ...] = field(default_factory=tuple)

    @property
    def start_turn(self) -> int:
        return self.time_range[0]

    @property
    def end_turn(self) -> int:
        return self.time_range[1]

    def __post_init__(self) -> None:
        start, end = self.time_range
        if start < 0 or end < start:
            raise ValueError("time_range must be a non-negative inclusive-exclusive range")
        if not self.raw_message_refs:
            object.__setattr__(self, "raw_message_refs", tuple(f"turn:{i}" for i in range(start, end)))
        for attr in (
            "decisions",
            "commitments",
            "unresolved_items",
            "memory_candidates",
            "verification_notes",
            "completed_items",
            "paused_items",
            "revoked_items",
            "open_threads",
            "pending_questions",
            "local_constraints",
            "evidence",
        ):
            object.__setattr__(self, attr, tuple(_clean_text(x, max_chars=220) for x in getattr(self, attr) if _clean_text(x)))
        for attr in ("local_summary", "emotional_delta", "current_task", "last_user_intent"):
            object.__setattr__(self, attr, _clean_text(getattr(self, attr), max_chars=360))

    def to_dict(self) -> dict[str, Any]:
        return {
            "segment_id": self.segment_id,
            "time_range": list(self.time_range),
            "raw_message_refs": list(self.raw_message_refs),
            "local_summary": self.local_summary,
            "decisions": list(self.decisions),
            "commitments": list(self.commitments),
            "unresolved_items": list(self.unresolved_items),
            "emotional_delta": self.emotional_delta,
            "memory_candidates": list(self.memory_candidates),
            "verification_notes": list(self.verification_notes),
            "completed_items": list(self.completed_items),
            "paused_items": list(self.paused_items),
            "revoked_items": list(self.revoked_items),
            "current_task": self.current_task,
            "last_user_intent": self.last_user_intent,
            "open_threads": list(self.open_threads),
            "pending_questions": list(self.pending_questions),
            "local_constraints": list(self.local_constraints),
            "evidence": list(self.evidence),
        }

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> "SessionSegment":
        start = int(value.get("start_turn", 0))
        end = int(value.get("end_turn", start))
        time_range_value = value.get("time_range")
        if isinstance(time_range_value, Sequence) and len(time_range_value) == 2:
            start, end = int(time_range_value[0]), int(time_range_value[1])
        return cls(
            segment_id=int(value.get("segment_id", 0)),
            time_range=(start, end),
            raw_message_refs=tuple(value.get("raw_message_refs") or ()),
            local_summary=str(value.get("local_summary") or value.get("current_task") or ""),
            decisions=tuple(value.get("decisions") or ()),
            commitments=tuple(value.get("commitments") or ()),
            unresolved_items=tuple(value.get("unresolved_items") or value.get("open_threads") or ()),
            emotional_delta=str(value.get("emotional_delta") or ""),
            memory_candidates=tuple(value.get("memory_candidates") or ()),
            verification_notes=tuple(value.get("verification_notes") or value.get("evidence") or ()),
            completed_items=tuple(value.get("completed_items") or ()),
            paused_items=tuple(value.get("paused_items") or ()),
            revoked_items=tuple(value.get("revoked_items") or ()),
            current_task=str(value.get("current_task") or value.get("local_summary") or ""),
            last_user_intent=str(value.get("last_user_intent") or ""),
            open_threads=tuple(value.get("open_threads") or value.get("unresolved_items") or ()),
            pending_questions=tuple(value.get("pending_questions") or ()),
            local_constraints=tuple(value.get("local_constraints") or ()),
            evidence=tuple(value.get("evidence") or value.get("verification_notes") or ()),
        )

    def render(self) -> str:
        lines = [f"  - segment {self.segment_id} turns {self.start_turn}-{self.end_turn}"]
        if self.local_summary:
            lines.append(f"    local_summary: {self.local_summary}")
        if self.raw_message_refs:
            lines.append("    raw_refs: " + ", ".join(self.raw_message_refs))
        if self.decisions:
            lines.append("    decisions: " + " | ".join(self.decisions))
        if self.commitments:
            lines.append("    commitments: " + " | ".join(self.commitments))
        if self.unresolved_items:
            lines.append("    unresolved: " + " | ".join(self.unresolved_items))
        if self.completed_items:
            lines.append("    completed: " + " | ".join(self.completed_items))
        if self.paused_items:
            lines.append("    paused: " + " | ".join(self.paused_items))
        if self.revoked_items:
            lines.append("    revoked: " + " | ".join(self.revoked_items))
        if self.verification_notes:
            lines.append("    verification: " + " | ".join(self.verification_notes))
        return "\n".join(lines)


SessionSummarySegment = SessionSegment


@dataclass(frozen=True)
class SessionSummaryChain:
    """Append-only, raw-ref-backed current-session summary chain."""

    session_id: str = "current"
    segments: tuple[SessionSegment, ...] = field(default_factory=tuple)
    current_spine: str = ""
    unresolved_index: tuple[str, ...] = field(default_factory=tuple)
    decision_index: tuple[str, ...] = field(default_factory=tuple)
    commitment_index: tuple[str, ...] = field(default_factory=tuple)
    completed_index: tuple[str, ...] = field(default_factory=tuple)
    paused_index: tuple[str, ...] = field(default_factory=tuple)
    revoked_index: tuple[str, ...] = field(default_factory=tuple)
    active_dialogue_state: ActiveDialogueState | None = None
    source_turn_count: int = 0
    segment_size: int = 8
    max_segments: int = 12

    @property
    def is_empty(self) -> bool:
        return not self.segments and not self.current_spine

    @property
    def current_task(self) -> str:
        if self.current_spine:
            return self.current_spine
        if self.active_dialogue_state and self.active_dialogue_state.current_task:
            return self.active_dialogue_state.current_task
        return self.segments[-1].current_task if self.segments else ""

    def __post_init__(self) -> None:
        segments = tuple(self.segments[-self.max_segments :]) if self.max_segments > 0 else tuple(self.segments)
        object.__setattr__(self, "segments", segments)
        if not self.current_spine:
            object.__setattr__(self, "current_spine", self.current_task)
        for attr in (
            "unresolved_index",
            "decision_index",
            "commitment_index",
            "completed_index",
            "paused_index",
            "revoked_index",
        ):
            object.__setattr__(self, attr, tuple(_clean_text(x, max_chars=220) for x in getattr(self, attr) if _clean_text(x)))

    def to_dict(self) -> dict[str, Any]:
        return {
            "session_id": self.session_id,
            "segments": [segment.to_dict() for segment in self.segments],
            "current_spine": self.current_spine,
            "unresolved_index": list(self.unresolved_index),
            "decision_index": list(self.decision_index),
            "commitment_index": list(self.commitment_index),
            "completed_index": list(self.completed_index),
            "paused_index": list(self.paused_index),
            "revoked_index": list(self.revoked_index),
            "active_dialogue_state": self.active_dialogue_state.to_dict() if self.active_dialogue_state else None,
            "source_turn_count": self.source_turn_count,
            "segment_size": self.segment_size,
            "max_segments": self.max_segments,
        }

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> "SessionSummaryChain":
        ads_value = value.get("active_dialogue_state")
        return cls(
            session_id=str(value.get("session_id") or "current"),
            segments=tuple(SessionSegment.from_dict(v) for v in value.get("segments") or ()),
            current_spine=str(value.get("current_spine") or value.get("current_task") or ""),
            unresolved_index=tuple(value.get("unresolved_index") or ()),
            decision_index=tuple(value.get("decision_index") or ()),
            commitment_index=tuple(value.get("commitment_index") or ()),
            completed_index=tuple(value.get("completed_index") or ()),
            paused_index=tuple(value.get("paused_index") or ()),
            revoked_index=tuple(value.get("revoked_index") or ()),
            active_dialogue_state=ActiveDialogueState.from_dict(ads_value) if isinstance(ads_value, Mapping) else None,
            source_turn_count=int(value.get("source_turn_count", 0)),
            segment_size=int(value.get("segment_size", 8)),
            max_segments=int(value.get("max_segments", 12)),
        )

    def render(self) -> str:
        """Render the limited injection view, not the whole chain as context."""
        if self.is_empty:
            return ""
        lines = ["【session_summary_chain】"]
        if self.current_spine:
            lines.append(f"current_spine: {self.current_spine}")
        if self.unresolved_index:
            lines.append("unresolved_items: " + " | ".join(self.unresolved_index))
        if self.decision_index:
            lines.append("recent_decisions: " + " | ".join(self.decision_index[-6:]))
        if self.commitment_index:
            lines.append("recent_commitments: " + " | ".join(self.commitment_index[-6:]))
        lines.append("construction: raw_turn_segments_not_summary_of_summaries")
        lines.append(f"source_turn_count: {self.source_turn_count}")
        if self.segments:
            lines.append("last_segment_summary:")
            lines.append(self.segments[-1].render())
        return "\n".join(lines)


def summarize_segment(
    turns: Sequence[DialogueTurn],
    *,
    segment_id: int,
    start_turn: int,
    prior_task: str = "",
) -> SessionSegment:
    current_task = _clean_text(prior_task)
    last_user_intent = ""
    decisions: tuple[str, ...] = ()
    commitments: tuple[str, ...] = ()
    unresolved: tuple[str, ...] = ()
    completed: tuple[str, ...] = ()
    paused: tuple[str, ...] = ()
    revoked: tuple[str, ...] = ()
    memory_candidates: tuple[str, ...] = ()
    verification: tuple[str, ...] = ()
    pending_questions: tuple[str, ...] = ()
    constraints: tuple[str, ...] = ()

    for offset, turn in enumerate(turns):
        text = _turn_text(turn)
        previous_task = current_task
        if turn.user:
            last_user_intent = _clean_text(turn.user)
            current_task = _clean_text(turn.user)
            unresolved = _append_unique(unresolved, turn.user)
            if "?" in turn.user or "？" in turn.user:
                pending_questions = _append_unique(pending_questions, turn.user, max_items=4)
            if _contains_any(turn.user, _DECISION_MARKERS):
                decisions = _append_unique(decisions, turn.user)
            if _contains_any(turn.user, _MEMORY_MARKERS):
                memory_candidates = _append_unique(memory_candidates, turn.user, max_items=4)
        if turn.assistant:
            if _contains_any(turn.assistant, _COMMITMENT_MARKERS):
                commitments = _append_unique(commitments, turn.assistant, max_items=4)
            if _contains_any(turn.assistant, ("passed", "通过", "验证", "测试")):
                verification = _append_unique(verification, turn.assistant, max_items=4)
        if _contains_any(text, _DONE_MARKERS):
            completed = _append_unique(completed, current_task or text, max_items=6)
            unresolved = tuple(item for item in unresolved if item != (current_task or text))
        if _contains_any(text, _PAUSE_MARKERS):
            paused = _append_unique(paused, current_task or text, max_items=6)
        if _contains_any(text, _EXPLICIT_REVOKE_MARKERS):
            revoked_task = (
                previous_task
                if turn.user and _contains_any(turn.user, _EXPLICIT_REVOKE_MARKERS) and previous_task
                else current_task or text
            )
            revoked = _append_unique(revoked, revoked_task, max_items=6)
            unresolved = tuple(item for item in unresolved if item != revoked_task)
        if _contains_any(text, ("不能", "必须", "不要", "优先", "边界")):
            constraints = _append_unique(constraints, text, max_items=4)

    local_summary = current_task or last_user_intent or (turns and _turn_text(turns[-1])) or ""
    refs = tuple(f"turn:{i}" for i in range(start_turn, start_turn + len(turns)))
    emotional_delta = "work-mode continuity pressure present" if any("用户" in _turn_text(t) for t in turns) else ""
    return SessionSegment(
        segment_id=segment_id,
        time_range=(start_turn, start_turn + len(turns)),
        raw_message_refs=refs,
        local_summary=local_summary,
        decisions=decisions,
        commitments=commitments,
        unresolved_items=unresolved,
        emotional_delta=emotional_delta,
        memory_candidates=memory_candidates,
        verification_notes=verification,
        completed_items=completed,
        paused_items=paused,
        revoked_items=revoked,
        current_task=current_task,
        last_user_intent=last_user_intent,
        open_threads=unresolved,
        pending_questions=pending_questions,
        local_constraints=constraints,
        evidence=verification or refs,
    )


def _chain_from_segments(
    segments: Sequence[SessionSegment],
    *,
    session_id: str,
    active_dialogue_state: ActiveDialogueState | None,
    source_turn_count: int,
    segment_size: int,
    max_segments: int,
) -> SessionSummaryChain:
    unresolved: tuple[str, ...] = ()
    decisions: tuple[str, ...] = ()
    commitments: tuple[str, ...] = ()
    completed: tuple[str, ...] = ()
    paused: tuple[str, ...] = ()
    revoked: tuple[str, ...] = ()
    for segment in segments:
        for item in segment.completed_items:
            completed = _append_unique(completed, item)
            unresolved = tuple(x for x in unresolved if x != item)
        for item in segment.revoked_items:
            revoked = _append_unique(revoked, item)
            unresolved = tuple(x for x in unresolved if x != item)
        for item in segment.paused_items:
            paused = _append_unique(paused, item)
            unresolved = tuple(x for x in unresolved if x != item)
        for item in segment.unresolved_items:
            if item not in completed and item not in paused and item not in revoked:
                unresolved = _append_unique(unresolved, item)
        for item in segment.decisions:
            decisions = _append_unique(decisions, item)
        for item in segment.commitments:
            commitments = _append_unique(commitments, item)
    current_spine = ""
    if active_dialogue_state and active_dialogue_state.current_task:
        current_spine = active_dialogue_state.current_task
    elif segments:
        current_spine = segments[-1].current_task or segments[-1].local_summary
    return SessionSummaryChain(
        session_id=session_id,
        segments=tuple(segments),
        current_spine=current_spine,
        unresolved_index=unresolved,
        decision_index=decisions,
        commitment_index=commitments,
        completed_index=completed,
        paused_index=paused,
        revoked_index=revoked,
        active_dialogue_state=active_dialogue_state,
        source_turn_count=source_turn_count,
        segment_size=segment_size,
        max_segments=max_segments,
    )


def build_session_summary_chain(
    turns: Sequence[DialogueTurn],
    *,
    active_dialogue_state: ActiveDialogueState | None = None,
    session_id: str = "current",
    segment_size: int = 8,
    max_segments: int = 12,
    max_tokens_per_segment: int | None = None,
) -> SessionSummaryChain:
    if segment_size <= 0:
        raise ValueError("segment_size must be positive")
    if not turns:
        return SessionSummaryChain(session_id=session_id, active_dialogue_state=active_dialogue_state, segment_size=segment_size, max_segments=max_segments)
    active_dialogue_state = active_dialogue_state or update_from_turns(turns)
    boundaries = segment_turns(turns, max_turns_per_segment=segment_size, max_tokens_per_segment=max_tokens_per_segment)
    built_segments: list[SessionSegment] = []
    prior_task = ""
    for boundary in boundaries:
        segment = summarize_segment(
            turns[boundary.start_turn : boundary.end_turn],
            segment_id=boundary.segment_id,
            start_turn=boundary.start_turn,
            prior_task=prior_task,
        )
        built_segments.append(segment)
        prior_task = segment.current_task or prior_task
    segments = tuple(built_segments)
    if max_segments > 0 and len(segments) > max_segments:
        segments = segments[-max_segments:]
    return _chain_from_segments(
        segments,
        session_id=session_id,
        active_dialogue_state=active_dialogue_state,
        source_turn_count=len(turns),
        segment_size=segment_size,
        max_segments=max_segments,
    )


def append_turns_to_chain(
    existing: SessionSummaryChain | None,
    turns: Sequence[DialogueTurn],
    *,
    active_dialogue_state: ActiveDialogueState | None = None,
    segment_size: int | None = None,
    max_segments: int | None = None,
) -> SessionSummaryChain:
    # Rebuild from raw turns only. Existing summaries are not used as source
    # material, which avoids recursive summary inheritance.
    return build_session_summary_chain(
        turns,
        active_dialogue_state=active_dialogue_state or (existing.active_dialogue_state if existing else None),
        session_id=existing.session_id if existing else "current",
        segment_size=segment_size or (existing.segment_size if existing else 8),
        max_segments=max_segments or (existing.max_segments if existing else 12),
    )


def is_summary_continuation_only(turn: DialogueTurn) -> bool:
    text = _clean_text(turn.user or "")
    lowered = text.lower()
    return lowered in {"继续", "继续任务", "continue"} or lowered.startswith("继续 ")
