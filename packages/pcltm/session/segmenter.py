"""Raw-turn segment boundaries for Session Summary Chain.

The segmenter is intentionally deterministic and evidence-preserving. It only
chooses boundaries over raw in-session turns; it never consumes prior summary
text as source material.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Sequence

from pcltm.state.update_active_dialogue import DialogueTurn


@dataclass(frozen=True)
class SegmentBoundary:
    """Inclusive-exclusive raw turn range for one session segment."""

    segment_id: int
    start_turn: int
    end_turn: int
    reason: str = "token_boundary"
    raw_message_refs: tuple[str, ...] = field(default_factory=tuple)

    def __post_init__(self) -> None:
        if self.start_turn < 0:
            raise ValueError("start_turn must be non-negative")
        if self.end_turn < self.start_turn:
            raise ValueError("end_turn must be >= start_turn")
        if not self.raw_message_refs:
            refs = tuple(f"turn:{i}" for i in range(self.start_turn, self.end_turn))
            object.__setattr__(self, "raw_message_refs", refs)


def _turn_text(turn: DialogueTurn) -> str:
    return "\n".join(part for part in (turn.user, turn.assistant) if part)


def _rough_token_count(turn: DialogueTurn) -> int:
    # Cheap deterministic estimate good enough for boundary control.
    text = _turn_text(turn)
    return max(1, len(text) // 4)


def _looks_like_event_boundary(turn: DialogueTurn) -> bool:
    user = (turn.user or "").lower()
    assistant = (turn.assistant or "").lower()
    markers = (
        "阶段",
        "phase",
        "下一步",
        "next",
        "停",
        "暂停",
        "撤销",
        "回滚",
        "完成后提交",
        "commit",
        "验收",
        "acceptance",
    )
    return any(marker in user or marker in assistant for marker in markers)


def segment_turns(
    turns: Sequence[DialogueTurn],
    *,
    max_turns_per_segment: int = 8,
    max_tokens_per_segment: int | None = None,
    event_boundaries: bool = True,
) -> tuple[SegmentBoundary, ...]:
    """Split raw dialogue turns into traceable segment boundaries.

    Boundaries are based on event markers and rough token limits. Existing
    summary text is not accepted here, which keeps this layer from becoming a
    recursive compression chain.
    """

    if max_turns_per_segment <= 0:
        raise ValueError("max_turns_per_segment must be positive")
    if max_tokens_per_segment is not None and max_tokens_per_segment <= 0:
        raise ValueError("max_tokens_per_segment must be positive")
    if not turns:
        return ()

    boundaries: list[SegmentBoundary] = []
    start = 0
    token_count = 0

    for index, turn in enumerate(turns):
        token_count += _rough_token_count(turn)
        turn_limit_hit = index + 1 - start >= max_turns_per_segment
        token_limit_hit = max_tokens_per_segment is not None and token_count >= max_tokens_per_segment
        event_hit = event_boundaries and index > start and _looks_like_event_boundary(turn)

        if turn_limit_hit or token_limit_hit or event_hit:
            reason = "event_boundary" if event_hit else "token_boundary"
            boundaries.append(
                SegmentBoundary(
                    segment_id=len(boundaries) + 1,
                    start_turn=start,
                    end_turn=index + 1,
                    reason=reason,
                )
            )
            start = index + 1
            token_count = 0

    if start < len(turns):
        boundaries.append(
            SegmentBoundary(
                segment_id=len(boundaries) + 1,
                start_turn=start,
                end_turn=len(turns),
                reason="tail",
            )
        )

    return tuple(boundaries)
