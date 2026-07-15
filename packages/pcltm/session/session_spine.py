"""Session spine projection for current-session continuity.

The spine is the small injected view of a Session Summary Chain: current main
line, unresolved work, recent decisions, and the last raw-turn-derived segment.
It is not a replacement for the chain and does not rewrite older summaries.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Sequence


@dataclass(frozen=True)
class SessionSpine:
    current_spine: str = ""
    unresolved_items: tuple[str, ...] = field(default_factory=tuple)
    recent_decisions: tuple[str, ...] = field(default_factory=tuple)
    last_segment_summary: str = ""
    raw_message_refs: tuple[str, ...] = field(default_factory=tuple)

    @property
    def is_empty(self) -> bool:
        return not any((self.current_spine, self.unresolved_items, self.recent_decisions, self.last_segment_summary))

    def render(self) -> str:
        if self.is_empty:
            return ""
        lines = ["[SessionSpine]"]
        if self.current_spine:
            lines.append(f"current_spine: {self.current_spine}")
        if self.unresolved_items:
            lines.append("unresolved_items: " + " | ".join(self.unresolved_items))
        if self.recent_decisions:
            lines.append("recent_decisions: " + " | ".join(self.recent_decisions))
        if self.last_segment_summary:
            lines.append(f"last_segment_summary: {self.last_segment_summary}")
        if self.raw_message_refs:
            lines.append("raw_message_refs: " + ", ".join(self.raw_message_refs))
        return "\n".join(lines)


def _first_nonempty(values: Sequence[str]) -> str:
    for value in values:
        if value:
            return value
    return ""


def _unique_tail(values: Sequence[str], *, max_items: int = 6) -> tuple[str, ...]:
    seen: set[str] = set()
    output: list[str] = []
    for value in reversed(values):
        value = (value or "").strip()
        if value and value not in seen:
            seen.add(value)
            output.append(value)
        if len(output) >= max_items:
            break
    return tuple(reversed(output))


def spine_from_chain(chain: object, *, recent_decision_limit: int = 6) -> SessionSpine:
    """Build the injection spine from a chain-like object.

    The function uses only the already materialized chain indexes and the last
    segment. It does not resummarize older summary text.
    """

    segments = tuple(getattr(chain, "segments", ()) or ())
    if not segments:
        return SessionSpine()

    current_spine = getattr(chain, "current_spine", "") or getattr(chain, "current_task", "")
    if callable(current_spine):
        current_spine = current_spine()
    unresolved = tuple(getattr(chain, "unresolved_index", ()) or ())
    decisions = _unique_tail(tuple(getattr(chain, "decision_index", ()) or ()), max_items=recent_decision_limit)
    last_segment = segments[-1]
    local_summary = getattr(last_segment, "local_summary", "") or ""
    refs = tuple(getattr(last_segment, "raw_message_refs", ()) or ())

    return SessionSpine(
        current_spine=_first_nonempty((str(current_spine), local_summary)),
        unresolved_items=unresolved,
        recent_decisions=decisions,
        last_segment_summary=local_summary,
        raw_message_refs=refs,
    )
