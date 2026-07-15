"""Procedural memory records for PCLTM.

Procedural memory stores reusable know-how: how to do a class of work. It is
kept separate from other PCLTM layers:

* persona memory stores identity, style, and relationship anchors;
* episodic memory stores what happened in a specific session;
* semantic memory stores stable factual knowledge; and
* procedural memory stores repeatable procedures worth turning into skills.

PCLTM procedural records decide what should be retained. Hermes skills carry
that procedure as executable/reusable operator guidance. This module therefore
models and validates the PCLTM side without mutating Hermes skill storage.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import UTC, datetime
from enum import StrEnum
from typing import Iterable, Mapping, Sequence


class SkillUpdateAction(StrEnum):
    """Allowed relationships between procedural memory and Hermes skills."""

    CREATE = "create"
    PATCH = "patch"
    SKIP = "skip"


_TASK_LOG_MARKERS = (
    "done",
    "finished",
    "completed",
    "phase",
    "阶段",
    "提交",
    "submitted",
    "commit",
    "sha",
    "pr #",
    "issue #",
    "ticket",
    "today",
    "yesterday",
    "tomorrow",
    "当前进度",
    "已完成",
    "完工",
)

_REUSABLE_MARKERS = (
    "when ",
    "use when",
    "trigger",
    "always",
    "before",
    "after",
    "verify",
    "pitfall",
    "avoid",
    "可复用",
    "流程",
    "步骤",
    "验证",
    "风险",
)


@dataclass(frozen=True)
class ProceduralMemory:
    """Reusable procedure candidate owned by PCLTM.

    The record is intentionally close to Hermes skill shape, but it is not a
    skill file. It records that a reusable workflow should be retained; exporting
    to Hermes is handled by :mod:`skill_exporter`.
    """

    skill_name: str
    trigger_conditions: tuple[str, ...]
    procedure: tuple[str, ...]
    verification_steps: tuple[str, ...] = ()
    pitfalls: tuple[str, ...] = ()
    source_sessions: tuple[str, ...] = ()
    last_updated: datetime = field(default_factory=lambda: datetime.now(UTC))
    confidence: float = 0.5
    category: str = "general"

    def __post_init__(self) -> None:
        object.__setattr__(self, "skill_name", _clean_name(self.skill_name))
        object.__setattr__(self, "category", (self.category or "general").strip())
        object.__setattr__(self, "trigger_conditions", _unique_clean(self.trigger_conditions))
        object.__setattr__(self, "procedure", _unique_clean(self.procedure))
        object.__setattr__(self, "verification_steps", _unique_clean(self.verification_steps))
        object.__setattr__(self, "pitfalls", _unique_clean(self.pitfalls))
        object.__setattr__(self, "source_sessions", _unique_clean(self.source_sessions))
        object.__setattr__(self, "confidence", max(0.0, min(1.0, float(self.confidence))))

        updated = self.last_updated
        if updated.tzinfo is None:
            updated = updated.replace(tzinfo=UTC)
        object.__setattr__(self, "last_updated", updated)

        if not self.skill_name:
            raise ValueError("skill_name is required")
        if not self.trigger_conditions:
            raise ValueError("trigger_conditions are required")
        if not self.procedure:
            raise ValueError("procedure is required")

    @property
    def is_exportable(self) -> bool:
        """Whether this record is safe to export to a Hermes skill.

        Export requires enough procedural substance and must reject session logs
        even if they were accidentally shaped like a procedure.
        """

        return (
            self.confidence >= 0.55
            and len(self.trigger_conditions) >= 1
            and len(self.procedure) >= 2
            and bool(self.verification_steps)
            and not self.looks_like_task_log()
        )

    def looks_like_task_log(self) -> bool:
        """Detect one-off progress/status records that must not become skills."""

        fields = [self.skill_name, *self.trigger_conditions, *self.procedure]
        text = "\n".join(fields).lower()
        marker_hits = sum(1 for marker in _TASK_LOG_MARKERS if marker in text)
        reusable_hits = sum(1 for marker in _REUSABLE_MARKERS if marker in text)

        # Explicit transient identifiers are strong evidence of a log entry.
        transient_id = any(token in text for token in ("commit ", "sha ", "pr #", "issue #"))
        if transient_id:
            return True

        # A record with mostly status wording and little procedural wording is a
        # log, not know-how.
        return marker_hits >= 2 and reusable_hits == 0

    def to_dict(self) -> dict[str, object]:
        """Serialize to a JSON-compatible dictionary."""

        return {
            "skill_name": self.skill_name,
            "trigger_conditions": list(self.trigger_conditions),
            "procedure": list(self.procedure),
            "verification_steps": list(self.verification_steps),
            "pitfalls": list(self.pitfalls),
            "source_sessions": list(self.source_sessions),
            "last_updated": self.last_updated.isoformat(),
            "confidence": self.confidence,
            "category": self.category,
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, object]) -> "ProceduralMemory":
        """Deserialize a procedural memory record."""

        raw_updated = payload.get("last_updated")
        if isinstance(raw_updated, datetime):
            updated = raw_updated
        elif isinstance(raw_updated, str) and raw_updated:
            updated = datetime.fromisoformat(raw_updated)
        else:
            updated = datetime.now(UTC)

        return cls(
            skill_name=str(payload.get("skill_name", "")),
            trigger_conditions=_as_tuple(payload.get("trigger_conditions")),
            procedure=_as_tuple(payload.get("procedure")),
            verification_steps=_as_tuple(payload.get("verification_steps")),
            pitfalls=_as_tuple(payload.get("pitfalls")),
            source_sessions=_as_tuple(payload.get("source_sessions")),
            last_updated=updated,
            confidence=_as_float(payload.get("confidence"), default=0.5),
            category=str(payload.get("category", "general")),
        )


def merge_procedural_memory(
    existing: ProceduralMemory,
    incoming: ProceduralMemory,
    *,
    now: datetime | None = None,
) -> ProceduralMemory:
    """Merge a newer candidate into an existing procedural memory record.

    The merge preserves the same reusable workflow while accumulating triggers,
    procedure refinements, verification steps, pitfalls, and source sessions. It
    is not a place for task progress; callers should only pass validated
    procedural records.
    """

    if existing.skill_name != incoming.skill_name:
        raise ValueError("cannot merge different procedural skills")

    timestamp = now or max(existing.last_updated, incoming.last_updated)
    if timestamp.tzinfo is None:
        timestamp = timestamp.replace(tzinfo=UTC)

    return ProceduralMemory(
        skill_name=existing.skill_name,
        trigger_conditions=_unique_clean((*existing.trigger_conditions, *incoming.trigger_conditions)),
        procedure=_unique_clean((*existing.procedure, *incoming.procedure)),
        verification_steps=_unique_clean((*existing.verification_steps, *incoming.verification_steps)),
        pitfalls=_unique_clean((*existing.pitfalls, *incoming.pitfalls)),
        source_sessions=_unique_clean((*existing.source_sessions, *incoming.source_sessions)),
        last_updated=timestamp,
        confidence=max(existing.confidence, incoming.confidence),
        category=incoming.category or existing.category,
    )


def _as_tuple(value: object) -> tuple[str, ...]:
    if value is None:
        return ()
    if isinstance(value, str):
        return (value,)
    if isinstance(value, Sequence):
        return tuple(str(item) for item in value)
    return (str(value),)


def _as_float(value: object, *, default: float) -> float:
    if value is None:
        return default
    try:
        return float(value)  # type: ignore[arg-type]
    except (TypeError, ValueError):
        return default


def _clean_name(value: str) -> str:
    return "-".join(str(value).strip().lower().replace("_", "-").split())


def _unique_clean(values: Iterable[str]) -> tuple[str, ...]:
    seen: set[str] = set()
    cleaned: list[str] = []
    for value in values:
        item = " ".join(str(value).strip().split())
        if not item:
            continue
        key = item.lower()
        if key in seen:
            continue
        seen.add(key)
        cleaned.append(item)
    return tuple(cleaned)
