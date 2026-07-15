"""Temporal fact model for governed PCLTM semantic memory.

Semantic memory stores stable reusable facts, not conversation logs.  Each fact is
valid for a time interval and can be superseded without erasing the historical
record that used to be true.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import UTC, datetime
from enum import StrEnum
from typing import Any
from uuid import uuid4


class SemanticNamespace(StrEnum):
    """Governed semantic memory namespaces."""

    USER_PREFERENCE = "user_preference"
    USER_PROFILE = "user_profile"
    RELATIONSHIP_ANCHOR = "relationship_anchor"
    PROJECT_FACT = "project_fact"
    RUNTIME_INVARIANT = "runtime_invariant"
    PERSONA_FACT = "persona_fact"
    BOUNDARY = "boundary"
    ENVIRONMENT_FACT = "environment_fact"


class Stability(StrEnum):
    """Expected durability of a semantic fact."""

    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    VERIFIED = "verified"


_ALLOWED_NAMESPACES = {item.value for item in SemanticNamespace}
_ALLOWED_STABILITY = {item.value for item in Stability}


def utc_now() -> datetime:
    return datetime.now(UTC)


def ensure_aware_utc(value: datetime | None) -> datetime | None:
    if value is None:
        return None
    if value.tzinfo is None:
        return value.replace(tzinfo=UTC)
    return value.astimezone(UTC)


def datetime_to_text(value: datetime | None) -> str | None:
    value = ensure_aware_utc(value)
    if value is None:
        return None
    return value.isoformat().replace("+00:00", "Z")


def datetime_from_text(value: str | None) -> datetime | None:
    if not value:
        return None
    return ensure_aware_utc(datetime.fromisoformat(value.replace("Z", "+00:00")))


@dataclass(frozen=True)
class TemporalFact:
    """A stable fact with temporal validity and provenance."""

    memory_id: str
    subject: str
    predicate: str
    object: str
    confidence: float
    valid_from: datetime
    valid_until: datetime | None = None
    source_refs: tuple[str, ...] = field(default_factory=tuple)
    stability: str = Stability.MEDIUM.value
    namespace: str = SemanticNamespace.PROJECT_FACT.value
    conflict_group: str | None = None
    supersedes: tuple[str, ...] = field(default_factory=tuple)
    superseded_by: str | None = None
    write_reason: str = ""
    last_verified_at: datetime | None = None

    def __post_init__(self) -> None:
        if not self.memory_id.strip():
            raise ValueError("memory_id is required")
        if not self.subject.strip():
            raise ValueError("subject is required")
        if not self.predicate.strip():
            raise ValueError("predicate is required")
        if self.object is None or not str(self.object).strip():
            raise ValueError("object is required")
        if not 0.0 <= float(self.confidence) <= 1.0:
            raise ValueError("confidence must be between 0.0 and 1.0")
        if self.namespace not in _ALLOWED_NAMESPACES:
            raise ValueError(f"unsupported semantic namespace: {self.namespace}")
        if self.stability not in _ALLOWED_STABILITY:
            raise ValueError(f"unsupported semantic stability: {self.stability}")
        valid_from = ensure_aware_utc(self.valid_from)
        valid_until = ensure_aware_utc(self.valid_until)
        last_verified_at = ensure_aware_utc(self.last_verified_at)
        if valid_until is not None and valid_until < valid_from:  # type: ignore[operator]
            raise ValueError("valid_until cannot be earlier than valid_from")
        object.__setattr__(self, "confidence", float(self.confidence))
        object.__setattr__(self, "valid_from", valid_from)
        object.__setattr__(self, "valid_until", valid_until)
        object.__setattr__(self, "last_verified_at", last_verified_at)
        object.__setattr__(self, "source_refs", tuple(ref for ref in self.source_refs if ref))
        object.__setattr__(self, "supersedes", tuple(mid for mid in self.supersedes if mid))

    @property
    def is_active(self) -> bool:
        return self.valid_until is None and self.superseded_by is None

    @property
    def fact_key(self) -> tuple[str, str, str]:
        return (self.namespace, self.subject, self.predicate)

    def with_supersession(
        self,
        *,
        superseded_by: str | None = None,
        valid_until: datetime | None = None,
        supersedes: tuple[str, ...] | None = None,
    ) -> "TemporalFact":
        return TemporalFact(
            memory_id=self.memory_id,
            subject=self.subject,
            predicate=self.predicate,
            object=self.object,
            confidence=self.confidence,
            valid_from=self.valid_from,
            valid_until=valid_until if valid_until is not None else self.valid_until,
            source_refs=self.source_refs,
            stability=self.stability,
            namespace=self.namespace,
            conflict_group=self.conflict_group,
            supersedes=supersedes if supersedes is not None else self.supersedes,
            superseded_by=superseded_by if superseded_by is not None else self.superseded_by,
            write_reason=self.write_reason,
            last_verified_at=self.last_verified_at,
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "memory_id": self.memory_id,
            "subject": self.subject,
            "predicate": self.predicate,
            "object": self.object,
            "confidence": self.confidence,
            "valid_from": datetime_to_text(self.valid_from),
            "valid_until": datetime_to_text(self.valid_until),
            "source_refs": list(self.source_refs),
            "stability": self.stability,
            "namespace": self.namespace,
            "conflict_group": self.conflict_group,
            "supersedes": list(self.supersedes),
            "superseded_by": self.superseded_by,
            "write_reason": self.write_reason,
            "last_verified_at": datetime_to_text(self.last_verified_at),
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "TemporalFact":
        return cls(
            memory_id=str(data["memory_id"]),
            subject=str(data["subject"]),
            predicate=str(data["predicate"]),
            object=str(data["object"]),
            confidence=float(data["confidence"]),
            valid_from=datetime_from_text(data.get("valid_from")) or utc_now(),
            valid_until=datetime_from_text(data.get("valid_until")),
            source_refs=tuple(data.get("source_refs") or ()),
            stability=str(data.get("stability") or Stability.MEDIUM.value),
            namespace=str(data.get("namespace") or SemanticNamespace.PROJECT_FACT.value),
            conflict_group=data.get("conflict_group"),
            supersedes=tuple(data.get("supersedes") or ()),
            superseded_by=data.get("superseded_by"),
            write_reason=str(data.get("write_reason") or ""),
            last_verified_at=datetime_from_text(data.get("last_verified_at")),
        )


def new_memory_id(prefix: str = "sem") -> str:
    return f"{prefix}_{uuid4().hex}"


def default_conflict_group(namespace: str, subject: str, predicate: str) -> str:
    clean = "::".join(part.strip().lower() for part in (namespace, subject, predicate))
    return clean
