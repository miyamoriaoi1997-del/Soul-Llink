"""Typed memory object contract for PCLTM.

This module is intentionally runtime-neutral. It defines the stable object
shape used by governance, retrieval, and future storage adapters without
mutating the existing event store or MemFS paths.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Mapping


class MemoryObjectType(str, Enum):
    """High-level role of a memory object."""

    IDENTITY = "identity_memory"
    RELATIONSHIP = "relationship_memory"
    PREFERENCE = "preference_memory"
    PROJECT = "project_memory"
    PROCEDURAL = "procedural_memory"
    EPISODIC = "episodic_memory"
    STATE_TRACE = "state_trace"
    TOOL_EVIDENCE = "tool_evidence"
    CONFLICT = "conflict_record"
    RETIRED = "retired_memory"


class MemoryObjectScope(str, Enum):
    """Where a memory object is allowed to apply."""

    GLOBAL = "global"
    USER = "user"
    PROJECT = "project"
    SESSION = "session"
    RUNTIME = "runtime"


class MemoryObjectStatus(str, Enum):
    """Governance lifecycle state."""

    PENDING = "pending"
    APPROVED = "approved"
    QUARANTINED = "quarantined"
    RETIRED = "retired"


class InjectionPolicy(str, Enum):
    """How the context composer may inject the object."""

    PINNED = "pinned"
    SELECTIVE = "selective"
    EVIDENCE_ONLY = "evidence_only"
    NEVER = "never"


@dataclass(frozen=True)
class StateAffinity:
    """State-machine hints used by retrieval without overriding identity."""

    modes: tuple[str, ...] = ()
    emotion_axes: tuple[str, ...] = ()
    min_intensity: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "modes": list(self.modes),
            "emotion_axes": list(self.emotion_axes),
            "min_intensity": self.min_intensity,
        }

    @classmethod
    def from_dict(cls, data: Mapping[str, Any] | None) -> "StateAffinity":
        if not data:
            return cls()
        return cls(
            modes=tuple(str(item) for item in data.get("modes", ()) or ()),
            emotion_axes=tuple(str(item) for item in data.get("emotion_axes", ()) or ()),
            min_intensity=data.get("min_intensity"),
        )


@dataclass(frozen=True)
class MemoryObject:
    """A governed, typed unit of PCLTM memory.

    The object is separate from persistence on purpose. Store adapters can map
    this contract to SQLite, MemFS, or future block-like storage without
    changing retrieval and context-composer semantics.
    """

    canonical_key: str
    object_type: MemoryObjectType
    content: str
    scope: MemoryObjectScope = MemoryObjectScope.USER
    status: MemoryObjectStatus = MemoryObjectStatus.PENDING
    injection_policy: InjectionPolicy = InjectionPolicy.SELECTIVE
    source: str | None = None
    confidence: float = 1.0
    stability: float = 0.5
    emotional_weight: float = 0.0
    budget_weight: float = 1.0
    state_affinity: StateAffinity = field(default_factory=StateAffinity)
    tags: tuple[str, ...] = ()
    conflict_keys: tuple[str, ...] = ()
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not self.canonical_key.strip():
            raise ValueError("canonical_key is required")
        if not self.content.strip():
            raise ValueError("content is required")
        _validate_unit_interval("confidence", self.confidence)
        _validate_unit_interval("stability", self.stability)
        _validate_unit_interval("emotional_weight", self.emotional_weight)
        if self.budget_weight <= 0:
            raise ValueError("budget_weight must be positive")
        if self.object_type is MemoryObjectType.IDENTITY and self.status is not MemoryObjectStatus.QUARANTINED and self.injection_policy is not InjectionPolicy.PINNED:
            raise ValueError("active identity memories must use pinned injection")
        if self.status is MemoryObjectStatus.RETIRED and self.injection_policy is not InjectionPolicy.NEVER:
            raise ValueError("retired memories must not be injectable")
        if self.status is MemoryObjectStatus.QUARANTINED and self.injection_policy is InjectionPolicy.PINNED:
            raise ValueError("quarantined memories cannot be pinned")

    def to_dict(self) -> dict[str, Any]:
        return {
            "canonical_key": self.canonical_key,
            "object_type": self.object_type.value,
            "content": self.content,
            "scope": self.scope.value,
            "status": self.status.value,
            "injection_policy": self.injection_policy.value,
            "source": self.source,
            "confidence": self.confidence,
            "stability": self.stability,
            "emotional_weight": self.emotional_weight,
            "budget_weight": self.budget_weight,
            "state_affinity": self.state_affinity.to_dict(),
            "tags": list(self.tags),
            "conflict_keys": list(self.conflict_keys),
            "metadata": dict(self.metadata),
        }

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "MemoryObject":
        return cls(
            canonical_key=str(data["canonical_key"]),
            object_type=MemoryObjectType(data["object_type"]),
            content=str(data["content"]),
            scope=MemoryObjectScope(data.get("scope", MemoryObjectScope.USER.value)),
            status=MemoryObjectStatus(data.get("status", MemoryObjectStatus.PENDING.value)),
            injection_policy=InjectionPolicy(data.get("injection_policy", InjectionPolicy.SELECTIVE.value)),
            source=data.get("source"),
            confidence=float(data.get("confidence", 1.0)),
            stability=float(data.get("stability", 0.5)),
            emotional_weight=float(data.get("emotional_weight", 0.0)),
            budget_weight=float(data.get("budget_weight", 1.0)),
            state_affinity=StateAffinity.from_dict(data.get("state_affinity")),
            tags=tuple(str(item) for item in data.get("tags", ()) or ()),
            conflict_keys=tuple(str(item) for item in data.get("conflict_keys", ()) or ()),
            metadata=dict(data.get("metadata", {}) or {}),
        )

    @property
    def injectable(self) -> bool:
        return self.status is MemoryObjectStatus.APPROVED and self.injection_policy is not InjectionPolicy.NEVER

    def matches_state(self, mode: str | None, emotion_axes: set[str] | None = None) -> bool:
        """Return whether this object is eligible for a runtime state.

        State affinity can narrow selection, but absence of hints means the
        object is state-neutral.
        """

        if self.state_affinity.modes and mode not in self.state_affinity.modes:
            return False
        if self.state_affinity.emotion_axes and not (emotion_axes or set()).intersection(self.state_affinity.emotion_axes):
            return False
        return True


def _validate_unit_interval(name: str, value: float) -> None:
    if value < 0 or value > 1:
        raise ValueError(f"{name} must be between 0 and 1")
