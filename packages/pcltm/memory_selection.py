from __future__ import annotations

from dataclasses import dataclass
from enum import Enum

from .memory_object import InjectionPolicy, MemoryObject, MemoryObjectStatus, MemoryObjectType


class PriorityClass(str, Enum):
    """Human-readable selection priority class for observability."""

    PINNED_IDENTITY = "pinned_identity"
    PINNED = "pinned"
    SELECTIVE = "selective"
    EVIDENCE_ONLY = "evidence_only"
    REJECTED = "rejected"


@dataclass(frozen=True)
class SelectionDecision:
    """Read-only explanation for a memory object's prompt eligibility."""

    selected: bool
    reason: str
    canonical_key: str
    priority_class: PriorityClass
    budget_weight: float
    matched_mode: str | None = None
    matched_emotion_axes: tuple[str, ...] = ()
    rejected_reason: str | None = None

    def to_dict(self) -> dict[str, object]:
        return {
            "selected": self.selected,
            "reason": self.reason,
            "canonical_key": self.canonical_key,
            "priority_class": self.priority_class.value,
            "budget_weight": self.budget_weight,
            "matched_mode": self.matched_mode,
            "matched_emotion_axes": list(self.matched_emotion_axes),
            "rejected_reason": self.rejected_reason,
        }


def explain_memory_selection(
    memory: MemoryObject,
    *,
    mode: str | None,
    emotion_axes: set[str] | None,
    budget_available: float | None,
) -> SelectionDecision:
    """Explain whether a memory object is eligible for prompt injection.

    This is an observability primitive only. It does not sort, mutate, persist,
    or call the existing retrieval path.
    """

    priority_class = _priority_class(memory)
    if memory.status is not MemoryObjectStatus.APPROVED:
        status_priority = PriorityClass.EVIDENCE_ONLY if memory.status is MemoryObjectStatus.QUARANTINED and memory.injection_policy is InjectionPolicy.EVIDENCE_ONLY else PriorityClass.REJECTED
        return _rejected(
            memory,
            reason="not eligible for injection",
            rejected_reason=f"status={memory.status.value}",
            priority_class=status_priority,
        )
    if memory.injection_policy is InjectionPolicy.NEVER:
        return _rejected(
            memory,
            reason="not eligible for injection",
            rejected_reason="injection_policy=never",
            priority_class=PriorityClass.REJECTED,
        )
    if memory.injection_policy is InjectionPolicy.EVIDENCE_ONLY:
        return _rejected(
            memory,
            reason="evidence-only memory",
            rejected_reason="injection_policy=evidence_only",
            priority_class=PriorityClass.EVIDENCE_ONLY,
        )

    matched_mode = _matched_mode(memory, mode)
    matched_axes = _matched_emotion_axes(memory, emotion_axes)
    if not memory.matches_state(mode, emotion_axes):
        return _rejected(
            memory,
            reason="state affinity mismatch",
            rejected_reason=_state_rejection_reason(memory, mode, emotion_axes),
            priority_class=PriorityClass.REJECTED,
            matched_mode=matched_mode,
            matched_emotion_axes=matched_axes,
        )

    if budget_available is not None and memory.budget_weight > budget_available:
        return _rejected(
            memory,
            reason="insufficient budget",
            rejected_reason="budget_weight_exceeds_available",
            priority_class=PriorityClass.REJECTED,
            matched_mode=matched_mode,
            matched_emotion_axes=matched_axes,
        )

    if memory.object_type is MemoryObjectType.IDENTITY and memory.injection_policy is InjectionPolicy.PINNED:
        reason = "pinned identity"
    elif memory.injection_policy is InjectionPolicy.PINNED:
        reason = "pinned memory"
    else:
        reason = "approved selective memory matched state"

    return SelectionDecision(
        selected=True,
        reason=reason,
        canonical_key=memory.canonical_key,
        priority_class=priority_class,
        budget_weight=memory.budget_weight,
        matched_mode=matched_mode,
        matched_emotion_axes=matched_axes,
    )


def _priority_class(memory: MemoryObject) -> PriorityClass:
    if memory.injection_policy is InjectionPolicy.EVIDENCE_ONLY:
        return PriorityClass.EVIDENCE_ONLY
    if memory.injection_policy is InjectionPolicy.PINNED:
        if memory.object_type is MemoryObjectType.IDENTITY:
            return PriorityClass.PINNED_IDENTITY
        return PriorityClass.PINNED
    if memory.injection_policy is InjectionPolicy.SELECTIVE:
        return PriorityClass.SELECTIVE
    return PriorityClass.REJECTED


def _matched_mode(memory: MemoryObject, mode: str | None) -> str | None:
    if not memory.state_affinity.modes:
        return mode
    return mode if mode in memory.state_affinity.modes else None


def _matched_emotion_axes(memory: MemoryObject, emotion_axes: set[str] | None) -> tuple[str, ...]:
    if not memory.state_affinity.emotion_axes:
        return ()
    axes = emotion_axes or set()
    return tuple(axis for axis in memory.state_affinity.emotion_axes if axis in axes)


def _state_rejection_reason(memory: MemoryObject, mode: str | None, emotion_axes: set[str] | None) -> str:
    if memory.state_affinity.modes and mode not in memory.state_affinity.modes:
        return "mode_mismatch"
    if memory.state_affinity.emotion_axes and not (emotion_axes or set()).intersection(memory.state_affinity.emotion_axes):
        return "emotion_axes_mismatch"
    return "state_mismatch"


def _rejected(
    memory: MemoryObject,
    *,
    reason: str,
    rejected_reason: str,
    priority_class: PriorityClass,
    matched_mode: str | None = None,
    matched_emotion_axes: tuple[str, ...] = (),
) -> SelectionDecision:
    return SelectionDecision(
        selected=False,
        reason=reason,
        canonical_key=memory.canonical_key,
        priority_class=priority_class,
        budget_weight=memory.budget_weight,
        matched_mode=matched_mode,
        matched_emotion_axes=matched_emotion_axes,
        rejected_reason=rejected_reason,
    )
