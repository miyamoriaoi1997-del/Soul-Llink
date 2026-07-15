"""Injection candidate primitives for PCLTM arbitration.

Candidates are normalized context fragments. Retrieval may produce many
candidates, but only the arbitrator can promote them into an injected context
packet.
"""

from __future__ import annotations

from dataclasses import dataclass, field, replace
from enum import Enum
from typing import Any, Mapping


class CandidateType(str, Enum):
    SYSTEM = "system"
    CORE_IDENTITY = "core_identity"
    RUNTIME_MODE = "runtime_mode"
    ACTIVE_DIALOGUE = "active_dialogue"
    CURRENT_TASK = "current_task"
    PERSONA_EMOTION = "persona_emotion"
    SESSION_SPINE = "session_spine"
    EPISODIC_MEMORY = "episodic_memory"
    SEMANTIC_MEMORY = "semantic_memory"
    PROCEDURAL_SKILL = "procedural_skill"
    BACKGROUND_CONTEXT = "background_context"


class RiskFlag(str, Enum):
    CORE_CONFLICT = "core_conflict"
    CURRENT_TASK_CONFLICT = "current_task_conflict"
    NEW_FACT_CONFLICT = "new_fact_conflict"
    EMOTION_FACT_CONTAMINATION = "emotion_fact_contamination"
    MISSING_SOURCE = "missing_source"
    MISSING_CONFIDENCE = "missing_confidence"
    PERMANENT_EPISODIC_CLAIM = "permanent_episodic_claim"
    PROCEDURAL_NOT_NEEDED = "procedural_not_needed"
    CONFLICTING_MEMORY = "conflicting_memory"


class CandidateSource(str, Enum):
    RUNTIME = "runtime"
    PERSONA = "persona"
    ADS = "ads"
    TASK = "task"
    SESSION = "session"
    EPISODIC = "episodic"
    SEMANTIC = "semantic"
    PROCEDURAL = "procedural"
    BACKGROUND = "background"


INJECTION_PRIORITY: Mapping[CandidateType, int] = {
    CandidateType.SYSTEM: 0,
    CandidateType.CORE_IDENTITY: 1,
    CandidateType.RUNTIME_MODE: 2,
    CandidateType.ACTIVE_DIALOGUE: 3,
    CandidateType.CURRENT_TASK: 4,
    CandidateType.PERSONA_EMOTION: 5,
    CandidateType.SESSION_SPINE: 6,
    CandidateType.EPISODIC_MEMORY: 7,
    CandidateType.SEMANTIC_MEMORY: 8,
    CandidateType.PROCEDURAL_SKILL: 9,
    CandidateType.BACKGROUND_CONTEXT: 10,
}


@dataclass(frozen=True)
class InjectionCandidate:
    """Normalized candidate considered by the injection arbitrator."""

    key: str
    content: str
    type: CandidateType
    source: str
    confidence: float | None = None
    freshness: float = 0.0
    relevance: float = 0.0
    priority: int | None = None
    risk_flags: tuple[RiskFlag | str, ...] = ()
    token_cost: int | None = None
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not self.key:
            raise ValueError("candidate key is required")
        if self.content is None:
            raise ValueError("candidate content is required")
        object.__setattr__(self, "type", CandidateType(self.type))
        object.__setattr__(self, "source", str(self.source))
        object.__setattr__(self, "freshness", _clamp(self.freshness))
        object.__setattr__(self, "relevance", _clamp(self.relevance))
        if self.confidence is not None:
            object.__setattr__(self, "confidence", _clamp(self.confidence))
        if self.priority is None:
            object.__setattr__(self, "priority", INJECTION_PRIORITY[self.type])
        if self.token_cost is None:
            object.__setattr__(self, "token_cost", estimate_token_cost(self.content))
        object.__setattr__(
            self,
            "risk_flags",
            tuple(_normalize_risk_flag(flag) for flag in self.risk_flags),
        )

    @property
    def rank_score(self) -> float:
        """Stable tie-break score inside a fixed budget bucket."""

        confidence = 0.5 if self.confidence is None else self.confidence
        return (self.relevance * 0.45) + (self.freshness * 0.30) + (confidence * 0.25)

    def with_risk(self, *flags: RiskFlag | str) -> "InjectionCandidate":
        merged = tuple(dict.fromkeys((*self.risk_flags, *flags)))
        return replace(self, risk_flags=merged)

    def with_content(self, content: str) -> "InjectionCandidate":
        return replace(self, content=content, token_cost=estimate_token_cost(content))

    def to_audit_dict(self) -> dict[str, Any]:
        return {
            "key": self.key,
            "type": self.type.value,
            "source": self.source,
            "confidence": self.confidence,
            "freshness": self.freshness,
            "relevance": self.relevance,
            "priority": self.priority,
            "risk_flags": [str(flag) for flag in self.risk_flags],
            "token_cost": self.token_cost,
            "metadata": dict(self.metadata),
        }


def normalize_candidate(candidate: InjectionCandidate | Mapping[str, Any]) -> InjectionCandidate:
    if isinstance(candidate, InjectionCandidate):
        return candidate
    return InjectionCandidate(**dict(candidate))


def estimate_token_cost(content: str) -> int:
    stripped = str(content).strip()
    if not stripped:
        return 0
    return max(1, (len(stripped) + 3) // 4)


def _clamp(value: float) -> float:
    return max(0.0, min(1.0, float(value)))


def _normalize_risk_flag(flag: RiskFlag | str) -> RiskFlag | str:
    try:
        return RiskFlag(flag)
    except ValueError:
        return str(flag)
