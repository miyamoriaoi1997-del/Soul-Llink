"""Type definitions for the context-aware multi-turn router.

All types are stdlib-only and abstract. No raw text, no keyword lists,
no production-specific imports.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any


class IntegrationLevel(str, Enum):
    """How deeply the context router participates in final decisions."""

    SHADOW_ONLY = "shadow_only"
    RECORD_ADVICE = "record_advice"
    CANDIDATE_WRITE = "candidate_write"
    GUARD_OVERRIDE = "guard_override"
    DIRECT_SWITCH = "direct_switch"


@dataclass
class ContextRouterConfig:
    """Runtime configuration for the context router."""

    enabled: bool = True
    integration: IntegrationLevel = IntegrationLevel.SHADOW_ONLY
    high_confidence_threshold: float = 0.72
    shadow_confidence_threshold: float = 0.42
    min_turns_between_switches: int = 2
    report_enabled: bool = True
    log_path: str = ""


@dataclass
class ContextSignals:
    """Abstract semantic signals extracted from a turn and its context."""

    meta_debug: bool = False
    quoted_or_hypothetical: bool = False
    explicit_task_request: bool = False
    work_intent: bool = False
    route_inspection_intent: bool = False
    relationship_context: bool = False
    continuation: bool = False
    scene_progression: bool = False
    boundary_signal: bool = False
    marker_group: str | None = None
    marker_in_task_context: bool = False


@dataclass
class BeliefScores:
    """Probabilistic belief over possible modes (not hard classification)."""

    work: float = 0.0
    daily: float = 0.0
    affectionate: float = 0.0
    intimacy_candidate: float = 0.0
    confirmed_intimacy: float = 0.0

    def top_mode(self) -> str:
        """Return the mode with highest belief score."""
        scores = {
            "work": self.work,
            "daily": self.daily,
            "affectionate": self.affectionate,
            "intimacy_candidate": self.intimacy_candidate,
            "confirmed_intimacy": self.confirmed_intimacy,
        }
        return max(scores, key=scores.get)  # type: ignore[arg-type]

    def top_score(self) -> float:
        return max(
            self.work, self.daily, self.affectionate,
            self.intimacy_candidate, self.confirmed_intimacy,
        )


@dataclass
class ContextRouteResult:
    """Complete routing decision from the context router."""

    top_mode: str
    work_submode: str | None = None
    relationship_submode: str | None = None
    referenced_mode: str | None = None
    secondary_candidate: str | None = None
    transition_type: str = "hold"
    confidence: float = 0.0
    reasons: list[str] = field(default_factory=list)
    signals: ContextSignals | None = None
    belief: BeliefScores | None = None

    def to_dict(self) -> dict[str, Any]:
        """Serialize to a flat dict for logging and shadow reports."""
        return {
            "routing": {
                "top_mode": self.top_mode,
                "work_submode": self.work_submode,
                "relationship_submode": self.relationship_submode,
                "referenced_mode": self.referenced_mode,
                "secondary_candidate": self.secondary_candidate,
                "transition_type": self.transition_type,
                "confidence": self.confidence,
            },
            "reasons": list(self.reasons),
            "signals": vars(self.signals) if self.signals else {},
            "belief": vars(self.belief) if self.belief else {},
        }


@dataclass
class AbstractTurn:
    """A single turn represented as abstract tags only."""

    abstract: str = ""
    tags: list[str] = field(default_factory=list)
    role: str = "user"


@dataclass
class AbstractInput:
    """Complete abstract input for the context router."""

    current_turn: AbstractTurn = field(default_factory=AbstractTurn)
    recent_history: list[AbstractTurn] = field(default_factory=list)
    previous_state: dict[str, Any] = field(default_factory=dict)
    affective_state: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "current_turn": {"abstract": self.current_turn.abstract, "tags": self.current_turn.tags},
            "recent_history": [
                {"role": t.role, "abstract": t.abstract, "tags": t.tags}
                for t in self.recent_history
            ],
            "previous_state": dict(self.previous_state),
            "affective_state": dict(self.affective_state),
        }


__all__ = [
    "IntegrationLevel",
    "ContextRouterConfig",
    "ContextSignals",
    "BeliefScores",
    "ContextRouteResult",
    "AbstractTurn",
    "AbstractInput",
]
