"""Shadow comparison for transition authority migration (Phase 5.3).

Runs both legacy TransitionManager and new TransitionTable in parallel,
logs disagreements, but keeps legacy as authority.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from .types import ModeDecision, TransitionDecision
    from .transition_policy import TransitionRule

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class TransitionComparison:
    """Record of a single transition decision comparison."""

    # Input context
    previous_mode: str | None
    nominated_mode: str
    confidence: float

    # Legacy result
    legacy_active_mode: str
    legacy_transition_label: str
    legacy_reason: str

    # New table result
    new_active_mode: str | None
    new_transition_label: str | None
    new_reason: str | None
    new_rule_id: str | None

    # Comparison
    agreement: bool
    disagreement_type: str | None  # "mode", "label", "no_match", "both_agree"

    # Context for analysis (opaque)
    case_context_hash: str | None = None


class TransitionShadowComparator:
    """Compare legacy and new transition logic without changing production behavior."""

    def __init__(self, enable_logging: bool = True):
        self.enable_logging = enable_logging
        self.comparisons: list[TransitionComparison] = []
        self._disagreement_count = 0
        self._total_count = 0

    def compare(
        self,
        previous_mode: str | None,
        nominated_mode: str,
        confidence: float,
        legacy_decision: TransitionDecision,
        new_rule: TransitionRule | None,
        case_context_hash: str | None = None,
    ) -> TransitionComparison:
        """Compare one decision and record the result."""

        self._total_count += 1

        # Extract new decision
        if new_rule is None:
            new_active = None
            new_label = None
            new_reason = None
            new_rule_id = None
            disagreement_type = "no_match"
            agreement = False
        else:
            new_active = new_rule.active_mode
            new_label = new_rule.transition_label
            new_reason = new_rule.reason
            new_rule_id = new_rule.rule_id

            # Check agreement
            if legacy_decision.active_mode == new_active:
                agreement = True
                disagreement_type = None
            else:
                agreement = False
                disagreement_type = "mode"

        if not agreement:
            self._disagreement_count += 1

        comparison = TransitionComparison(
            previous_mode=previous_mode,
            nominated_mode=nominated_mode,
            confidence=confidence,
            legacy_active_mode=legacy_decision.active_mode,
            legacy_transition_label=legacy_decision.transition,
            legacy_reason=legacy_decision.reason,
            new_active_mode=new_active,
            new_transition_label=new_label,
            new_reason=new_reason,
            new_rule_id=new_rule_id,
            agreement=agreement,
            disagreement_type=disagreement_type,
            case_context_hash=case_context_hash,
        )

        self.comparisons.append(comparison)

        if self.enable_logging and not agreement:
            logger.warning(
                f"Transition disagreement: legacy={legacy_decision.active_mode} "
                f"new={new_active} prev={previous_mode} nom={nominated_mode} "
                f"conf={confidence:.2f} rule={new_rule_id}"
            )

        return comparison

    def get_agreement_rate(self) -> float:
        """Get the proportion of decisions that agreed."""
        if self._total_count == 0:
            return 1.0
        return (self._total_count - self._disagreement_count) / self._total_count

    def get_disagreements(self) -> list[TransitionComparison]:
        """Get all disagreements for analysis."""
        return [c for c in self.comparisons if not c.agreement]

    def get_all_comparisons(self) -> list[TransitionComparison]:
        """Get all comparisons for detailed analysis."""
        return self.comparisons

    def get_summary(self) -> dict:
        """Get summary statistics."""
        disagreements = self.get_disagreements()

        disagreement_by_type = {}
        for d in disagreements:
            disagreement_by_type[d.disagreement_type or "unknown"] = \
                disagreement_by_type.get(d.disagreement_type or "unknown", 0) + 1

        return {
            "total_decisions": self._total_count,
            "total_comparisons": self._total_count,  # Alias for compatibility
            "agreements": self._total_count - self._disagreement_count,
            "disagreements": self._disagreement_count,
            "agreement_rate": self.get_agreement_rate(),
            "disagreement_by_type": disagreement_by_type,
        }

    def reset(self):
        """Clear recorded comparisons."""
        self.comparisons = []
        self._disagreement_count = 0
        self._total_count = 0
