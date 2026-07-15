"""Core context-aware router: signals → belief → policy → transition.

Integrated into persona_engine's StateOrchestrator pipeline after the
standalone context-router scaffold was retired.

No raw keyword lists. No production-specific text matching.
Inputs must be abstract tags and labels only.
"""

from __future__ import annotations

from typing import Any

from .types import (
    BeliefScores,
    ContextRouteResult,
    ContextRouterConfig,
    ContextSignals,
    IntegrationLevel,
)


class ContextRouter:
    """Four-layer context-aware router: extract → compute → gate → transition."""

    def __init__(self, config: ContextRouterConfig | None = None):
        self.config = config or ContextRouterConfig()

    # ─── Public API ───────────────────────────────────────────────────────

    def analyze(self, case_input: dict[str, Any]) -> ContextRouteResult:
        """Full pipeline: extract signals, compute belief, apply policy."""
        signals = self.extract_signals(case_input)
        belief = self.compute_belief(signals, case_input)
        return self.apply_policy(signals, belief, case_input)

    # ─── Layer 1: Signal Extraction ───────────────────────────────────────

    def extract_signals(self, case_input: dict[str, Any]) -> ContextSignals:
        """Extract abstract semantic signals from tags and context."""
        current = case_input.get("current_turn") or {}
        tags = self._tags(current)
        hist = case_input.get("recent_history") or []
        hist_tags: set[str] = set()
        for item in hist:
            hist_tags |= self._tags(item)

        previous = case_input.get("previous_state") or {}
        abstract = self._abstract(current)

        sig = ContextSignals()
        sig.meta_debug = "task_debug" in tags
        sig.quoted_or_hypothetical = "quoted_context" in tags
        sig.explicit_task_request = "task_request" in tags or "task_debug" in tags
        sig.route_inspection_intent = (
            "task_debug" in tags
            or abstract == "ROUTE_DEBUG_WITH_REFERENCED_SCENE"
        )
        sig.work_intent = sig.explicit_task_request or sig.route_inspection_intent
        sig.relationship_context = (
            "relationship_context" in tags
            or "relationship_context" in hist_tags
        )
        sig.continuation = (
            "continuation_cue" in tags
            and sig.relationship_context
            and not sig.work_intent
        )
        sig.scene_progression = (
            "progression_setup" in tags
            or (
                "relationship_marker_medium" in tags
                and sig.relationship_context
                and sig.continuation
                and not sig.work_intent
            )
            or (
                "relationship_marker_strong" in tags
                and sig.relationship_context
                and not sig.quoted_or_hypothetical
                and not sig.work_intent
            )
        )
        sig.boundary_signal = "boundary_signal" in tags or "safety_stop" in tags
        sig.marker_group = self._marker_group(tags)
        sig.marker_in_task_context = (
            sig.marker_group is not None
            and sig.marker_group != "ABSTRACT_TASK_MARKER"
            and (sig.work_intent or sig.meta_debug or sig.quoted_or_hypothetical)
        )
        return sig

    # ─── Layer 2: Belief Computation ──────────────────────────────────────

    def compute_belief(self, sig: ContextSignals, case_input: dict[str, Any]) -> BeliefScores:
        """Compute probabilistic belief scores over modes."""
        b = BeliefScores()
        affective = case_input.get("affective_state") or {}
        score_band = affective.get("score_band", "neutral")
        desire_band = affective.get("desire_band", "low")

        # Work belief
        if sig.work_intent:
            b.work = 0.85
        if sig.meta_debug:
            b.work = max(b.work, 0.88)
        if sig.explicit_task_request:
            b.work = max(b.work, 0.82)
        if sig.quoted_or_hypothetical and sig.marker_in_task_context:
            b.work = max(b.work, 0.80)

        # Daily belief
        if sig.relationship_context and not sig.work_intent:
            b.daily = 0.55
        if sig.continuation and not sig.scene_progression:
            b.daily = max(b.daily, 0.60)

        # Affectionate belief
        if sig.relationship_context and sig.continuation and not sig.work_intent:
            b.affectionate = 0.50
            if score_band in ("warm", "high"):
                b.affectionate += 0.15

        # Intimacy candidate belief
        if sig.scene_progression and not sig.work_intent and not sig.boundary_signal:
            b.intimacy_candidate = 0.55
            if desire_band in ("ambivalent", "elevated"):
                b.intimacy_candidate += 0.15
            if score_band == "high":
                b.intimacy_candidate += 0.10

        # Confirmed intimacy belief
        if (
            sig.scene_progression
            and sig.relationship_context
            and not sig.work_intent
            and not sig.boundary_signal
            and not sig.quoted_or_hypothetical
        ):
            if desire_band == "elevated":
                b.confirmed_intimacy = 0.75
            elif desire_band == "ambivalent" and score_band in ("warm", "high"):
                b.confirmed_intimacy = 0.65
            elif sig.marker_group == "ABSTRACT_RELATIONSHIP_MARKER_STRONG":
                b.confirmed_intimacy = 0.60

        # Boundary override
        if sig.boundary_signal:
            b.work = 0.0
            b.intimacy_candidate = 0.0
            b.confirmed_intimacy = 0.0
            b.daily = 0.90

        return b

    # ─── Layer 3: Policy Gate ─────────────────────────────────────────────

    def apply_policy(
        self, sig: ContextSignals, b: BeliefScores, case_input: dict[str, Any]
    ) -> ContextRouteResult:
        """Apply gate rules: boundary > work > anti-flap > progression."""
        previous = case_input.get("previous_state") or {}
        turns_since = previous.get("turns_since_last_switch", 99)
        high_conf = self.config.high_confidence_threshold
        shadow_conf = self.config.shadow_confidence_threshold

        # Gate 1: Boundary always wins
        if sig.boundary_signal:
            return ContextRouteResult(
                top_mode="relationship",
                relationship_submode="cooldown",
                transition_type="downgrade",
                confidence=0.95,
                reasons=["signal:boundary_signal", "gate:boundary_override"],
                signals=sig,
                belief=b,
            )

        # Gate 2: Work intent overrides relationship
        if sig.work_intent and b.work >= high_conf:
            submode = "route_inspection" if sig.route_inspection_intent else None
            if not submode and sig.explicit_task_request:
                submode = "code_change"
            ref_mode = None
            if sig.marker_in_task_context:
                ref_mode = "intimacy_candidate" if sig.scene_progression else "affectionate"
            return ContextRouteResult(
                top_mode="work",
                work_submode=submode,
                referenced_mode=ref_mode,
                transition_type="switch" if previous.get("top_mode") != "work" else "hold",
                confidence=b.work,
                reasons=self._work_reasons(sig),
                signals=sig,
                belief=b,
            )

        # Gate 3: Confirmed intimacy with desire gate
        if b.confirmed_intimacy >= high_conf:
            if turns_since < self.config.min_turns_between_switches:
                return ContextRouteResult(
                    top_mode="relationship",
                    relationship_submode=previous.get("submode") or "affectionate",
                    secondary_candidate="confirmed_intimacy",
                    transition_type="hold",
                    confidence=b.confirmed_intimacy,
                    reasons=[
                        "signal:scene_progression",
                        "transition:anti_flapping",
                        "transition:secondary_candidate",
                    ],
                    signals=sig,
                    belief=b,
                )
            return ContextRouteResult(
                top_mode="relationship",
                relationship_submode="confirmed_intimacy",
                transition_type="switch",
                confidence=b.confirmed_intimacy,
                reasons=[
                    "signal:scene_progression",
                    "signal:relationship_context",
                    "gate:desire_allowed",
                ],
                signals=sig,
                belief=b,
            )

        # Gate 4: Intimacy candidate (shadow only, no switch)
        if b.intimacy_candidate >= shadow_conf and sig.relationship_context:
            return ContextRouteResult(
                top_mode="relationship",
                relationship_submode=previous.get("submode") or "affectionate",
                secondary_candidate="intimacy_candidate",
                transition_type="hold",
                confidence=b.intimacy_candidate,
                reasons=["transition:secondary_candidate", "gate:hold"],
                signals=sig,
                belief=b,
            )

        # Gate 5: Affectionate continuation
        if b.affectionate >= shadow_conf and sig.continuation:
            return ContextRouteResult(
                top_mode="relationship",
                relationship_submode="affectionate",
                transition_type="hold" if previous.get("top_mode") == "relationship" else "switch",
                confidence=b.affectionate,
                reasons=["signal:continuation", "belief:affectionate"],
                signals=sig,
                belief=b,
            )

        # Default: relationship/daily hold
        return ContextRouteResult(
            top_mode="relationship",
            relationship_submode="daily",
            transition_type="hold",
            confidence=max(b.daily, 0.5),
            reasons=["belief:low_context", "gate:hold"],
            signals=sig,
            belief=b,
        )

    # ─── Helpers ──────────────────────────────────────────────────────────

    @staticmethod
    def _tags(turn: dict[str, Any] | None) -> set[str]:
        if not turn:
            return set()
        return {str(x) for x in turn.get("tags", [])}

    @staticmethod
    def _abstract(turn: dict[str, Any] | None) -> str:
        if not turn:
            return ""
        return str(turn.get("abstract", ""))

    @staticmethod
    def _marker_group(tags: set[str]) -> str | None:
        mapping = {
            "relationship_marker_weak": "ABSTRACT_RELATIONSHIP_MARKER_WEAK",
            "relationship_marker_medium": "ABSTRACT_RELATIONSHIP_MARKER_MEDIUM",
            "relationship_marker_strong": "ABSTRACT_RELATIONSHIP_MARKER_STRONG",
            "task_marker": "ABSTRACT_TASK_MARKER",
            "boundary_marker": "ABSTRACT_BOUNDARY_MARKER",
        }
        for tag, group in mapping.items():
            if tag in tags:
                return group
        return None

    @staticmethod
    def _work_reasons(sig: ContextSignals) -> list[str]:
        reasons = []
        if sig.meta_debug:
            reasons.append("signal:meta_debug")
        if sig.explicit_task_request:
            reasons.append("signal:explicit_task_request")
        if sig.route_inspection_intent:
            reasons.append("signal:route_inspection_intent")
        if sig.quoted_or_hypothetical:
            reasons.append("signal:quoted_or_hypothetical")
        if sig.marker_in_task_context:
            reasons.append("signal:marker_in_task_context")
        reasons.append("gate:work_override")
        return reasons
