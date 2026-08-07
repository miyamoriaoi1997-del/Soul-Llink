"""Enhanced TransitionManager with shadow transition table support (Phase 5.3).

This module extends the legacy TransitionManager with optional shadow comparison
to the new data-driven transition table. The legacy path remains authoritative
until explicit migration approval.

Phase 5.4: Adds bounded activation for low-risk pairs only.
"""

from __future__ import annotations

import logging
import os
from typing import TYPE_CHECKING

from .transition_manager import TransitionManager as LegacyTransitionManager
from .transition_policy import (
    ContinuationTarget,
    ForceEvent,
    GateResult,
    TransitionTable,
    build_legacy_transition_table,
    apply_legacy_conditions,
)
from .transition_shadow import TransitionShadowComparator

if TYPE_CHECKING:
    from .types import ModeDecision, TransitionDecision

logger = logging.getLogger(__name__)

# Feature gate: Set SOULLINK_ENABLE_TRANSITION_TABLE_SHADOW=1 to enable comparison
_ENABLE_SHADOW = os.environ.get("SOULLINK_ENABLE_TRANSITION_TABLE_SHADOW", "0") == "1"

# Feature gate: Set SOULLINK_ENABLE_BOUNDED_ACTIVATION=1 to enable new table for low-risk pairs
_ENABLE_BOUNDED_ACTIVATION = os.environ.get("SOULLINK_ENABLE_BOUNDED_ACTIVATION", "0") == "1"

# Low-risk mode pairs that can use new table when bounded activation is enabled
_LOW_RISK_PAIRS = {
    ("daily", "work"),
    ("work", "daily"),
    ("daily", "daily"),  # Hold daily
    ("work", "work"),    # Hold work
}


class TransitionManagerV2:
    """TransitionManager with optional shadow transition table.

    Phase 5.3 behavior:
    - Legacy TransitionManager is always authoritative
    - New TransitionTable runs in shadow mode when enabled
    - Comparisons logged for analysis

    Phase 5.4 behavior (bounded activation):
    - New table takes authority for low-risk pairs (Daily<->Work) only
    - Protected/high-risk pairs (sex mode, crisis) remain legacy authority
    - Explicit opt-in required: SOULLINK_ENABLE_BOUNDED_ACTIVATION=1
    - Falls back to legacy on any failure

    Feature gates:
    - SOULLINK_ENABLE_TRANSITION_TABLE_SHADOW=1 (shadow comparison)
    - SOULLINK_ENABLE_BOUNDED_ACTIVATION=1 (bounded activation)
    """

    def __init__(
        self,
        enable_shadow_table: bool | None = None,
        enable_shadow_logging: bool = True,
        enable_bounded_activation: bool | None = None,
    ):
        """Initialize with legacy manager and optional shadow table.

        Args:
            enable_shadow_table: Override environment variable for shadow mode.
                                If None, uses SOULLINK_ENABLE_TRANSITION_TABLE_SHADOW.
            enable_shadow_logging: Enable logging of disagreements.
            enable_bounded_activation: Override environment variable for bounded activation.
                                      If None, uses SOULLINK_ENABLE_BOUNDED_ACTIVATION.
        """
        # Legacy manager is always active
        self.legacy_manager = LegacyTransitionManager()

        # Shadow table is opt-in
        self.enable_shadow = enable_shadow_table if enable_shadow_table is not None else _ENABLE_SHADOW

        # Bounded activation is opt-in and requires shadow table
        # Read from environment at runtime to allow test patching
        if enable_bounded_activation is not None:
            self.enable_bounded_activation = enable_bounded_activation
        else:
            self.enable_bounded_activation = os.environ.get("SOULLINK_ENABLE_BOUNDED_ACTIVATION", "0") == "1"

        # Bounded activation requires shadow table to be meaningful
        if self.enable_bounded_activation and not self.enable_shadow:
            logger.warning("Bounded activation requested but shadow table disabled; forcing shadow table ON")
            self.enable_shadow = True

        if self.enable_shadow:
            self.shadow_table = build_legacy_transition_table()
            self.shadow_comparator = TransitionShadowComparator(enable_logging=enable_shadow_logging)
        else:
            self.shadow_table = None
            self.shadow_comparator = None

    def transition(
        self,
        previous_mode: str | None,
        decision: ModeDecision,
        safety_flags: list[str] | None = None,
        desire_tier: str | None = None,
        enable_active_sex: bool = False,
        emotion_score: float | None = None,
    ) -> TransitionDecision:
        """Apply transition rules with optional bounded activation.

        Phase 5.3: Legacy always authoritative, shadow comparison only.
        Phase 5.4: Bounded activation for low-risk pairs if explicitly enabled.

        Returns:
            TransitionDecision with authority_source attribute indicating
            which path made the decision.
        """
        decision_safety_flags = list(getattr(decision, "safety_flags", None) or [])
        effective_safety_flags = list(safety_flags or [])
        for flag in decision_safety_flags:
            if flag not in effective_safety_flags:
                effective_safety_flags.append(flag)

        # Always execute legacy path first
        legacy_result = self.legacy_manager.transition(
            previous_mode=previous_mode,
            decision=decision,
            safety_flags=effective_safety_flags,
            desire_tier=desire_tier,
            enable_active_sex=enable_active_sex,
            emotion_score=emotion_score,
        )

        # Determine if we can use bounded activation
        use_new_table = self._should_use_new_table(
            previous_mode=previous_mode,
            nominated_mode=decision.mode,
            safety_flags=effective_safety_flags,
            enable_active_sex=enable_active_sex,
        )
        # The legacy manager owns the new bounded semantic-release signal until
        # the transition table has an equivalent typed condition. Letting the
        # generic WORK.HOLD rule consume it would reintroduce work stickiness.
        if decision.signals.get("semantic_explicit_daily_intent"):
            use_new_table = False

        # Try new table for low-risk pairs if bounded activation enabled
        if use_new_table and self.enable_bounded_activation and self.shadow_table:
            try:
                new_result = self._execute_new_table(
                    previous_mode=previous_mode,
                    decision=decision,
                    safety_flags=effective_safety_flags,
                    desire_tier=desire_tier,
                    enable_active_sex=enable_active_sex,
                )

                # Validate new table result
                if new_result and self._is_valid_result(new_result):
                    # Add authority source
                    new_result.authority_source = "new_table"

                    # Still record comparison if shadow enabled
                    if self.shadow_comparator:
                        self._record_comparison_from_results(
                            previous_mode=previous_mode,
                            decision=decision,
                            legacy_result=legacy_result,
                            new_result=new_result,
                        )

                    return new_result
                else:
                    logger.warning("Invalid result from new table, falling back to legacy")

            except Exception as e:
                logger.exception(f"Exception in new table execution, falling back to legacy: {e}")

        # Shadow comparison if enabled and not already done
        if self.enable_shadow and self.shadow_table and self.shadow_comparator:
            if not (use_new_table and self.enable_bounded_activation):
                self._shadow_compare(
                    previous_mode=previous_mode,
                    decision=decision,
                    legacy_result=legacy_result,
                    safety_flags=effective_safety_flags,
                    desire_tier=desire_tier,
                    enable_active_sex=enable_active_sex,
                )

        # Return legacy result with authority marker
        legacy_result.authority_source = "legacy"
        return legacy_result

    def _should_use_new_table(
        self,
        previous_mode: str | None,
        nominated_mode: str,
        safety_flags: list[str] | None,
        enable_active_sex: bool,
    ) -> bool:
        """Determine if new table can be used for this transition.

        Returns True only for low-risk pairs with no high-risk conditions.
        """
        # High-risk condition: sex mode involvement
        if nominated_mode == "sex" or previous_mode == "sex":
            return False

        # High-risk condition: crisis events
        if safety_flags and "crisis_guard" in safety_flags:
            return False

        # Same-mode "transitions" (holds) involve complex continuation logic
        # Keep these with legacy for now
        if previous_mode == nominated_mode and previous_mode is not None:
            return False

        # Check if this is a low-risk pair
        pair = (previous_mode or "none", nominated_mode)
        if pair not in _LOW_RISK_PAIRS:
            # Also allow none->daily/work as low-risk (initial state)
            if previous_mode is None and nominated_mode in {"daily", "work"}:
                return True
            return False

        return True

    def _execute_new_table(
        self,
        previous_mode: str | None,
        decision: ModeDecision,
        safety_flags: list[str],
        desire_tier: str | None,
        enable_active_sex: bool,
    ) -> TransitionDecision:
        """Execute new table to get transition decision.

        Returns a TransitionDecision object matching legacy format.
        """
        # Map decision signals to new table inputs
        force_event = self._map_force_event(decision, previous_mode, None, safety_flags)
        continuation_target = self._map_continuation_target(decision)
        gate_result = self._map_gate_result(decision, desire_tier, enable_active_sex)

        # Query new table
        new_rule = self.shadow_table.decide(
            previous_mode=previous_mode,
            nominated_mode=decision.mode,
            force_event=force_event,
            continuation_target=continuation_target,
            gate_result=gate_result,
            submode=decision.submode,
            confidence=decision.confidence,
        )

        if not new_rule:
            return None

        # Apply legacy behavioral conditions to refine the rule
        new_rule = apply_legacy_conditions(
            rule=new_rule,
            previous_mode=previous_mode,
            nominated_mode=decision.mode,
            confidence=decision.confidence,
            is_short_message=self._is_short_message(decision),
            is_explicit_task=self._is_explicit_task(decision),
            is_machine_status=self._is_machine_status(decision),
            is_context_continuation=self._is_context_continuation(decision),
        )

        if not new_rule:
            return None

        # Convert rule to TransitionDecision format
        # Import here to avoid circular dependency
        from .types import TransitionDecision

        result = TransitionDecision(
            previous_mode=previous_mode,
            requested_mode=decision.mode,
            active_mode=new_rule.active_mode,
            transition=new_rule.transition_label,
            confidence=decision.confidence,
            reason=new_rule.reason,
            safety_flags=safety_flags,
        )

        return result

    def _is_valid_result(self, result: TransitionDecision) -> bool:
        """Validate that a transition result is well-formed."""
        if not result:
            return False
        if not hasattr(result, 'active_mode'):
            return False
        if result.active_mode not in {"daily", "work", "sex"}:
            return False
        return True

    def _record_comparison_from_results(
        self,
        previous_mode: str | None,
        decision: ModeDecision,
        legacy_result: TransitionDecision,
        new_result: TransitionDecision,
    ):
        """Record bounded-path comparison without manufacturing disagreement.

        The comparator consumes a transition-rule-shaped object, while the
        bounded path already has a validated TransitionDecision. Adapt that
        result explicitly so observability reflects the decision actually
        returned by the new table.
        """
        from types import SimpleNamespace

        new_rule = SimpleNamespace(
            active_mode=new_result.active_mode,
            transition_label=new_result.transition,
            reason=new_result.reason,
            rule_id="bounded_runtime_result",
        )
        self.shadow_comparator.compare(
            previous_mode=previous_mode,
            nominated_mode=decision.mode,
            confidence=decision.confidence,
            legacy_decision=legacy_result,
            new_rule=new_rule,
        )

    def _shadow_compare(
        self,
        previous_mode: str | None,
        decision: ModeDecision,
        legacy_result: TransitionDecision,
        safety_flags: list[str],
        desire_tier: str | None,
        enable_active_sex: bool,
    ):
        """Run shadow table and compare with legacy result."""

        # Map decision signals to new table inputs
        force_event = self._map_force_event(decision, previous_mode, legacy_result, safety_flags)
        continuation_target = self._map_continuation_target(decision)
        gate_result = self._map_gate_result(decision, desire_tier, enable_active_sex)

        # Query shadow table
        shadow_rule = self.shadow_table.decide(
            previous_mode=previous_mode,
            nominated_mode=decision.mode,
            force_event=force_event,
            continuation_target=continuation_target,
            gate_result=gate_result,
            submode=decision.submode,
            confidence=decision.confidence,
        )

        # Apply legacy behavioral conditions to refine the rule
        if shadow_rule:
            shadow_rule = apply_legacy_conditions(
                rule=shadow_rule,
                previous_mode=previous_mode,
                nominated_mode=decision.mode,
                confidence=decision.confidence,
                is_short_message=self._is_short_message(decision),
                is_explicit_task=self._is_explicit_task(decision),
                is_machine_status=self._is_machine_status(decision),
                is_context_continuation=self._is_context_continuation(decision),
            )

        # Record comparison
        self.shadow_comparator.compare(
            previous_mode=previous_mode,
            nominated_mode=decision.mode,
            confidence=decision.confidence,
            legacy_decision=legacy_result,
            new_rule=shadow_rule,
        )

    def _map_force_event(
        self,
        decision: ModeDecision,
        previous_mode: str | None,
        legacy_result: TransitionDecision,
        safety_flags: list[str] | None = None,
    ) -> ForceEvent:
        """Map decision signals to ForceEvent enum."""

        # Crisis guard
        effective_safety_flags = safety_flags if safety_flags is not None else (decision.safety_flags or [])
        if "crisis_guard" in effective_safety_flags:
            return ForceEvent.CRISIS

        # Explicit task/system request
        if decision.signals.get("explicit_task_request") or decision.signals.get("explicit_system_request"):
            return ForceEvent.EXPLICIT_TASK

        # Explicit progression (for protected mode)
        if decision.submode == "explicit_progression":
            return ForceEvent.EXPLICIT_PROGRESSION

        # Scene close
        if decision.signals.get("sex_scene_close"):
            return ForceEvent.SCENE_CLOSE

        return ForceEvent.NONE

    def _map_continuation_target(self, decision: ModeDecision) -> ContinuationTarget:
        """Map decision signals to ContinuationTarget enum."""

        # Sex scene continuation
        if decision.signals.get("sex_scene_continue"):
            return ContinuationTarget.SCENE

        # Work context continuation (detected by legacy manager helper)
        text = str(decision.signals.get("normalized_text") or "")
        if self.legacy_manager._is_work_continuation_exit_message(decision):
            return ContinuationTarget.WORK

        # Context continuation (could be work or relationship)
        if self.legacy_manager._is_context_continuation_message(decision):
            # Distinguish based on signal type
            if self.legacy_manager._is_context_action_message(decision):
                return ContinuationTarget.WORK
            elif self.legacy_manager._is_context_question_message(decision):
                return ContinuationTarget.WORK
            else:
                return ContinuationTarget.AMBIGUOUS

        return ContinuationTarget.NONE

    def _map_gate_result(
        self,
        decision: ModeDecision,
        desire_tier: str | None,
        enable_active_sex: bool,
    ) -> GateResult:
        """Map desire gate status to GateResult enum."""

        # Only relevant for sex mode nominations
        if decision.mode != "sex":
            return GateResult.NOT_APPLICABLE

        # Check gate conditions (matching legacy logic)
        if enable_active_sex and desire_tier in {"uninhibited", "ambivalent"}:
            return GateResult.PASS
        else:
            return GateResult.FAIL

    def _is_short_message(self, decision: ModeDecision) -> bool:
        """Check if message is short (<=4 chars)."""
        message_length = decision.signals.get("message_length")
        try:
            return int(message_length) <= 4
        except (TypeError, ValueError):
            return False

    def _is_explicit_task(self, decision: ModeDecision) -> bool:
        """Check if this is an explicit task request."""
        return bool(
            decision.signals.get("explicit_task_request")
            or decision.signals.get("explicit_system_request")
        )

    def _is_machine_status(self, decision: ModeDecision) -> bool:
        """Check if this is machine status output."""
        return self.legacy_manager._is_machine_status_output(decision)

    def _is_context_continuation(self, decision: ModeDecision) -> bool:
        """Check if this is a context continuation message."""
        return self.legacy_manager._is_context_continuation_message(decision)

    def get_shadow_summary(self) -> dict | None:
        """Get shadow comparison summary statistics.

        Returns None if shadow mode is not enabled.
        """
        if self.shadow_comparator:
            return self.shadow_comparator.get_summary()
        return None

    def get_shadow_disagreements(self):
        """Get list of all shadow disagreements for analysis."""
        if self.shadow_comparator:
            return self.shadow_comparator.get_disagreements()
        return []

    def reset_shadow_stats(self):
        """Reset shadow comparison statistics."""
        if self.shadow_comparator:
            self.shadow_comparator.reset()


# Backward compatibility: default export is V2 with shadow capability
TransitionManager = TransitionManagerV2
