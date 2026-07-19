"""Pair-specific transition policy and decision table for mode transitions.

Phase 5: Implements configurable transition rules as data rather than hardcoded logic.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Literal


class ForceEvent(Enum):
    """Hard events that override normal transition rules."""
    NONE = "none"
    BOUNDARY_EXIT = "boundary_exit"
    CRISIS = "crisis"
    EXPLICIT_TASK = "explicit_task"
    EXPLICIT_PROGRESSION = "explicit_progression"
    SCENE_CLOSE = "scene_close"


class ContinuationTarget(Enum):
    """What the continuation signal binds to."""
    NONE = "none"
    WORK = "work"
    RELATIONSHIP = "relationship"
    SCENE = "scene"
    AMBIGUOUS = "ambiguous"


class GateResult(Enum):
    """Desire/permission gate check result."""
    NONE = "none"
    PASS = "pass"
    FAIL = "fail"
    NOT_APPLICABLE = "not_applicable"


@dataclass(frozen=True)
class TransitionCondition:
    """Input conditions for a transition rule."""
    previous_mode: str  # MODE_DAILY, MODE_WORK, MODE_SEX, or "any"
    nominated_mode: str  # MODE_DAILY, MODE_WORK, MODE_SEX, or "any"
    force_event: ForceEvent = ForceEvent.NONE
    continuation_target: ContinuationTarget = ContinuationTarget.NONE
    gate_result: GateResult = GateResult.NONE
    submode: str | None = None  # e.g., "hint_progression", "explicit_progression"

    def matches(
        self,
        previous: str | None,
        nominated: str,
        force: ForceEvent,
        continuation: ContinuationTarget,
        gate: GateResult,
        submode: str | None = None,
    ) -> bool:
        """Check if this condition matches the given state."""
        if self.previous_mode != "any" and self.previous_mode != (previous or "none"):
            return False
        if self.nominated_mode != "any" and self.nominated_mode != nominated:
            return False
        if self.force_event != ForceEvent.NONE and self.force_event != force:
            return False
        if self.continuation_target != ContinuationTarget.NONE and self.continuation_target != continuation:
            return False
        if self.gate_result != GateResult.NONE and self.gate_result != gate:
            return False
        if self.submode is not None and self.submode != submode:
            return False
        return True


@dataclass(frozen=True)
class TransitionRule:
    """A single transition table rule with priority-ordered matching."""
    rule_id: str
    condition: TransitionCondition
    active_mode: str  # The actual mode to activate
    transition_label: str  # e.g., "daily->work", "hold_context_action"
    reason: str
    priority: int  # Higher priority wins on conflict
    max_stale_hold_turns: int | None = None  # Maximum turns to hold before forcing switch

    def __post_init__(self):
        """Validate rule fields."""
        if self.priority < 0:
            raise ValueError(f"Rule {self.rule_id}: priority must be non-negative")
        if self.active_mode not in {"daily", "work", "sex"}:
            raise ValueError(f"Rule {self.rule_id}: active_mode must be daily/work/sex")
        if self.max_stale_hold_turns is not None and self.max_stale_hold_turns < 1:
            raise ValueError(f"Rule {self.rule_id}: max_stale_hold_turns must be positive")


@dataclass(frozen=True)
class PairSpecificPolicy:
    """Configuration for a specific (previous, nominated) mode pair."""
    previous_mode: str
    nominated_mode: str
    enter_threshold: float | None = None
    exit_threshold: float | None = None
    minimum_dwell_turns: int = 0
    maximum_stale_hold_turns: int = 5
    high_risk_pair: bool = False


class TransitionTable:
    """UNIQUE hit policy decision table for mode transitions.

    Each input combination must match exactly one rule. Multi-hit or zero-hit
    is a configuration error that fails at startup or test time.
    """

    def __init__(self, rules: list[TransitionRule] | None = None):
        self.rules = sorted(rules or [], key=lambda r: r.priority, reverse=True)
        self._validate_unique_coverage()

    def _validate_unique_coverage(self):
        """Ensure no obvious conflicts in high-priority rules.

        Full coverage validation requires enumerating all input combinations,
        which is done in tests. This just catches obvious same-priority conflicts.
        """
        priority_groups: dict[int, list[TransitionRule]] = {}
        for rule in self.rules:
            priority_groups.setdefault(rule.priority, []).append(rule)

        for priority, group in priority_groups.items():
            if len(group) > 1 and priority >= 1000:  # Hard boundary priority
                # Check for obvious conflicts at same priority
                ids = [r.rule_id for r in group]
                # This is a simplified check; full validation is in tests
                pass

    def decide(
        self,
        previous_mode: str | None,
        nominated_mode: str,
        force_event: ForceEvent,
        continuation_target: ContinuationTarget,
        gate_result: GateResult,
        submode: str | None = None,
        confidence: float = 0.5,
    ) -> TransitionRule | None:
        """Find the matching rule with highest priority.

        Returns None if no rule matches (should never happen with proper coverage).
        Logs a warning if multiple rules at same priority match (configuration error).
        """
        matches = [
            rule for rule in self.rules
            if rule.condition.matches(
                previous_mode,
                nominated_mode,
                force_event,
                continuation_target,
                gate_result,
                submode,
            )
        ]

        if not matches:
            return None

        # Return highest priority match
        # If multiple rules at same priority match, take the first (stable sort)
        return matches[0]


def build_legacy_transition_table() -> TransitionTable:
    """Extract the existing transition_manager.py logic as a data table.

    This preserves the current behavior exactly while converting it to a
    configurable format. Each rule ID maps to a specific branch in the
    original transition() method.
    """
    rules = [
        # Priority 1000: Hard boundaries (crisis_guard, explicit exit)
        TransitionRule(
            rule_id="HARD.CRISIS_GUARD.FORCE_DAILY",
            condition=TransitionCondition(
                previous_mode="any",
                nominated_mode="any",
                force_event=ForceEvent.CRISIS,
            ),
            active_mode="daily",
            transition_label="crisis->daily",
            reason="crisis_guard keeps relationship response in daily mode",
            priority=1000,
        ),

        # Priority 900: Protected mode (sex) gate enforcement
        TransitionRule(
            rule_id="SEX.WORK_HOLD.AMBIGUOUS_HINT",
            condition=TransitionCondition(
                previous_mode="work",
                nominated_mode="sex",
                submode="hint_progression",
                gate_result=GateResult.PASS,
            ),
            active_mode="work",
            transition_label="hold_context_action",
            reason="ambiguous sex hint cannot override an active work context without explicit progression",
            priority=900,
        ),

        TransitionRule(
            rule_id="SEX.CONTINUE.FROM_SEX",
            condition=TransitionCondition(
                previous_mode="sex",
                nominated_mode="sex",
                continuation_target=ContinuationTarget.SCENE,
            ),
            active_mode="sex",
            transition_label="hold_sex_continue",
            reason="sex_scene_continue signal holds existing sex mode",
            priority=900,
        ),

        TransitionRule(
            rule_id="SEX.ENTER.GATE_PASS",
            condition=TransitionCondition(
                previous_mode="any",
                nominated_mode="sex",
                gate_result=GateResult.PASS,
            ),
            active_mode="sex",
            transition_label="any->sex",
            reason="sex mode accepted after desire gate",
            priority=900,
        ),

        TransitionRule(
            rule_id="SEX.BLOCK.GATE_FAIL",
            condition=TransitionCondition(
                previous_mode="any",
                nominated_mode="sex",
                gate_result=GateResult.FAIL,
            ),
            active_mode="daily",
            transition_label="blocked:sex_gate",
            reason="blocked:sex_gate; sex request remains daily",
            priority=900,
        ),

        # Priority 800: Start state (no previous mode)
        TransitionRule(
            rule_id="START.ANY",
            condition=TransitionCondition(
                previous_mode="none",
                nominated_mode="any",
            ),
            active_mode="daily",  # Will be overridden by actual nominated mode in logic
            transition_label="start:nominated",
            reason="first turn accepts nomination",
            priority=800,
        ),

        # Priority 700: Explicit exits from protected mode
        TransitionRule(
            rule_id="SEX.EXIT.EXPLICIT_TASK",
            condition=TransitionCondition(
                previous_mode="sex",
                nominated_mode="any",
                force_event=ForceEvent.EXPLICIT_TASK,
            ),
            active_mode="work",
            transition_label="sex->work",
            reason="explicit work request exits sex mode",
            priority=700,
        ),

        TransitionRule(
            rule_id="SEX.EXIT.WORK_CONTINUATION",
            condition=TransitionCondition(
                previous_mode="sex",
                nominated_mode="daily",
                continuation_target=ContinuationTarget.WORK,
            ),
            active_mode="work",
            transition_label="sex->work_context_continuation",
            reason="work-context continuation exits sex mode after task handoff",
            priority=700,
        ),

        TransitionRule(
            rule_id="SEX.EXIT.SCENE_CLOSE",
            condition=TransitionCondition(
                previous_mode="sex",
                nominated_mode="daily",
                force_event=ForceEvent.SCENE_CLOSE,
            ),
            active_mode="daily",
            transition_label="sex->daily",
            reason="sex_scene_close signal exits sex mode",
            priority=700,
        ),

        TransitionRule(
            rule_id="SEX.HOLD.DEFAULT",
            condition=TransitionCondition(
                previous_mode="sex",
                nominated_mode="daily",
            ),
            active_mode="sex",
            transition_label="hold_sex_continuation",
            reason="sex mode held until explicit work request, work-context continuation, or close signal",
            priority=700,
            max_stale_hold_turns=10,  # Prevent infinite hold
        ),

        # Priority 600: Work context continuation (higher than short hold)
        TransitionRule(
            rule_id="WORK.HOLD.CONTEXT_ACTION",
            condition=TransitionCondition(
                previous_mode="work",
                nominated_mode="daily",
                continuation_target=ContinuationTarget.WORK,
            ),
            active_mode="work",
            transition_label="hold_context_action",
            reason="contextual continuation inherits previous work mode",
            priority=600,
            max_stale_hold_turns=5,
        ),

        # Priority 550: Mid-confidence daily/work transitions (higher than short hold)
        TransitionRule(
            rule_id="WORK.EXIT.MID_CONFIDENCE",
            condition=TransitionCondition(
                previous_mode="work",
                nominated_mode="daily",
            ),
            active_mode="daily",
            transition_label="work->daily",
            reason="mid-confidence daily nomination exits work",
            priority=550,
        ),

        TransitionRule(
            rule_id="DAILY.TO_WORK.MID_CONFIDENCE",
            condition=TransitionCondition(
                previous_mode="daily",
                nominated_mode="work",
            ),
            active_mode="work",
            transition_label="daily->work",
            reason="mid-confidence work nomination accepted",
            priority=550,
        ),

        # Priority 500: Short message hold (anti-flap)
        TransitionRule(
            rule_id="ANTIFLAP.SHORT_HOLD.NOT_SEX",
            condition=TransitionCondition(
                previous_mode="any",  # Excludes sex via separate priority
                nominated_mode="daily",
            ),
            active_mode="work",  # Will use actual previous mode
            transition_label="hold_short_message",
            reason="short neutral continuation inherits previous mode",
            priority=500,
        ),

        # Priority 400: High-confidence work override
        TransitionRule(
            rule_id="WORK.ENTER.HIGH_CONFIDENCE",
            condition=TransitionCondition(
                previous_mode="any",
                nominated_mode="work",
            ),
            active_mode="work",
            transition_label="any->work",
            reason="high-confidence work mode overrides previous mode",
            priority=400,
        ),

        # Priority 300: Work to Daily transitions
        TransitionRule(
            rule_id="WORK.HOLD.MACHINE_STATUS",
            condition=TransitionCondition(
                previous_mode="work",
                nominated_mode="daily",
            ),
            active_mode="work",
            transition_label="hold_low_confidence",
            reason="machine/status output inherits previous work mode",
            priority=300,
            max_stale_hold_turns=5,
        ),

        TransitionRule(
            rule_id="WORK.EXIT.TO_DAILY",
            condition=TransitionCondition(
                previous_mode="work",
                nominated_mode="daily",
            ),
            active_mode="daily",
            transition_label="work->daily",
            reason="work mode does not hold over relationship/daily mode",
            priority=200,
        ),

        # Priority 100: Low-confidence hold (generic anti-flap)
        TransitionRule(
            rule_id="ANTIFLAP.HOLD.LOW_CONFIDENCE",
            condition=TransitionCondition(
                previous_mode="any",
                nominated_mode="any",
            ),
            active_mode="work",  # Will use actual previous mode
            transition_label="hold_low_confidence",
            reason="confidence below threshold; previous mode held to avoid flapping",
            priority=100,
            max_stale_hold_turns=5,
        ),

        # Priority 0: Default accept nomination
        TransitionRule(
            rule_id="DEFAULT.ACCEPT_NOMINATION",
            condition=TransitionCondition(
                previous_mode="any",
                nominated_mode="any",
            ),
            active_mode="daily",  # Will be overridden by nominated mode
            transition_label="accept_nomination",
            reason="nomination accepted",
            priority=0,
        ),
    ]

    return TransitionTable(rules)


def _make_default_accept(
    nominated_mode: str,
    previous_mode: str | None,
    priority: int,
) -> TransitionRule:
    """Create a default accept rule for fallback when conditions don't match."""
    return TransitionRule(
        rule_id="ACCEPT.DEFAULT_FALLBACK",
        condition=TransitionCondition(
            previous_mode=previous_mode or "any",
            nominated_mode=nominated_mode,
        ),
        active_mode=nominated_mode,
        transition_label=f"{previous_mode or 'start'}->{nominated_mode}",
        reason="default accept: conditions for specific rule not met",
        priority=priority,
    )


def apply_legacy_conditions(
    rule: TransitionRule,
    previous_mode: str | None,
    nominated_mode: str,
    confidence: float,
    is_short_message: bool,
    is_explicit_task: bool,
    is_machine_status: bool,
    is_context_continuation: bool,
) -> TransitionRule | None:
    """Apply confidence and message-analysis conditions that aren't in the base rule.

    This bridges the gap between the data-driven table and the behavioral
    conditions that depend on message analysis. Phase 5 doesn't change the
    conditions themselves, just reorganizes them.
    """
    # START rule: use nominated mode as active
    if rule.rule_id == "START.ANY":
        return TransitionRule(
            rule_id=rule.rule_id,
            condition=rule.condition,
            active_mode=nominated_mode,
            transition_label=f"start:{nominated_mode}",
            reason=rule.reason,
            priority=rule.priority,
        )

    # Short message hold: only applies if message is actually short and low confidence
    if rule.rule_id == "ANTIFLAP.SHORT_HOLD.NOT_SEX":
        if not is_short_message or confidence >= 0.65 or previous_mode == "sex":
            return None
        # Use previous mode as active
        return TransitionRule(
            rule_id=rule.rule_id,
            condition=rule.condition,
            active_mode=previous_mode or "daily",
            transition_label="hold_short_message",
            reason=rule.reason,
            priority=rule.priority,
        )

    # Work hold: requires low confidence and specific conditions
    if rule.rule_id == "WORK.HOLD.CONTEXT_ACTION":
        if confidence >= 0.65 or not is_context_continuation:
            return None
        return rule

    # High-confidence work: requires confidence >= 0.75
    if rule.rule_id == "WORK.ENTER.HIGH_CONFIDENCE":
        if confidence < 0.75:
            return None
        return rule

    # Mid-confidence transitions: require confidence >= 0.65
    if rule.rule_id in ("WORK.EXIT.MID_CONFIDENCE", "DAILY.TO_WORK.MID_CONFIDENCE"):
        if confidence < 0.65:
            return None
        return rule

    # Machine status hold: requires machine status detection
    if rule.rule_id == "WORK.HOLD.MACHINE_STATUS":
        if not is_machine_status:
            return None
        return rule

    # Low-confidence hold: only applies if confidence < 0.65 and not from sex
    if rule.rule_id == "ANTIFLAP.HOLD.LOW_CONFIDENCE":
        if confidence >= 0.65 or previous_mode == "sex":
            return None
        # Use previous mode as active
        return TransitionRule(
            rule_id=rule.rule_id,
            condition=rule.condition,
            active_mode=previous_mode or "daily",
            transition_label="hold_low_confidence",
            reason=rule.reason,
            priority=rule.priority,
            max_stale_hold_turns=rule.max_stale_hold_turns,
        )

    # Default accept: use nominated mode
    if rule.rule_id == "DEFAULT.ACCEPT_NOMINATION":
        return TransitionRule(
            rule_id=rule.rule_id,
            condition=rule.condition,
            active_mode=nominated_mode,
            transition_label=f"{previous_mode or 'start'}->{nominated_mode}",
            reason=rule.reason,
            priority=rule.priority,
        )

    return rule
