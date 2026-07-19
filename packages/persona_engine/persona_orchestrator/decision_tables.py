"""Decision tables for nomination and transition logic.

Maps evidence collections to finite input equivalence classes for
UNIQUE decision table lookup.
"""
from __future__ import annotations

from typing import Sequence

from .rule_schema import (
    ConfidenceBand,
    ContinuationTarget,
    DiscourseMode,
    EvidenceStrength,
    FeatureClass,
    HardEvent,
    MatchEvidence,
    Mode,
    NominationDecision,
)


# ── Evidence Classification Functions ───────────────────────────────────────


def classify_hard_event(evidences: Sequence[MatchEvidence]) -> HardEvent:
    """Extract hard event from evidence collection.

    Args:
        evidences: Collection of match evidence

    Returns:
        Highest priority hard event, or NONE
    """
    hard_boundary_evidence = [
        e for e in evidences
        if e.feature_class == FeatureClass.HARD_BOUNDARY and e.is_valid()
    ]

    if not hard_boundary_evidence:
        return HardEvent.NONE

    # For now, classify as boundary_exit
    # Future: parse rule metadata for specific event types
    return HardEvent.BOUNDARY_EXIT


def classify_task_evidence(evidences: Sequence[MatchEvidence]) -> EvidenceStrength:
    """Classify task evidence strength.

    Args:
        evidences: Collection of match evidence

    Returns:
        Task evidence strength band
    """
    task_evidence = [
        e for e in evidences
        if e.feature_class == FeatureClass.EXPLICIT_TASK and e.is_valid()
    ]

    if not task_evidence:
        return EvidenceStrength.NONE

    # Calculate aggregate strength
    max_weight = max(e.weight for e in task_evidence)
    max_priority = max(e.priority for e in task_evidence)

    # Thresholds
    if max_weight >= 0.85 and max_priority >= 800:
        return EvidenceStrength.EXPLICIT
    elif max_weight >= 0.70 or max_priority >= 700:
        return EvidenceStrength.STRONG
    else:
        return EvidenceStrength.WEAK


def classify_protected_evidence(evidences: Sequence[MatchEvidence]) -> EvidenceStrength:
    """Classify protected progression evidence strength.

    Args:
        evidences: Collection of match evidence

    Returns:
        Protected evidence strength band
    """
    protected_evidence = [
        e for e in evidences
        if e.feature_class == FeatureClass.PROTECTED_PROGRESSION and e.is_valid()
    ]

    if not protected_evidence:
        return EvidenceStrength.NONE

    max_weight = max(e.weight for e in protected_evidence)
    max_priority = max(e.priority for e in protected_evidence)

    if max_weight >= 0.80 and max_priority >= 700:
        return EvidenceStrength.EXPLICIT
    elif max_weight >= 0.60:
        return EvidenceStrength.STRONG
    else:
        return EvidenceStrength.WEAK


def classify_relationship_evidence(evidences: Sequence[MatchEvidence]) -> EvidenceStrength:
    """Classify relationship evidence strength.

    Args:
        evidences: Collection of match evidence

    Returns:
        Relationship evidence strength band
    """
    relationship_evidence = [
        e for e in evidences
        if e.feature_class == FeatureClass.RELATIONSHIP and e.is_valid()
    ]

    if not relationship_evidence:
        return EvidenceStrength.NONE

    max_weight = max(e.weight for e in relationship_evidence)
    max_priority = max(e.priority for e in relationship_evidence)

    if max_weight >= 0.80 and max_priority >= 600:
        return EvidenceStrength.EXPLICIT
    elif max_weight >= 0.60:
        return EvidenceStrength.STRONG
    else:
        return EvidenceStrength.WEAK


def classify_discourse(evidences: Sequence[MatchEvidence]) -> DiscourseMode:
    """Determine discourse mode from evidence flags.

    Priority: meta > hypothetical > quoted > negated > direct

    Args:
        evidences: Collection of match evidence

    Returns:
        Discourse mode
    """
    if not evidences:
        return DiscourseMode.DIRECT

    # Check highest priority discourse markers
    has_meta = any(e.meta_context for e in evidences)
    has_hypothetical = any(e.hypothetical for e in evidences)
    has_quoted = any(e.quoted for e in evidences)
    has_negated = any(e.negated for e in evidences)

    if has_meta:
        return DiscourseMode.META
    elif has_hypothetical:
        return DiscourseMode.HYPOTHETICAL
    elif has_quoted:
        return DiscourseMode.QUOTED
    elif has_negated:
        return DiscourseMode.NEGATED
    else:
        return DiscourseMode.DIRECT


def classify_continuation_target(evidences: Sequence[MatchEvidence]) -> ContinuationTarget:
    """Determine continuation target from context.

    Args:
        evidences: Collection of match evidence

    Returns:
        Continuation target
    """
    continuation_evidence = [
        e for e in evidences
        if e.feature_class == FeatureClass.CONTINUATION_BINDING and e.is_valid()
    ]

    if not continuation_evidence:
        return ContinuationTarget.NONE

    # Check which mode continuations are satisfied
    work_continuations = [
        e for e in continuation_evidence
        if e.candidate_mode == Mode.WORK
    ]
    relationship_continuations = [
        e for e in continuation_evidence
        if e.candidate_mode == Mode.DAILY
    ]

    has_work = len(work_continuations) > 0
    has_relationship = len(relationship_continuations) > 0

    if has_work and has_relationship:
        return ContinuationTarget.AMBIGUOUS
    elif has_work:
        return ContinuationTarget.WORK
    elif has_relationship:
        return ContinuationTarget.RELATIONSHIP
    else:
        return ContinuationTarget.AMBIGUOUS


def classify_confidence(evidences: Sequence[MatchEvidence]) -> ConfidenceBand:
    """Map evidence quality to confidence bands.

    Args:
        evidences: Collection of match evidence

    Returns:
        Confidence band
    """
    if not evidences:
        return ConfidenceBand.LOW

    valid_evidence = [e for e in evidences if e.is_valid()]

    if not valid_evidence:
        return ConfidenceBand.LOW

    # Calculate aggregate confidence
    max_weight = max(e.weight for e in valid_evidence)
    max_priority = max(e.priority for e in valid_evidence)
    count = len(valid_evidence)

    # High confidence: strong evidence, multiple supporting rules
    if max_weight >= 0.85 and max_priority >= 800 and count >= 2:
        return ConfidenceBand.HIGH
    elif max_weight >= 0.85 or (max_priority >= 700 and count >= 2):
        return ConfidenceBand.HIGH
    elif max_weight >= 0.70 or max_priority >= 600:
        return ConfidenceBand.MEDIUM
    else:
        return ConfidenceBand.LOW


# ── Nomination Decision Table ───────────────────────────────────────────────


def nominate_mode(
    hard_event: HardEvent,
    task_evidence: EvidenceStrength,
    protected_evidence: EvidenceStrength,
    relationship_evidence: EvidenceStrength,
    discourse: DiscourseMode,
    continuation: ContinuationTarget,
    confidence: ConfidenceBand,
) -> NominationDecision:
    """UNIQUE nomination decision table.

    Maps finite input equivalence classes to a single nomination decision.

    Args:
        hard_event: Hard event classification
        task_evidence: Task evidence strength
        protected_evidence: Protected evidence strength
        relationship_evidence: Relationship evidence strength
        discourse: Discourse mode
        continuation: Continuation target
        confidence: Confidence band

    Returns:
        Nomination decision (must be unique for each input combination)
    """
    # Priority 1: Hard events override everything
    if hard_event == HardEvent.BOUNDARY_EXIT:
        return NominationDecision(
            nominated_mode=Mode.DAILY,
            nominated_submode="",
            force_event=HardEvent.BOUNDARY_EXIT,
            confidence_band=ConfidenceBand.HIGH,
            reason_codes=("hard_boundary_exit",)
        )

    # Priority 2: Meta discussion routes to Work
    if discourse == DiscourseMode.META:
        return NominationDecision(
            nominated_mode=Mode.WORK,
            nominated_submode="",
            force_event=HardEvent.NONE,
            confidence_band=confidence,
            reason_codes=("meta_discussion",)
        )

    # Priority 3: Explicit task with direct discourse
    if task_evidence == EvidenceStrength.EXPLICIT and discourse == DiscourseMode.DIRECT:
        return NominationDecision(
            nominated_mode=Mode.WORK,
            nominated_submode="",
            force_event=HardEvent.NONE,
            confidence_band=confidence,
            reason_codes=("explicit_task",)
        )

    # Priority 4: Strong task evidence
    if task_evidence in (EvidenceStrength.STRONG, EvidenceStrength.EXPLICIT):
        return NominationDecision(
            nominated_mode=Mode.WORK,
            nominated_submode="",
            force_event=HardEvent.NONE,
            confidence_band=confidence,
            reason_codes=("task_evidence",)
        )

    # Priority 5: Protected progression (requires gate)
    if protected_evidence in (EvidenceStrength.EXPLICIT, EvidenceStrength.STRONG):
        return NominationDecision(
            nominated_mode=Mode.SEX,
            nominated_submode="",
            force_event=HardEvent.NONE,
            confidence_band=confidence,
            reason_codes=("protected_progression",),
            required_gate="desire_check"
        )

    # Priority 6: Explicit relationship evidence
    if relationship_evidence == EvidenceStrength.EXPLICIT:
        return NominationDecision(
            nominated_mode=Mode.DAILY,
            nominated_submode="",
            force_event=HardEvent.NONE,
            confidence_band=confidence,
            reason_codes=("explicit_relationship",)
        )

    # Priority 7: Continuation binding
    if continuation == ContinuationTarget.WORK:
        return NominationDecision(
            nominated_mode=Mode.WORK,
            nominated_submode="",
            force_event=HardEvent.NONE,
            confidence_band=confidence,
            reason_codes=("continuation_work",)
        )
    elif continuation == ContinuationTarget.RELATIONSHIP:
        return NominationDecision(
            nominated_mode=Mode.DAILY,
            nominated_submode="",
            force_event=HardEvent.NONE,
            confidence_band=confidence,
            reason_codes=("continuation_relationship",)
        )

    # Priority 8: Weak relationship evidence
    if relationship_evidence in (EvidenceStrength.WEAK, EvidenceStrength.STRONG):
        return NominationDecision(
            nominated_mode=Mode.DAILY,
            nominated_submode="",
            force_event=HardEvent.NONE,
            confidence_band=confidence,
            reason_codes=("relationship_evidence",)
        )

    # Default: Daily with low confidence
    return NominationDecision(
        nominated_mode=Mode.DAILY,
        nominated_submode="",
        force_event=HardEvent.NONE,
        confidence_band=ConfidenceBand.LOW,
        reason_codes=("default_daily",)
    )
