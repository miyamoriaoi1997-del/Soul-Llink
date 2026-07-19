"""Immutable typed schema for structured rule engine.

Defines the data contracts for rule compilation, evidence matching, and
decision table execution. All types are frozen and validated at construction
to prevent silent data corruption.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any


# ── Enumerations ────────────────────────────────────────────────────────────


class Mode(str, Enum):
    """Valid persona operating modes."""
    DAILY = "daily"
    WORK = "work"
    SEX = "sex"


class FeatureClass(str, Enum):
    """Semantic feature categories for evidence."""
    HARD_BOUNDARY = "hard_boundary"
    EXPLICIT_META = "explicit_meta"
    EXPLICIT_TASK = "explicit_task"
    PROTECTED_PROGRESSION = "protected_progression"
    RELATIONSHIP = "relationship"
    CONTINUATION_BINDING = "continuation_binding"
    FALLBACK = "fallback"


class MatchScope(str, Enum):
    """Scope in which rule matching occurs."""
    CLAUSE = "clause"
    MESSAGE = "message"


class QuotePolicy(str, Enum):
    """How to handle terms found in quoted/cited text."""
    SUPPRESS = "suppress"  # Don't match inside quotes
    ALLOW = "allow"       # Match normally
    META_ONLY = "meta_only"  # Only for meta discussion


class NegationPolicy(str, Enum):
    """How to handle negated terms."""
    SUPPRESS = "suppress"  # Don't match if negated
    INVERT = "invert"     # Match inverted signal
    ALLOW = "allow"       # Match regardless


class DiscourseMode(str, Enum):
    """Classification of discourse structure."""
    DIRECT = "direct"
    QUOTED = "quoted"
    NEGATED = "negated"
    HYPOTHETICAL = "hypothetical"
    META = "meta"


class ConfidenceBand(str, Enum):
    """Discrete confidence levels."""
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"


class HardEvent(str, Enum):
    """Force events that override normal nomination."""
    NONE = "none"
    BOUNDARY_EXIT = "boundary_exit"
    CRISIS = "crisis"
    CREDENTIAL = "credential"
    PERMISSION = "permission"


class EvidenceStrength(str, Enum):
    """Evidence strength classification."""
    NONE = "none"
    WEAK = "weak"
    STRONG = "strong"
    EXPLICIT = "explicit"


class ContinuationTarget(str, Enum):
    """What a continuation signal binds to."""
    NONE = "none"
    WORK = "work"
    RELATIONSHIP = "relationship"
    AMBIGUOUS = "ambiguous"


# ── Immutable data types ────────────────────────────────────────────────────


@dataclass(frozen=True)
class RuleTerm:
    """A single matching term with optional metadata."""
    term: str
    weight: float = 1.0
    requires_context: tuple[str, ...] = field(default_factory=tuple)
    forbids_context: tuple[str, ...] = field(default_factory=tuple)

    def __post_init__(self) -> None:
        if not self.term or not isinstance(self.term, str):
            raise ValueError("term must be non-empty string")
        if not (0.0 <= self.weight <= 1.0):
            raise ValueError(f"weight must be in [0, 1], got {self.weight}")


@dataclass(frozen=True)
class CompiledRule:
    """A fully validated rule ready for execution."""
    rule_id: str
    feature_class: FeatureClass
    candidate_mode: Mode | None
    terms: tuple[RuleTerm, ...]
    weight: float
    priority: int
    match_any: bool  # True = any term, False = all terms
    requires_all: tuple[str, ...] = field(default_factory=tuple)
    requires_any: tuple[str, ...] = field(default_factory=tuple)
    forbids_any: tuple[str, ...] = field(default_factory=tuple)
    quote_policy: QuotePolicy = QuotePolicy.SUPPRESS
    negation_policy: NegationPolicy = NegationPolicy.SUPPRESS
    scope: MatchScope = MatchScope.MESSAGE
    activation_group: str = ""
    stop_processing: bool = False
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not self.rule_id:
            raise ValueError("rule_id must be non-empty")
        if not self.terms:
            raise ValueError(f"rule {self.rule_id}: must have at least one term")
        if not (0.0 <= self.weight <= 1.0):
            raise ValueError(f"rule {self.rule_id}: weight must be in [0, 1]")
        if self.priority < 0 or self.priority > 1000:
            raise ValueError(f"rule {self.rule_id}: priority must be in [0, 1000]")
        if self.stop_processing and self.feature_class != FeatureClass.HARD_BOUNDARY:
            raise ValueError(
                f"rule {self.rule_id}: stop_processing only allowed for hard_boundary"
            )


@dataclass(frozen=True)
class MatchEvidence:
    """Evidence from a single rule activation."""
    rule_id: str
    feature_class: FeatureClass
    candidate_mode: Mode | None
    weight: float
    priority: int
    clause_index: int
    span_start: int | None = None
    span_end: int | None = None
    matched_term: str = ""  # Opaque placeholder, never logged
    quoted: bool = False
    negated: bool = False
    hypothetical: bool = False
    meta_context: bool = False
    requires_satisfied: bool = True
    forbids_triggered: bool = False

    def is_valid(self) -> bool:
        """Check if this evidence should count."""
        return self.requires_satisfied and not self.forbids_triggered


@dataclass(frozen=True)
class EvidenceSet:
    """Collection of evidence from all matched rules."""
    evidences: tuple[MatchEvidence, ...]
    hard_events: tuple[HardEvent, ...]
    task_evidence: EvidenceStrength
    protected_evidence: EvidenceStrength
    relationship_evidence: EvidenceStrength
    discourse_mode: DiscourseMode
    continuation_target: ContinuationTarget
    confidence_band: ConfidenceBand


@dataclass(frozen=True)
class NominationDecision:
    """Output of the nomination decision table."""
    nominated_mode: Mode
    nominated_submode: str
    force_event: HardEvent
    confidence_band: ConfidenceBand
    reason_codes: tuple[str, ...]
    required_gate: str = ""
    supporting_evidence: tuple[str, ...] = field(default_factory=tuple)


@dataclass(frozen=True)
class TransitionContext:
    """Context for transition decision."""
    previous_mode: Mode
    nominated_mode: Mode
    force_event: HardEvent
    continuation_target: ContinuationTarget
    gate_status: str
    turns_in_previous: int
    confidence_band: ConfidenceBand


@dataclass(frozen=True)
class TransitionDecision:
    """Result of transition decision table."""
    active_mode: Mode
    transition_label: str
    same_turn_switch: bool
    reason_codes: tuple[str, ...]
    hold_reason: str = ""
    max_stale_hold: int = 0


# ── Validation utilities ────────────────────────────────────────────────────


def validate_weight(value: float, context: str = "") -> None:
    """Validate weight is in [0, 1]."""
    if not isinstance(value, (int, float)):
        raise TypeError(f"{context}: weight must be numeric, got {type(value)}")
    if not (0.0 <= value <= 1.0):
        raise ValueError(f"{context}: weight must be in [0, 1], got {value}")


def validate_priority(value: int, context: str = "") -> None:
    """Validate priority is in valid range."""
    if not isinstance(value, int):
        raise TypeError(f"{context}: priority must be int, got {type(value)}")
    if not (0 <= value <= 1000):
        raise ValueError(f"{context}: priority must be in [0, 1000], got {value}")


def validate_mode(value: str, context: str = "") -> Mode:
    """Validate and convert mode string."""
    try:
        return Mode(value.lower())
    except (ValueError, AttributeError) as e:
        raise ValueError(f"{context}: invalid mode '{value}'") from e


def validate_feature_class(value: str, context: str = "") -> FeatureClass:
    """Validate and convert feature class string."""
    try:
        return FeatureClass(value.lower())
    except (ValueError, AttributeError) as e:
        raise ValueError(f"{context}: invalid feature_class '{value}'") from e
