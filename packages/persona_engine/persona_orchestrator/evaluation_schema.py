"""Extended schema for human-labeled evaluation cases and sequences.

Phase 4 Task 4.1: Extend annotation schema with tracking, risk classification,
and sequence support while treating all sensitive content as opaque.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Any, Optional


class RiskClass(Enum):
    """Risk classification for mode transitions."""
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


class ReviewerStatus(Enum):
    """Human review status."""
    PENDING = "pending"
    APPROVED = "approved"
    REJECTED = "rejected"
    NEEDS_REVISION = "needs_revision"


@dataclass(frozen=True)
class SingleTurnCase:
    """Extended single-turn evaluation case.

    Preserves backward compatibility with existing human_truth_cases.jsonl
    while adding Phase 4 tracking and classification fields.
    """
    # Core identity (backward compatible)
    id: str
    source: str
    message: str  # Opaque - never echoed in reports

    # Context
    recent_context: Optional[list[str]]
    previous_mode: Optional[str]
    emotion_score: float

    # Expected outcomes (backward compatible)
    expected_mode: str
    expected_transition: str
    expected_layers: list[str]
    forbidden_layers: list[str]

    # Phase 4 extensions
    case_id: Optional[str] = None  # Stable ID for tracking across versions
    label_version: Optional[str] = None
    evidence_class: Optional[str] = None  # Opaque classification
    expected_nomination: Optional[str] = None  # Classifier nomination vs final mode
    expected_active_mode: Optional[str] = None  # Alias for expected_mode
    risk_class: Optional[RiskClass] = None
    reviewer_status: Optional[ReviewerStatus] = None

    # Metadata
    rationale: Optional[str] = None

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> SingleTurnCase:
        """Load from JSONL dict, handling both legacy and extended formats."""
        # Handle risk_class conversion
        risk_class = data.get('risk_class')
        if risk_class and isinstance(risk_class, str):
            risk_class = RiskClass(risk_class)

        # Handle reviewer_status conversion
        reviewer_status = data.get('reviewer_status')
        if reviewer_status and isinstance(reviewer_status, str):
            reviewer_status = ReviewerStatus(reviewer_status)

        return cls(
            id=data['id'],
            source=data['source'],
            message=data['message'],
            recent_context=data.get('recent_context'),
            previous_mode=data.get('previous_mode'),
            emotion_score=data.get('emotion_score', 0.0),
            expected_mode=data['expected_mode'],
            expected_transition=data['expected_transition'],
            expected_layers=data.get('expected_layers', []),
            forbidden_layers=data.get('forbidden_layers', []),
            case_id=data.get('case_id'),
            label_version=data.get('label_version'),
            evidence_class=data.get('evidence_class'),
            expected_nomination=data.get('expected_nomination'),
            expected_active_mode=data.get('expected_active_mode'),
            risk_class=risk_class,
            reviewer_status=reviewer_status,
            rationale=data.get('rationale'),
        )


@dataclass(frozen=True)
class SequenceTurnCase:
    """Multi-turn sequence case for testing state transitions over time."""
    # Sequence identity
    sequence_id: str
    turn_index: int
    thread_id: Optional[str]

    # Turn content (opaque)
    message: str
    emotion_score: float

    # Expected outcomes
    expected_mode: str
    expected_nomination: Optional[str]
    expected_transition: str
    expected_layers: list[str]
    forbidden_layers: list[str]

    # Sequence-specific expectations
    expected_switch_turn: Optional[int]  # When mode should change
    max_allowed_delay: Optional[int]  # Maximum stale hold turns
    expected_selected_layer: Optional[str]
    expected_selected_model: Optional[str]

    # Classification
    evidence_class: Optional[str]
    risk_class: Optional[RiskClass]

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> SequenceTurnCase:
        """Load from JSONL dict."""
        risk_class = data.get('risk_class')
        if risk_class and isinstance(risk_class, str):
            risk_class = RiskClass(risk_class)

        return cls(
            sequence_id=data['sequence_id'],
            turn_index=data['turn_index'],
            thread_id=data.get('thread_id'),
            message=data['message'],
            emotion_score=data.get('emotion_score', 0.0),
            expected_mode=data['expected_mode'],
            expected_nomination=data.get('expected_nomination'),
            expected_transition=data['expected_transition'],
            expected_layers=data.get('expected_layers', []),
            forbidden_layers=data.get('forbidden_layers', []),
            expected_switch_turn=data.get('expected_switch_turn'),
            max_allowed_delay=data.get('max_allowed_delay'),
            expected_selected_layer=data.get('expected_selected_layer'),
            expected_selected_model=data.get('expected_selected_model'),
            evidence_class=data.get('evidence_class'),
            risk_class=risk_class,
        )


@dataclass(frozen=True)
class EvaluationDataset:
    """Container for evaluation cases with metadata."""
    name: str
    kind: str  # "regression", "development", "holdout", "adversarial"
    single_turn_cases: list[SingleTurnCase]
    sequence_cases: list[SequenceTurnCase]
    frozen: bool  # True for holdout/regression

    def __post_init__(self):
        """Validate dataset integrity."""
        if self.kind not in ("regression", "development", "holdout", "adversarial"):
            raise ValueError(f"Invalid dataset kind: {self.kind}")

        # Check for duplicate IDs
        single_ids = [c.id for c in self.single_turn_cases]
        if len(single_ids) != len(set(single_ids)):
            raise ValueError("Duplicate single-turn case IDs")

        sequence_keys = [(c.sequence_id, c.turn_index) for c in self.sequence_cases]
        if len(sequence_keys) != len(set(sequence_keys)):
            raise ValueError("Duplicate sequence (id, turn) pairs")
