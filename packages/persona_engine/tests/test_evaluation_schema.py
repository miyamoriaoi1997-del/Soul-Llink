"""Tests for Phase 4 Task 4.1: Extended evaluation schema."""

import pytest
from packages.persona_engine.persona_orchestrator.evaluation_schema import (
    SingleTurnCase,
    SequenceTurnCase,
    EvaluationDataset,
    RiskClass,
    ReviewerStatus,
)


class TestSingleTurnCase:
    """Test extended single-turn case schema."""

    def test_backward_compatible_legacy_format(self):
        """Legacy cases without Phase 4 fields still load."""
        data = {
            "id": "test_case_1",
            "source": "human_labeled",
            "message": "test message",
            "recent_context": None,
            "previous_mode": "daily",
            "emotion_score": 1.0,
            "expected_mode": "work",
            "expected_transition": "daily->work",
            "expected_layers": ["work"],
            "forbidden_layers": ["sex"],
        }

        case = SingleTurnCase.from_dict(data)
        assert case.id == "test_case_1"
        assert case.expected_mode == "work"
        assert case.case_id is None
        assert case.risk_class is None

    def test_extended_format_with_phase4_fields(self):
        """Phase 4 extensions load correctly."""
        data = {
            "id": "test_case_2",
            "source": "human_labeled",
            "message": "test message",
            "previous_mode": "daily",
            "emotion_score": 1.0,
            "expected_mode": "work",
            "expected_transition": "daily->work",
            "expected_layers": ["work"],
            "forbidden_layers": [],
            "case_id": "stable_id_v1",
            "label_version": "2026-07-19",
            "evidence_class": "opaque_class_a",
            "expected_nomination": "work",
            "risk_class": "high",
            "reviewer_status": "approved",
        }

        case = SingleTurnCase.from_dict(data)
        assert case.case_id == "stable_id_v1"
        assert case.label_version == "2026-07-19"
        assert case.evidence_class == "opaque_class_a"
        assert case.expected_nomination == "work"
        assert case.risk_class == RiskClass.HIGH
        assert case.reviewer_status == ReviewerStatus.APPROVED

    def test_frozen_immutability(self):
        """Cases are immutable after creation."""
        case = SingleTurnCase(
            id="test",
            source="test",
            message="msg",
            recent_context=None,
            previous_mode=None,
            emotion_score=0.0,
            expected_mode="daily",
            expected_transition="*",
            expected_layers=[],
            forbidden_layers=[],
        )

        with pytest.raises((AttributeError, TypeError)):
            case.expected_mode = "work"  # type: ignore


class TestSequenceTurnCase:
    """Test multi-turn sequence case schema."""

    def test_sequence_case_creation(self):
        """Sequence cases support turn tracking."""
        data = {
            "sequence_id": "seq_001",
            "turn_index": 0,
            "thread_id": "thread_a",
            "message": "first turn",
            "emotion_score": 1.0,
            "expected_mode": "daily",
            "expected_transition": "enter",
            "expected_layers": ["daily"],
            "forbidden_layers": [],
            "expected_switch_turn": 2,
            "max_allowed_delay": 3,
        }

        case = SequenceTurnCase.from_dict(data)
        assert case.sequence_id == "seq_001"
        assert case.turn_index == 0
        assert case.expected_switch_turn == 2
        assert case.max_allowed_delay == 3

    def test_sequence_with_model_expectations(self):
        """Sequence can specify expected layer and model."""
        data = {
            "sequence_id": "seq_002",
            "turn_index": 1,
            "message": "second turn",
            "emotion_score": 1.0,
            "expected_mode": "work",
            "expected_transition": "daily->work",
            "expected_layers": ["work"],
            "forbidden_layers": [],
            "expected_selected_layer": "work",
            "expected_selected_model": "opus",
        }

        case = SequenceTurnCase.from_dict(data)
        assert case.expected_selected_layer == "work"
        assert case.expected_selected_model == "opus"

    def test_sequence_frozen(self):
        """Sequence cases are immutable."""
        case = SequenceTurnCase(
            sequence_id="seq",
            turn_index=0,
            thread_id=None,
            message="msg",
            emotion_score=0.0,
            expected_mode="daily",
            expected_nomination=None,
            expected_transition="*",
            expected_layers=[],
            forbidden_layers=[],
            expected_switch_turn=None,
            max_allowed_delay=None,
            expected_selected_layer=None,
            expected_selected_model=None,
            evidence_class=None,
            risk_class=None,
        )

        with pytest.raises((AttributeError, TypeError)):
            case.turn_index = 1  # type: ignore


class TestEvaluationDataset:
    """Test dataset container and validation."""

    def test_dataset_creation_valid(self):
        """Dataset validates and stores cases."""
        case1 = SingleTurnCase(
            id="case1",
            source="test",
            message="msg1",
            recent_context=None,
            previous_mode=None,
            emotion_score=0.0,
            expected_mode="daily",
            expected_transition="*",
            expected_layers=[],
            forbidden_layers=[],
        )

        dataset = EvaluationDataset(
            name="test_dataset",
            kind="development",
            single_turn_cases=[case1],
            sequence_cases=[],
            frozen=False,
        )

        assert dataset.name == "test_dataset"
        assert dataset.kind == "development"
        assert len(dataset.single_turn_cases) == 1
        assert dataset.frozen is False

    def test_dataset_rejects_invalid_kind(self):
        """Dataset kind must be recognized."""
        with pytest.raises(ValueError, match="Invalid dataset kind"):
            EvaluationDataset(
                name="bad",
                kind="unknown_kind",
                single_turn_cases=[],
                sequence_cases=[],
                frozen=False,
            )

    def test_dataset_detects_duplicate_single_turn_ids(self):
        """Duplicate IDs are rejected."""
        case1 = SingleTurnCase(
            id="duplicate",
            source="test",
            message="msg1",
            recent_context=None,
            previous_mode=None,
            emotion_score=0.0,
            expected_mode="daily",
            expected_transition="*",
            expected_layers=[],
            forbidden_layers=[],
        )
        case2 = SingleTurnCase(
            id="duplicate",
            source="test",
            message="msg2",
            recent_context=None,
            previous_mode=None,
            emotion_score=0.0,
            expected_mode="work",
            expected_transition="*",
            expected_layers=[],
            forbidden_layers=[],
        )

        with pytest.raises(ValueError, match="Duplicate single-turn case IDs"):
            EvaluationDataset(
                name="bad",
                kind="development",
                single_turn_cases=[case1, case2],
                sequence_cases=[],
                frozen=False,
            )

    def test_dataset_detects_duplicate_sequence_keys(self):
        """Duplicate (sequence_id, turn_index) pairs are rejected."""
        seq1 = SequenceTurnCase(
            sequence_id="seq1",
            turn_index=0,
            thread_id=None,
            message="msg1",
            emotion_score=0.0,
            expected_mode="daily",
            expected_nomination=None,
            expected_transition="*",
            expected_layers=[],
            forbidden_layers=[],
            expected_switch_turn=None,
            max_allowed_delay=None,
            expected_selected_layer=None,
            expected_selected_model=None,
            evidence_class=None,
            risk_class=None,
        )
        seq2 = SequenceTurnCase(
            sequence_id="seq1",
            turn_index=0,  # Duplicate
            thread_id=None,
            message="msg2",
            emotion_score=0.0,
            expected_mode="work",
            expected_nomination=None,
            expected_transition="*",
            expected_layers=[],
            forbidden_layers=[],
            expected_switch_turn=None,
            max_allowed_delay=None,
            expected_selected_layer=None,
            expected_selected_model=None,
            evidence_class=None,
            risk_class=None,
        )

        with pytest.raises(ValueError, match="Duplicate sequence"):
            EvaluationDataset(
                name="bad",
                kind="development",
                single_turn_cases=[],
                sequence_cases=[seq1, seq2],
                frozen=False,
            )

    def test_frozen_dataset_for_holdout(self):
        """Holdout datasets can be marked frozen."""
        dataset = EvaluationDataset(
            name="holdout",
            kind="holdout",
            single_turn_cases=[],
            sequence_cases=[],
            frozen=True,
        )

        assert dataset.frozen is True
        assert dataset.kind == "holdout"
