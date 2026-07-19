"""Tests for Phase 4 Task 4.2: Layered dataset management."""

import json
import pytest
from pathlib import Path

from packages.persona_engine.persona_orchestrator.dataset_loader import DatasetLoader, DatasetValidator
from packages.persona_engine.persona_orchestrator.evaluation_schema import EvaluationDataset, SingleTurnCase


class TestDatasetLoader:
    """Test dataset loading from JSONL files."""

    def test_load_regression_set_from_existing_file(self, tmp_path):
        """Load existing human_truth_cases.jsonl as regression set."""
        fixtures = tmp_path / "fixtures"
        fixtures.mkdir()

        # Create sample regression file
        cases_file = fixtures / "human_truth_cases.jsonl"
        with cases_file.open('w', encoding='utf-8') as f:
            f.write(json.dumps({
                "id": "reg_case_1",
                "source": "human_labeled",
                "message": "test message",
                "previous_mode": "daily",
                "emotion_score": 1.0,
                "expected_mode": "work",
                "expected_transition": "daily->work",
                "expected_layers": ["work"],
                "forbidden_layers": [],
            }) + '\n')

        loader = DatasetLoader(fixtures)
        dataset = loader.load_regression_set()

        assert dataset.name == "regression"
        assert dataset.kind == "regression"
        assert dataset.frozen is True
        assert len(dataset.single_turn_cases) == 1
        assert dataset.single_turn_cases[0].id == "reg_case_1"

    def test_load_regression_empty_when_file_missing(self, tmp_path):
        """Regression set is empty if file doesn't exist yet."""
        fixtures = tmp_path / "fixtures"
        fixtures.mkdir()

        loader = DatasetLoader(fixtures)
        dataset = loader.load_regression_set()

        assert dataset.name == "regression"
        assert dataset.frozen is True
        assert len(dataset.single_turn_cases) == 0

    def test_load_development_set(self, tmp_path):
        """Load development set from separate file."""
        fixtures = tmp_path / "fixtures"
        fixtures.mkdir()

        dev_file = fixtures / "development_cases.jsonl"
        with dev_file.open('w', encoding='utf-8') as f:
            f.write(json.dumps({
                "id": "dev_case_1",
                "source": "synthetic",
                "message": "dev message",
                "previous_mode": None,
                "emotion_score": 0.0,
                "expected_mode": "daily",
                "expected_transition": "enter",
                "expected_layers": [],
                "forbidden_layers": [],
            }) + '\n')

        loader = DatasetLoader(fixtures)
        dataset = loader.load_development_set()

        assert dataset.name == "development"
        assert dataset.kind == "development"
        assert dataset.frozen is False
        assert len(dataset.single_turn_cases) == 1

    def test_load_holdout_set(self, tmp_path):
        """Load holdout set (frozen)."""
        fixtures = tmp_path / "fixtures"
        fixtures.mkdir()

        holdout_file = fixtures / "holdout_cases.jsonl"
        with holdout_file.open('w', encoding='utf-8') as f:
            f.write(json.dumps({
                "id": "holdout_case_1",
                "source": "human_labeled",
                "message": "holdout message",
                "previous_mode": None,
                "emotion_score": 0.0,
                "expected_mode": "daily",
                "expected_transition": "enter",
                "expected_layers": [],
                "forbidden_layers": [],
            }) + '\n')

        loader = DatasetLoader(fixtures)
        dataset = loader.load_holdout_set()

        assert dataset.name == "holdout"
        assert dataset.kind == "holdout"
        assert dataset.frozen is True
        assert len(dataset.single_turn_cases) == 1

    def test_load_adversarial_set(self, tmp_path):
        """Load adversarial edge-case set."""
        fixtures = tmp_path / "fixtures"
        fixtures.mkdir()

        adv_file = fixtures / "adversarial_cases.jsonl"
        with adv_file.open('w', encoding='utf-8') as f:
            f.write(json.dumps({
                "id": "adv_case_1",
                "source": "adversarial",
                "message": "edge case",
                "previous_mode": None,
                "emotion_score": 0.0,
                "expected_mode": "daily",
                "expected_transition": "enter",
                "expected_layers": [],
                "forbidden_layers": [],
            }) + '\n')

        loader = DatasetLoader(fixtures)
        dataset = loader.load_adversarial_set()

        assert dataset.name == "adversarial"
        assert dataset.kind == "adversarial"
        assert dataset.frozen is False
        assert len(dataset.single_turn_cases) == 1

    def test_load_all_datasets(self, tmp_path):
        """Load all four dataset types at once."""
        fixtures = tmp_path / "fixtures"
        fixtures.mkdir()

        # Create minimal files
        (fixtures / "human_truth_cases.jsonl").write_text(
            json.dumps({
                "id": "reg_1",
                "source": "human",
                "message": "m",
                "emotion_score": 0.0,
                "expected_mode": "daily",
                "expected_transition": "*",
                "expected_layers": [],
                "forbidden_layers": [],
            }) + '\n',
            encoding='utf-8'
        )

        loader = DatasetLoader(fixtures)
        datasets = loader.load_all_datasets()

        assert set(datasets.keys()) == {"regression", "development", "holdout", "adversarial"}
        assert datasets["regression"].frozen is True
        assert datasets["holdout"].frozen is True
        assert datasets["development"].frozen is False
        assert datasets["adversarial"].frozen is False


class TestDatasetValidator:
    """Test cross-contamination detection and validation."""

    def test_no_overlap_valid(self):
        """Valid when development and holdout have no shared IDs."""
        dev = EvaluationDataset(
            name="development",
            kind="development",
            single_turn_cases=[
                SingleTurnCase(
                    id="dev_1",
                    source="test",
                    message="m1",
                    recent_context=None,
                    previous_mode=None,
                    emotion_score=0.0,
                    expected_mode="daily",
                    expected_transition="*",
                    expected_layers=[],
                    forbidden_layers=[],
                )
            ],
            sequence_cases=[],
            frozen=False,
        )

        holdout = EvaluationDataset(
            name="holdout",
            kind="holdout",
            single_turn_cases=[
                SingleTurnCase(
                    id="holdout_1",
                    source="test",
                    message="m2",
                    recent_context=None,
                    previous_mode=None,
                    emotion_score=0.0,
                    expected_mode="daily",
                    expected_transition="*",
                    expected_layers=[],
                    forbidden_layers=[],
                )
            ],
            sequence_cases=[],
            frozen=True,
        )

        datasets = {"development": dev, "holdout": holdout}
        violations = DatasetValidator.check_no_overlap(datasets)

        assert violations == []

    def test_detects_single_turn_overlap(self):
        """Detect when same ID appears in both dev and holdout."""
        dev = EvaluationDataset(
            name="development",
            kind="development",
            single_turn_cases=[
                SingleTurnCase(
                    id="shared_id",
                    source="test",
                    message="m1",
                    recent_context=None,
                    previous_mode=None,
                    emotion_score=0.0,
                    expected_mode="daily",
                    expected_transition="*",
                    expected_layers=[],
                    forbidden_layers=[],
                )
            ],
            sequence_cases=[],
            frozen=False,
        )

        holdout = EvaluationDataset(
            name="holdout",
            kind="holdout",
            single_turn_cases=[
                SingleTurnCase(
                    id="shared_id",
                    source="test",
                    message="m2",
                    recent_context=None,
                    previous_mode=None,
                    emotion_score=0.0,
                    expected_mode="work",
                    expected_transition="*",
                    expected_layers=[],
                    forbidden_layers=[],
                )
            ],
            sequence_cases=[],
            frozen=True,
        )

        datasets = {"development": dev, "holdout": holdout}
        violations = DatasetValidator.check_no_overlap(datasets)

        assert len(violations) == 1
        assert "Single-turn ID overlap" in violations[0]
        assert "shared_id" in violations[0]

    def test_validate_all_comprehensive(self):
        """Comprehensive validation report."""
        regression = EvaluationDataset(
            name="regression",
            kind="regression",
            single_turn_cases=[],
            sequence_cases=[],
            frozen=True,
        )

        dev = EvaluationDataset(
            name="development",
            kind="development",
            single_turn_cases=[],
            sequence_cases=[],
            frozen=False,
        )

        holdout = EvaluationDataset(
            name="holdout",
            kind="holdout",
            single_turn_cases=[],
            sequence_cases=[],
            frozen=True,
        )

        adversarial = EvaluationDataset(
            name="adversarial",
            kind="adversarial",
            single_turn_cases=[],
            sequence_cases=[],
            frozen=False,
        )

        datasets = {
            "regression": regression,
            "development": dev,
            "holdout": holdout,
            "adversarial": adversarial,
        }

        report = DatasetValidator.validate_all(datasets)

        assert report["valid"] is True
        assert report["violations"] == []
        assert report["frozen_status"]["regression"] is True
        assert report["frozen_status"]["holdout"] is True
        assert report["frozen_status"]["development"] is False
        assert report["frozen_status"]["adversarial"] is False

    def test_validate_fails_if_regression_not_frozen(self):
        """Regression must always be frozen."""
        regression = EvaluationDataset(
            name="regression",
            kind="regression",
            single_turn_cases=[],
            sequence_cases=[],
            frozen=False,  # WRONG
        )

        report = DatasetValidator.validate_all({"regression": regression})

        assert report["valid"] is False
        assert any("Regression set must be frozen" in v for v in report["violations"])
