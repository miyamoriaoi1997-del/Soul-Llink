"""Dataset management for Phase 4 Task 4.2: Layered evaluation datasets.

Manages four dataset types:
1. Regression: Historical confirmed cases (frozen)
2. Development: Cases for rule tuning (editable)
3. Holdout: Time-separated, frozen after creation
4. Adversarial: Edge cases, mixed intent, rapid reversal (editable)

Ensures no cross-contamination between development and holdout.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from .evaluation_schema import (
    EvaluationDataset,
    SingleTurnCase,
    SequenceTurnCase,
)


class DatasetLoader:
    """Load and validate evaluation datasets."""

    def __init__(self, fixtures_root: Path):
        self.fixtures_root = fixtures_root

    def load_single_turn_jsonl(self, path: Path) -> list[SingleTurnCase]:
        """Load single-turn cases from JSONL file."""
        cases = []
        with path.open('r', encoding='utf-8') as f:
            for line_no, line in enumerate(f, 1):
                line = line.strip()
                if not line:
                    continue
                try:
                    data = json.loads(line)
                    case = SingleTurnCase.from_dict(data)
                    cases.append(case)
                except Exception as e:
                    raise ValueError(f"Error loading line {line_no} from {path}: {e}") from e
        return cases

    def load_sequence_jsonl(self, path: Path) -> list[SequenceTurnCase]:
        """Load sequence cases from JSONL file."""
        cases = []
        with path.open('r', encoding='utf-8') as f:
            for line_no, line in enumerate(f, 1):
                line = line.strip()
                if not line:
                    continue
                try:
                    data = json.loads(line)
                    case = SequenceTurnCase.from_dict(data)
                    cases.append(case)
                except Exception as e:
                    raise ValueError(f"Error loading line {line_no} from {path}: {e}") from e
        return cases

    def load_regression_set(self) -> EvaluationDataset:
        """Load locked regression set from human_truth_cases.jsonl.

        This is the historical baseline - always frozen.
        """
        path = self.fixtures_root / "human_truth_cases.jsonl"
        if not path.exists():
            # Return empty dataset if file doesn't exist yet
            return EvaluationDataset(
                name="regression",
                kind="regression",
                single_turn_cases=[],
                sequence_cases=[],
                frozen=True,
            )

        cases = self.load_single_turn_jsonl(path)
        return EvaluationDataset(
            name="regression",
            kind="regression",
            single_turn_cases=cases,
            sequence_cases=[],
            frozen=True,
        )

    def load_development_set(self) -> EvaluationDataset:
        """Load development set for rule tuning."""
        single_path = self.fixtures_root / "development_cases.jsonl"
        sequence_path = self.fixtures_root / "development_sequences.jsonl"

        single_cases = []
        if single_path.exists():
            single_cases = self.load_single_turn_jsonl(single_path)

        sequence_cases = []
        if sequence_path.exists():
            sequence_cases = self.load_sequence_jsonl(sequence_path)

        return EvaluationDataset(
            name="development",
            kind="development",
            single_turn_cases=single_cases,
            sequence_cases=sequence_cases,
            frozen=False,
        )

    def load_holdout_set(self) -> EvaluationDataset:
        """Load time-separated holdout set (frozen after initial creation)."""
        single_path = self.fixtures_root / "holdout_cases.jsonl"
        sequence_path = self.fixtures_root / "holdout_sequences.jsonl"

        single_cases = []
        if single_path.exists():
            single_cases = self.load_single_turn_jsonl(single_path)

        sequence_cases = []
        if sequence_path.exists():
            sequence_cases = self.load_sequence_jsonl(sequence_path)

        return EvaluationDataset(
            name="holdout",
            kind="holdout",
            single_turn_cases=single_cases,
            sequence_cases=sequence_cases,
            frozen=True,
        )

    def load_adversarial_set(self) -> EvaluationDataset:
        """Load adversarial edge-case set."""
        single_path = self.fixtures_root / "adversarial_cases.jsonl"
        sequence_path = self.fixtures_root / "adversarial_sequences.jsonl"

        single_cases = []
        if single_path.exists():
            single_cases = self.load_single_turn_jsonl(single_path)

        sequence_cases = []
        if sequence_path.exists():
            sequence_cases = self.load_sequence_jsonl(sequence_path)

        return EvaluationDataset(
            name="adversarial",
            kind="adversarial",
            single_turn_cases=single_cases,
            sequence_cases=sequence_cases,
            frozen=False,
        )

    def load_all_datasets(self) -> dict[str, EvaluationDataset]:
        """Load all four dataset types."""
        return {
            "regression": self.load_regression_set(),
            "development": self.load_development_set(),
            "holdout": self.load_holdout_set(),
            "adversarial": self.load_adversarial_set(),
        }


class DatasetValidator:
    """Validate dataset integrity and cross-contamination."""

    @staticmethod
    def check_no_overlap(datasets: dict[str, EvaluationDataset]) -> list[str]:
        """Check for ID overlap between development and holdout sets.

        Returns list of violations (empty if valid).
        """
        violations = []

        # Get IDs from development and holdout
        dev = datasets.get("development")
        holdout = datasets.get("holdout")

        if not dev or not holdout:
            return violations

        # Check single-turn overlap
        dev_single_ids = {c.id for c in dev.single_turn_cases}
        holdout_single_ids = {c.id for c in holdout.single_turn_cases}
        overlap_single = dev_single_ids & holdout_single_ids
        if overlap_single:
            violations.append(f"Single-turn ID overlap: {sorted(overlap_single)}")

        # Check sequence overlap
        dev_seq_ids = {c.sequence_id for c in dev.sequence_cases}
        holdout_seq_ids = {c.sequence_id for c in holdout.sequence_cases}
        overlap_seq = dev_seq_ids & holdout_seq_ids
        if overlap_seq:
            violations.append(f"Sequence ID overlap: {sorted(overlap_seq)}")

        return violations

    @staticmethod
    def validate_all(datasets: dict[str, EvaluationDataset]) -> dict[str, Any]:
        """Comprehensive validation report."""
        report = {
            "valid": True,
            "violations": [],
            "counts": {},
            "frozen_status": {},
        }

        # Check overlap
        overlap_violations = DatasetValidator.check_no_overlap(datasets)
        if overlap_violations:
            report["valid"] = False
            report["violations"].extend(overlap_violations)

        # Count cases
        for name, dataset in datasets.items():
            report["counts"][name] = {
                "single_turn": len(dataset.single_turn_cases),
                "sequences": len(dataset.sequence_cases),
            }
            report["frozen_status"][name] = dataset.frozen

        # Verify frozen status
        if datasets.get("regression") and not datasets["regression"].frozen:
            report["valid"] = False
            report["violations"].append("Regression set must be frozen")

        if datasets.get("holdout") and not datasets["holdout"].frozen:
            report["valid"] = False
            report["violations"].append("Holdout set must be frozen")

        return report
