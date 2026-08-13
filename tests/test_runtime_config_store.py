from __future__ import annotations

from pathlib import Path

import pytest
import yaml

from soul_link.runtime_config import RuntimeConfigStore


def test_save_migrates_n_minus_one_and_returns_hash_bound_receipt(tmp_path: Path) -> None:
    path = tmp_path / "config.yaml"
    path.write_text("plugins:\n  entries:\n    soullink:\n      state_machine: {}\n", encoding="utf-8")
    store = RuntimeConfigStore(path)

    receipt = store.save(store.load())
    persisted = yaml.safe_load(path.read_text(encoding="utf-8"))

    assert persisted["plugins"]["entries"]["soullink"]["schema_version"] == 1
    assert receipt.schema_from == 0
    assert receipt.schema_to == 1
    assert receipt.before_sha256 != receipt.after_sha256
    assert receipt.backup_path.is_file()


def test_rollback_restores_exact_previous_config(tmp_path: Path) -> None:
    path = tmp_path / "config.yaml"
    original = "model:\n  default: original\n"
    path.write_text(original, encoding="utf-8")
    store = RuntimeConfigStore(path)
    config = store.load()
    config["model"]["default"] = "updated"

    receipt = store.save(config)
    store.rollback(receipt)

    assert path.read_text(encoding="utf-8") == original
    assert not receipt.backup_path.exists()


def test_rollback_rejects_config_modified_after_receipt(tmp_path: Path) -> None:
    path = tmp_path / "config.yaml"
    path.write_text("model: {}\n", encoding="utf-8")
    store = RuntimeConfigStore(path)
    receipt = store.save(store.load())
    path.write_text("model:\n  default: tampered\n", encoding="utf-8")

    with pytest.raises(RuntimeError, match="current config hash mismatch"):
        store.rollback(receipt)


def test_rollback_removes_config_created_from_missing_state(tmp_path: Path) -> None:
    path = tmp_path / "config.yaml"
    store = RuntimeConfigStore(path)

    receipt = store.save({})
    store.rollback(receipt)

    assert not path.exists()
