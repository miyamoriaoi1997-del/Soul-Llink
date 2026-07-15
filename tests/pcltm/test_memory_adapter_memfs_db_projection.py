from __future__ import annotations

import json
import sqlite3
from pathlib import Path

import pytest

from pcltm import memory_adapter


SCHEMA = """
CREATE TABLE memory_records (
    record_id INTEGER PRIMARY KEY,
    candidate_id TEXT NOT NULL,
    kind TEXT NOT NULL,
    target_file TEXT NOT NULL,
    content TEXT NOT NULL,
    confidence REAL NOT NULL,
    sensitivity TEXT NOT NULL,
    source_event_ids TEXT NOT NULL,
    source_node_ids TEXT NOT NULL,
    status TEXT NOT NULL,
    metadata TEXT NOT NULL,
    created_at TEXT
)
"""


def _insert(con: sqlite3.Connection, record_id: int, content: str, status: str = "approved") -> None:
    con.execute(
        """INSERT INTO memory_records
           VALUES (?, ?, 'fact', 'USER.md', ?, 1.0, 'normal', '[]', '[]', ?, ?,
                   '2026-01-01T00:00:00Z')""",
        (record_id, f"candidate-{record_id}", content, status, json.dumps({"buckets": ["runtime_boundary"]})),
    )


def _write_memfs(root: Path, name: str, body: str, *, record_id: int | None = None, candidate_id: str | None = None) -> None:
    path = root / "pinned" / name
    path.parent.mkdir(parents=True, exist_ok=True)
    identity = ((f"  record_id: {record_id}\n" if record_id is not None else "") + (f"  candidate_id: {candidate_id}\n" if candidate_id is not None else ""))
    metadata = f"metadata:\n{identity}  target_file: USER.md\n" if identity else ""
    path.write_text(
        "---\n"
        "description: projection fixture\n"
        "authority: pinned\n"
        "mode_scope: [work]\n"
        "buckets: [runtime_boundary]\n"
        f"{metadata}"
        "---\n"
        f"{body}\n",
        encoding="utf-8",
    )


@pytest.fixture
def projected_context(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> tuple[Path, Path]:
    db = tmp_path / "pcltm.db"
    con = sqlite3.connect(db)
    try:
        con.execute(SCHEMA)
        _insert(con, 1, "projected approved record")
        _insert(con, 2, "approved record missing from projection")
        _insert(con, 3, "stale retired projection", "retired")
        con.commit()
    finally:
        con.close()
    root = tmp_path / "memfs"
    _write_memfs(root, "record-one.md", "projected approved record", record_id=1)
    _write_memfs(root, "record-three.md", "stale retired projection", record_id=3)
    _write_memfs(root, "local-contract.md", "MemFS-only legitimate pinned contract")
    monkeypatch.setattr(memory_adapter, "DEFAULT_DB", db)
    monkeypatch.setenv("HERMES_PCLTM_DB", str(db))
    monkeypatch.setattr(memory_adapter, "MEMFS_ROOT", root)
    monkeypatch.setattr(memory_adapter, "_semantic_scores_for_query", lambda *args, **kwargs: {})
    return db, root


def test_partial_memfs_projection_is_completed_and_governed_by_db(projected_context: tuple[Path, Path]) -> None:
    view = memory_adapter.load_layered_prompt_context(mode="work", budgets={"pinned": 5000})
    bodies = [item.body.strip() for item in view.pinned.items]

    assert bodies.count("projected approved record") == 1
    assert "approved record missing from projection" in bodies
    assert "stale retired projection" not in bodies
    assert "MemFS-only legitimate pinned contract" in bodies
    assert view.selection_source == "memfs+db"


def test_claimed_but_missing_db_identity_is_discarded_while_memfs_only_contract_survives(projected_context: tuple[Path, Path]) -> None:
    _db, root = projected_context
    _write_memfs(root, "missing-record.md", "must not survive by missing record", record_id=999)
    _write_memfs(root, "missing-candidate.md", "must not survive by missing candidate", candidate_id="candidate-missing")
    view = memory_adapter.load_layered_prompt_context(mode="work", budgets={"pinned": 5000})
    bodies = [item.body.strip() for item in view.pinned.items]
    assert "must not survive by missing record" not in bodies
    assert "must not survive by missing candidate" not in bodies
    assert "MemFS-only legitimate pinned contract" in bodies


def test_dual_identity_must_resolve_to_same_approved_row(projected_context: tuple[Path, Path]) -> None:
    _db, root = projected_context
    _write_memfs(root, "conflict.md", "conflicting dual identity", record_id=1, candidate_id="candidate-2")
    _write_memfs(root, "missing-half.md", "missing candidate half", record_id=1, candidate_id="candidate-missing")
    bodies = [item.body.strip() for item in memory_adapter.load_layered_prompt_context(mode="work", budgets={"pinned": 5000}).pinned.items]
    assert "conflicting dual identity" not in bodies
    assert "missing candidate half" not in bodies


def test_candidate_only_projection_is_not_duplicated(projected_context: tuple[Path, Path]) -> None:
    _db, root = projected_context
    _write_memfs(root, "candidate-two.md", "approved record missing from projection", candidate_id="candidate-2")
    bodies = [item.body.strip() for item in memory_adapter.load_layered_prompt_context(mode="work", budgets={"pinned": 5000}).pinned.items]
    assert bodies.count("approved record missing from projection") == 1
