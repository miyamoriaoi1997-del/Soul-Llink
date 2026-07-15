from __future__ import annotations

import logging
import os
import sqlite3

import pcltm.memory_adapter as memory_adapter
from pcltm.semantic_index import SemanticIndex, get_index
from pcltm.store import EventStore


def _add_memory(db, content: str) -> int:
    store = EventStore(db)
    try:
        result = store.add_memory_record(
            candidate_id=f"candidate-{content}",
            kind="project_fact",
            target_file="MEMORY.md",
            content=content,
            confidence=0.9,
            sensitivity="normal",
            source_event_ids=[],
            source_node_ids=[],
            status="approved",
            metadata={},
        )
        return result[0]
    finally:
        store.close()


def test_semantic_index_query_executes_bm25_and_ranks_match(tmp_path) -> None:
    db = tmp_path / "query.db"
    matching_id = _add_memory(db, "project uses pytest verification workflow")
    _add_memory(db, "persona enjoys watercolor painting")

    results = SemanticIndex(db).query("pytest verification", min_score=0.0)

    assert results
    assert results[0][0] == matching_id
    assert results[0][1] > 0


def test_get_index_isolated_by_resolved_database_path(tmp_path) -> None:
    db = tmp_path / "one.db"
    _add_memory(db, "first database memory")

    relative = db.parent / "." / db.name
    first = get_index(relative)
    same = get_index(db.resolve())

    assert same is first
    assert same.db_path == db.resolve()


def test_get_index_rebuilds_when_same_database_changes(tmp_path) -> None:
    db = tmp_path / "changing.db"
    _add_memory(db, "initial memory")
    first = get_index(db)
    assert len(first.records) == 1

    _add_memory(db, "newly inserted searchable memory")
    refreshed = get_index(db)

    assert refreshed is not first
    assert len(refreshed.records) == 2
    assert refreshed.query("newly inserted", min_score=0.0)


def test_get_index_rebuilds_when_sqlite_content_changes_with_restored_stat(tmp_path) -> None:
    db = tmp_path / "restored-stat.db"
    record_id = _add_memory(db, "alpha searchable memory")
    first = get_index(db)
    original = db.stat()
    con = sqlite3.connect(db)
    con.execute("UPDATE memory_records SET content = ? WHERE record_id = ?", ("bravo searchable memory", record_id))
    con.commit()
    con.close()
    assert db.stat().st_size == original.st_size
    os.utime(db, ns=(original.st_atime_ns, original.st_mtime_ns))
    refreshed = get_index(db)
    assert refreshed is not first
    assert refreshed.records[0].content == "bravo searchable memory"


def test_memory_adapter_uses_active_database_path(monkeypatch, tmp_path) -> None:
    db = tmp_path / "adapter.db"
    seen = []

    class FakeIndex:
        def query(self, *args, **kwargs):
            return []

    monkeypatch.setattr(memory_adapter, "db_path", lambda: db)
    monkeypatch.setattr("pcltm.semantic_index.get_index", lambda path: seen.append(path) or FakeIndex())

    assert memory_adapter._semantic_scores_for_query("verification") == {}
    assert seen == [db]


def test_memory_adapter_reports_semantic_query_failure(monkeypatch, tmp_path, caplog) -> None:
    db = tmp_path / "adapter.db"
    monkeypatch.setattr(memory_adapter, "db_path", lambda: db)
    monkeypatch.setattr("pcltm.semantic_index.get_index", lambda path: (_ for _ in ()).throw(RuntimeError("bm25 exploded")))

    with caplog.at_level(logging.WARNING):
        assert memory_adapter._semantic_scores_for_query("verification") == {}

    assert "bm25 exploded" in caplog.text
