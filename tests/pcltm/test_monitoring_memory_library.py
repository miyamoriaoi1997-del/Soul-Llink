from __future__ import annotations

import sqlite3
from pathlib import Path

from pcltm.cli import init_runtime
from pcltm.monitoring.memory_library import collect_memory_library_stats


def test_memory_library_stats_counts_only_persistent_entities(tmp_path: Path) -> None:
    db = tmp_path / "runtime.db"
    memfs = tmp_path / "memfs"
    init_runtime(db_path=db, memfs_root=memfs)
    with sqlite3.connect(db) as connection:
        event_columns = [row[1] for row in connection.execute("PRAGMA table_info(events)")]
        chunk_columns = [row[1] for row in connection.execute("PRAGMA table_info(event_chunks)")]
        assert event_columns and chunk_columns
        connection.execute(
            "INSERT INTO memory_records(candidate_id,kind,target_file,content,confidence,sensitivity,source_event_ids,source_node_ids,status,metadata) VALUES(?,?,?,?,?,?,?,?,?,?)",
            ("c1", "preference", "USER.md", "opaque test body", 1.0, "normal", "[]", "[]", "approved", "{}"),
        )

    stats = collect_memory_library_stats(db)

    assert stats["event_count"] == 0
    assert stats["active_memory_count"] == 0
    assert stats["active_event_derived_count"] == 0
    assert stats["active_other_lineage_count"] == 0
    assert stats["derived_memory_count"] == 1
    assert stats["persistent_memory_total"] == 1
    assert stats["evidence_chunk_count"] == 0
    assert stats["provenance"]["counting_rule"] == "persistent_memory_total = events + memory_records; event_chunks excluded"


def test_memory_library_stats_empty_database_is_zero_and_read_only(tmp_path: Path) -> None:
    db = tmp_path / "runtime.db"
    memfs = tmp_path / "memfs"
    init_runtime(db_path=db, memfs_root=memfs)
    before = db.read_bytes()

    stats = collect_memory_library_stats(db)

    assert stats["event_count"] == 0
    assert stats["active_memory_count"] == 0
    assert stats["active_event_derived_count"] == 0
    assert stats["derived_memory_count"] == 0
    assert stats["persistent_memory_total"] == 0
    assert stats["evidence_chunk_count"] == 0
    assert db.read_bytes() == before
