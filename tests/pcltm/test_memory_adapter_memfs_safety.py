from __future__ import annotations

import importlib
import sqlite3

from pcltm import memory_adapter


def test_configured_memfs_root_rejects_env_escape(monkeypatch):
    monkeypatch.setenv("HERMES_PCLTM_MEMFS_ROOT", "/tmp/not-soul-link-memfs")
    reloaded = importlib.reload(memory_adapter)
    try:
        assert reloaded.MEMFS_ROOT == reloaded.DEFAULT_MEMFS_ROOT
    finally:
        importlib.reload(memory_adapter)


def test_safe_memfs_record_path_stays_under_monkeypatched_root(tmp_path, monkeypatch):
    root = tmp_path / "memfs"
    monkeypatch.setattr(memory_adapter, "MEMFS_ROOT", root)

    path = memory_adapter._safe_memfs_record_path(None, "pinned/user-000001-safe.md")

    assert path == root.resolve() / "pinned" / "user-000001-safe.md"


def test_safe_memfs_record_path_rejects_relative_escape(tmp_path, monkeypatch):
    monkeypatch.setattr(memory_adapter, "MEMFS_ROOT", tmp_path / "memfs")

    try:
        memory_adapter._safe_memfs_record_path(None, "../escape.md")
    except ValueError as exc:
        assert "unsafe MemFS relative path" in str(exc)
    else:  # pragma: no cover - defensive assertion message
        raise AssertionError("expected MemFS escape path to be rejected")



def test_memfs_record_file_path_matches_materialized_path(tmp_path, monkeypatch):
    monkeypatch.setattr(memory_adapter, "MEMFS_ROOT", tmp_path / "memfs")
    row = {
        "record_id": 42,
        "target_file": "USER.md",
        "content": "Stable literal memory",
        "reviewed_at": "2026-01-01T00:00:00Z",
        "created_at": "2026-01-01T00:00:00Z",
        "metadata_json": "{}",
        "kind": "user_preference",
    }

    assert memory_adapter._materialize_memfs_record(row)
    path = memory_adapter._memfs_record_file_path(row)
    assert path.exists()

    memory_adapter._remove_memfs_record_file(row)

    assert not path.exists()


def test_superseded_lookup_treats_like_wildcards_literally(tmp_path):
    db = tmp_path / "pcltm.db"
    con = sqlite3.connect(db)
    con.row_factory = sqlite3.Row
    try:
        con.execute(
            """
            CREATE TABLE memory_records (
                record_id INTEGER PRIMARY KEY,
                content TEXT NOT NULL,
                target_file TEXT NOT NULL,
                kind TEXT NOT NULL,
                status TEXT NOT NULL,
                source TEXT NOT NULL,
                decision_reason TEXT NOT NULL
            )
            """
        )
        con.execute(
            """
            INSERT INTO memory_records(content, target_file, kind, status, source, decision_reason)
            VALUES (?, 'USER.md', 'user_preference', 'superseded', 'test', 'literal')
            """,
            ("literal 100%_match",),
        )
        con.execute(
            """
            INSERT INTO memory_records(content, target_file, kind, status, source, decision_reason)
            VALUES (?, 'USER.md', 'user_preference', 'superseded', 'test', 'wildcard')
            """,
            ("literal 100XAmatch",),
        )
        con.commit()

        rows = memory_adapter._fetch_superseded_records(con, "USER.md", "100%_match")

        assert [row["content"] for row in rows] == ["literal 100%_match"]
    finally:
        con.close()
