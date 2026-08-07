from __future__ import annotations

import hashlib
import json
import sqlite3
from pathlib import Path

from pcltm.hermes_history import HermesHistoryIngestor
from pcltm.store import EventStore
from pcltm.transcript_search import search_exact_evidence


def _create_hermes_db(path: Path) -> None:
    conn = sqlite3.connect(path)
    try:
        conn.executescript(
            """
            CREATE TABLE sessions (
                id TEXT PRIMARY KEY, source TEXT NOT NULL, parent_session_id TEXT,
                started_at REAL NOT NULL, ended_at REAL, end_reason TEXT,
                archived INTEGER NOT NULL DEFAULT 0, rewind_count INTEGER NOT NULL DEFAULT 0,
                system_prompt TEXT
            );
            CREATE TABLE messages (
                id INTEGER PRIMARY KEY, session_id TEXT NOT NULL, role TEXT NOT NULL,
                content TEXT, tool_call_id TEXT, tool_calls TEXT, tool_name TEXT,
                timestamp REAL NOT NULL, token_count INTEGER, finish_reason TEXT,
                reasoning TEXT, reasoning_content TEXT, reasoning_details TEXT,
                codex_reasoning_items TEXT, codex_message_items TEXT,
                platform_message_id TEXT, observed INTEGER DEFAULT 0,
                active INTEGER NOT NULL DEFAULT 1, compacted INTEGER NOT NULL DEFAULT 0
            );
            """
        )
        conn.execute(
            "INSERT INTO sessions VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)",
            ("s1", "tui", "parent-1", 100.0, None, None, 0, 1, "private system body"),
        )
        rows = [
            (1, "s1", "system", "secret system prompt", None, None, None, 101.0, 3, None, "hidden", "hidden", "hidden", "hidden", "hidden", None, 1, 1, 0),
            (2, "s1", "developer", "secret developer prompt", None, None, None, 102.0, 3, None, "hidden", "hidden", "hidden", "hidden", "hidden", None, 1, 1, 0),
            (3, "s1", "user", "老师的原始问题", None, None, None, 103.0, 5, None, "never persist reasoning", "never persist reasoning", None, None, None, "platform-3", 1, 1, 0),
            (4, "s1", "assistant", "可见回答", None, json.dumps([{"id": "call-1", "function": {"name": "read_file"}}]), None, 104.0, 6, "tool_calls", "private chain", None, None, None, None, None, 1, 1, 0),
            (5, "s1", "tool", "工具完整结果", "call-1", None, "read_file", 105.0, 7, None, None, None, None, None, None, None, 1, 0, 1),
        ]
        conn.executemany("INSERT INTO messages VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)", rows)
        conn.commit()
    finally:
        conn.close()


def test_ingests_visible_tools_prompt_hashes_and_session_lifecycle(tmp_path: Path) -> None:
    hermes_db = tmp_path / "state.db"
    pcltm_db = tmp_path / "pcltm.db"
    _create_hermes_db(hermes_db)
    store = EventStore(pcltm_db)
    try:
        report = HermesHistoryIngestor(store, hermes_db).ingest()
        events = store.list_events(limit=20)
        ingests = [store.find_ingest_event(f"hermes-message:{index}") for index in range(1, 6)]
        lifecycle = store.find_ingest_event("hermes-session:s1")
        projection_statuses = {
            row["status"]: row["count"]
            for row in store._conn.execute(
                "SELECT status, COUNT(*) AS count FROM projection_outbox GROUP BY status"
            )
        }
        exact = search_exact_evidence(store, "老师的原始问题", limit=1)
    finally:
        store.close()

    assert report == {"scanned": 5, "inserted": 6, "updated": 0, "existing": 0, "sessions": 1}
    assert len(events) == 6
    assert projection_statuses == {"applied": 12}
    assert len(exact) == 1 and exact[0].verified is True
    by_role = {event["role"]: event for event in events}
    assert by_role["user"]["content"] == "老师的原始问题"
    assert by_role["assistant"]["content"] == "可见回答"
    assert by_role["tool"]["content"] == "工具完整结果"
    assert "secret system prompt" not in by_role["system"]["content"]
    assert hashlib.sha256("secret system prompt".encode()).hexdigest() in by_role["system"]["content"]
    assert "secret developer prompt" not in by_role["developer"]["content"]
    assert all(item is not None for item in ingests)
    assert lifecycle is not None
    assert lifecycle["payload_metadata"]["parent_session_id"] == "parent-1"
    assert lifecycle["payload_metadata"]["system_prompt"]["storage"] == "hash_only"
    user_meta = ingests[2]["payload_metadata"]
    assistant_meta = ingests[3]["payload_metadata"]
    tool_meta = ingests[4]["payload_metadata"]
    assert "reasoning" not in json.dumps(user_meta)
    assert assistant_meta["tool_calls"][0]["id"] == "call-1"
    assert tool_meta["active"] is False
    assert tool_meta["compacted"] is True
    assert tool_meta["tool_call_id"] == "call-1"


def test_reingest_is_idempotent_and_session_scoped(tmp_path: Path) -> None:
    hermes_db = tmp_path / "state.db"
    pcltm_db = tmp_path / "pcltm.db"
    _create_hermes_db(hermes_db)
    with sqlite3.connect(hermes_db) as conn:
        conn.execute("INSERT INTO sessions VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)", ("s2", "gateway", None, 200.0, None, None, 0, 0, None))
        conn.execute("INSERT INTO messages (id, session_id, role, content, timestamp) VALUES (6, 's2', 'user', 'other', 201.0)")
    store = EventStore(pcltm_db)
    try:
        first = HermesHistoryIngestor(store, hermes_db).ingest(session_id="s1")
        second = HermesHistoryIngestor(store, hermes_db).ingest(session_id="s1")
        counts = store.fts_counts()
        other = store.find_ingest_event("hermes-message:6")
    finally:
        store.close()

    assert first == {"scanned": 5, "inserted": 6, "updated": 0, "existing": 0, "sessions": 1}
    assert second == {"scanned": 5, "inserted": 0, "updated": 0, "existing": 6, "sessions": 1}
    assert counts["events"] == counts["event_fts"] == 6
    assert other is None


def test_raw_history_is_searchable_but_not_approved_prompt_memory(tmp_path: Path, monkeypatch) -> None:
    from pcltm import memory_adapter

    hermes_db = tmp_path / "state.db"
    pcltm_db = tmp_path / "pcltm.db"
    _create_hermes_db(hermes_db)
    store = EventStore(pcltm_db)
    try:
        HermesHistoryIngestor(store, hermes_db).ingest()
        hits = store.search_events("老师的原始问题", limit=10)
    finally:
        store.close()

    monkeypatch.setenv("HERMES_PCLTM_DB", str(pcltm_db))
    monkeypatch.setattr(memory_adapter, "_load_system_core_entries", lambda **kwargs: [])
    monkeypatch.setattr(memory_adapter, "enabled", lambda: True)
    rendered = memory_adapter.load_prompt_context(mode="work", query="老师的原始问题")

    assert any(hit["snippet"] == "老师的原始问题" for hit in hits)
    assert rendered == ""


def test_reingest_appends_revisions_without_overwriting_prior_events(tmp_path: Path) -> None:
    hermes_db = tmp_path / "state.db"
    pcltm_db = tmp_path / "pcltm.db"
    _create_hermes_db(hermes_db)
    store = EventStore(pcltm_db)
    try:
        HermesHistoryIngestor(store, hermes_db).ingest()
        original_lifecycle = store.find_ingest_event("hermes-session:s1")
        original_message = store.find_ingest_event("hermes-message:3")
        with sqlite3.connect(hermes_db) as conn:
            conn.execute("UPDATE sessions SET ended_at=300, end_reason='compression', rewind_count=2 WHERE id='s1'")
            conn.execute("UPDATE messages SET content='修正后的原始问题', active=0, compacted=1 WHERE id=3")
        report = HermesHistoryIngestor(store, hermes_db).ingest()
        lifecycle = store.find_ingest_event("hermes-session:s1")
        message = store.find_ingest_event("hermes-message:3")
        counts = store.fts_counts()
        all_events = store.list_events(limit=20)
    finally:
        store.close()

    assert report == {"scanned": 5, "inserted": 0, "updated": 2, "existing": 4, "sessions": 1}
    assert original_lifecycle is not None
    assert original_message is not None
    assert lifecycle is not None
    assert message is not None
    assert lifecycle["event_id"] != original_lifecycle["event_id"]
    assert message["event_id"] != original_message["event_id"]
    assert lifecycle["payload_metadata"]["end_reason"] == "compression"
    assert lifecycle["payload_metadata"]["rewind_count"] == 2
    assert message["payload_metadata"]["active"] is False
    assert message["payload_metadata"]["compacted"] is True
    assert counts["events"] == counts["event_fts"] == 8
    assert any(event["event_id"] == original_message["event_id"] and event["content"] == "老师的原始问题" for event in all_events)
    assert any(event["event_id"] == message["event_id"] and event["content"] == "修正后的原始问题" for event in all_events)


def test_ingest_releases_hermes_db_file_handle(tmp_path: Path) -> None:
    """Windows regression gate: ingest() must close the read-only hermes DB connection.

    `with sqlite3.connect(...) as conn` only commits/rolls back — it does NOT close
    the connection. On Windows the held handle makes tempdir cleanup fail with
    PermissionError. The connection must be closed explicitly in finally.
    """
    hermes_db = tmp_path / "state.db"
    pcltm_db = tmp_path / "pcltm.db"
    _create_hermes_db(hermes_db)
    store = EventStore(pcltm_db)
    try:
        HermesHistoryIngestor(store, hermes_db).ingest()
    finally:
        store.close()
    # If the read-only connection is still open, unlinking fails on Windows
    # (and on POSIX the lock is advisory, so this is a Windows-specific gate).
    hermes_db.unlink()
    assert not hermes_db.exists()
