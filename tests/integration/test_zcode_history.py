from __future__ import annotations

import json
import sqlite3
from pathlib import Path

import pytest
from pcltm.store import EventStore
from pcltm.zcode_history import ZCodeHistoryIngestor


def _make_zcode_db(path: Path) -> Path:
    conn = sqlite3.connect(path)
    conn.executescript(
        """
        CREATE TABLE message (
            id INTEGER PRIMARY KEY,
            session_id TEXT,
            time_created TEXT,
            time_updated TEXT,
            data TEXT,
            sequence INTEGER
        );
        CREATE TABLE part (
            id INTEGER PRIMARY KEY,
            message_id INTEGER,
            session_id TEXT,
            time_created TEXT,
            time_updated TEXT,
            data TEXT,
            sequence INTEGER
        );
        """
    )
    conn.execute(
        "INSERT INTO message (id, session_id, data, sequence) VALUES (?, ?, ?, ?)",
        (1, "sess-1", json.dumps({"role": "user", "time": "2026-01-01T00:00:00Z"}), 0),
    )
    conn.execute(
        "INSERT INTO message (id, session_id, data, sequence) VALUES (?, ?, ?, ?)",
        (2, "sess-1", json.dumps({"role": "assistant", "model": "m", "time": "2026-01-01T00:00:01Z"}), 1),
    )
    conn.execute(
        "INSERT INTO message (id, session_id, data, sequence) VALUES (?, ?, ?, ?)",
        (3, "sess-2", json.dumps({"role": "user"}), 0),
    )
    conn.execute(
        "INSERT INTO part (id, message_id, session_id, data, sequence) VALUES (?, ?, ?, ?, ?)",
        (1, 1, "sess-1", json.dumps({"type": "text", "text": "hello soul"}), 0),
    )
    conn.execute(
        "INSERT INTO part (id, message_id, session_id, data, sequence) VALUES (?, ?, ?, ?, ?)",
        (2, 2, "sess-1", json.dumps({"type": "text", "text": "hi there"}), 0),
    )
    conn.execute(
        "INSERT INTO part (id, message_id, session_id, data, sequence) VALUES (?, ?, ?, ?, ?)",
        (3, 1, "sess-1", json.dumps({"type": "timeline", "timelineType": "model_change"}), 1),
    )
    conn.commit()
    conn.close()
    return path


def test_ingest_zcode_history_into_pcltm(tmp_path: Path) -> None:
    zcode_db = _make_zcode_db(tmp_path / "db.sqlite")
    store = EventStore(tmp_path / "pcltm.db")
    try:
        result = ZCodeHistoryIngestor(store, zcode_db).ingest()
        assert result["scanned"] == 3
        assert result["inserted"] == 3
        first = store.find_ingest_event("zcode-message:1")
        assert first is not None
        assert first["payload_metadata"]["zcode_message_id"] == 1
        from pcltm.transcript_search import search_exact_evidence

        exact = search_exact_evidence(store, "hello soul", limit=1)
        assert len(exact) == 1
        assert exact[0].verified is True
    finally:
        store.close()


def test_ingest_is_idempotent(tmp_path: Path) -> None:
    zcode_db = _make_zcode_db(tmp_path / "db.sqlite")
    store = EventStore(tmp_path / "pcltm.db")
    try:
        first = ZCodeHistoryIngestor(store, zcode_db).ingest()
        second = ZCodeHistoryIngestor(store, zcode_db).ingest()
        assert second["inserted"] == 0
        assert second["existing"] >= first["scanned"]
    finally:
        store.close()


def test_ingest_scoped_to_session(tmp_path: Path) -> None:
    zcode_db = _make_zcode_db(tmp_path / "db.sqlite")
    store = EventStore(tmp_path / "pcltm.db")
    try:
        result = ZCodeHistoryIngestor(store, zcode_db).ingest(session_id="sess-2")
        assert result["scanned"] == 1
    finally:
        store.close()


def test_ingest_rejects_missing_database(tmp_path: Path) -> None:
    store = EventStore(tmp_path / "pcltm.db")
    try:
        with pytest.raises(FileNotFoundError, match="missing"):
            ZCodeHistoryIngestor(store, tmp_path / "nope.sqlite").ingest()
    finally:
        store.close()


def test_ingest_rejects_unsupported_schema(tmp_path: Path) -> None:
    conn = sqlite3.connect(tmp_path / "bad.sqlite")
    conn.execute("CREATE TABLE message (id INTEGER PRIMARY KEY)")
    conn.commit()
    conn.close()
    store = EventStore(tmp_path / "pcltm.db")
    try:
        with pytest.raises(RuntimeError, match="unsupported"):
            ZCodeHistoryIngestor(store, tmp_path / "bad.sqlite").ingest()
    finally:
        store.close()
