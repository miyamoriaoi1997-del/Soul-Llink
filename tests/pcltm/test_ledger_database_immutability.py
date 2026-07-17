from __future__ import annotations

import sqlite3
from pathlib import Path

import pytest

from pcltm.store import EventStore


def test_authoritative_events_and_revisions_reject_update_and_delete(tmp_path: Path) -> None:
    store = EventStore(tmp_path / "pcltm.db")
    try:
        event_id, _ = store.ingest_external_event(
            external_id="source:1", source_hash="hash", kind="chat_message",
            session_id="s", conversation_id="c", platform="desktop",
            role="user", source="test", content="immutable",
            category="raw_conversation", subcategory="user", inject_policy="retrieve_only",
        )
        with pytest.raises(sqlite3.IntegrityError, match="events are immutable"):
            store._conn.execute("UPDATE events SET content='changed' WHERE event_id=?", (event_id,))
        store._conn.rollback()
        with pytest.raises(sqlite3.IntegrityError, match="events are immutable"):
            store._conn.execute("DELETE FROM events WHERE event_id=?", (event_id,))
        store._conn.rollback()
        with pytest.raises(sqlite3.IntegrityError, match="event revisions are immutable"):
            store._conn.execute("UPDATE event_revisions SET source_hash='changed' WHERE event_id=?", (event_id,))
        store._conn.rollback()
        with pytest.raises(sqlite3.IntegrityError, match="event revisions are immutable"):
            store._conn.execute("DELETE FROM event_revisions WHERE event_id=?", (event_id,))
        store._conn.rollback()
        event = store.get_event(event_id)
    finally:
        store.close()

    assert event["content"] == "immutable"
