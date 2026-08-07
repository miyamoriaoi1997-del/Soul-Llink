from __future__ import annotations

import sqlite3

import pytest

from pcltm.store import EventStore


def test_missing_event_fts_table_is_not_silently_like_fallback(tmp_path) -> None:
    store = EventStore(tmp_path / "pcltm.db")
    try:
        store._conn.execute("DROP TABLE event_fts")
        with pytest.raises(sqlite3.OperationalError, match="no such table"):
            store.search_events("anything")
    finally:
        store.close()


def test_missing_summary_fts_table_is_not_silently_like_fallback(tmp_path) -> None:
    store = EventStore(tmp_path / "pcltm.db")
    try:
        store._conn.execute("DROP TABLE summary_fts")
        with pytest.raises(sqlite3.OperationalError, match="no such table"):
            store.search_summaries("anything")
    finally:
        store.close()
