from __future__ import annotations

import sqlite3

from pcltm.dac import DACStore
from pcltm.store import EventStore


def test_event_store_migrates_legacy_dac_context_snapshot_schema(tmp_path) -> None:
    db_path = tmp_path / "legacy-pcltm.db"
    conn = sqlite3.connect(db_path)
    conn.execute(
        """
        CREATE TABLE dac_context_snapshots (
            snapshot_id INTEGER PRIMARY KEY AUTOINCREMENT,
            session_id TEXT NOT NULL,
            budget_tokens INTEGER DEFAULT 0,
            selected_node_ids TEXT NOT NULL DEFAULT '[]',
            metadata TEXT NOT NULL DEFAULT '{}',
            created_at REAL NOT NULL
        )
        """
    )
    conn.commit()
    conn.close()

    store = EventStore(db_path)
    columns = {row[1] for row in store._conn.execute("PRAGMA table_info(dac_context_snapshots)")}

    assert {
        "turn_id",
        "mode",
        "snapshot_type",
        "selected_raw_ids",
        "fresh_tail_count",
    } <= columns

    snapshot_id = DACStore(store).add_context_snapshot(
        session_id="session-1",
        turn_id="turn-1",
        mode="active_prompt",
        budget_tokens=123,
        selected_node_ids=[1, 2],
        selected_raw_ids=[3],
        fresh_tail_count=4,
        metadata={"source": "test"},
    )

    snapshot = DACStore(store).get_context_snapshot(snapshot_id)
    assert snapshot["turn_id"] == "turn-1"
    assert snapshot["mode"] == "active_prompt"
    assert snapshot["snapshot_type"] == "dac_active_prompt"
    assert snapshot["selected_node_ids"] == [1, 2]
    assert snapshot["selected_raw_ids"] == [3]
    assert snapshot["fresh_tail_count"] == 4
    assert snapshot["metadata"]["source"] == "test"
    assert snapshot["metadata"]["snapshot_type"] == "dac_active_prompt"
