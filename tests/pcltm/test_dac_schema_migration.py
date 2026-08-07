from __future__ import annotations

import sqlite3

from pcltm.store import EventStore


def test_event_store_does_not_migrate_legacy_dac_context_snapshot_schema(tmp_path) -> None:
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
    store.close()
    assert db_path.read_bytes() != b""
    with sqlite3.connect(db_path) as conn:
        columns = {row[1] for row in conn.execute("PRAGMA table_info(dac_context_snapshots)")}
    assert columns == {"snapshot_id", "session_id", "budget_tokens", "selected_node_ids", "metadata", "created_at"}
