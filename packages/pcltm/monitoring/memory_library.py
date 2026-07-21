"""Authoritative persistent-memory library statistics for the read-only monitor."""

from __future__ import annotations

import sqlite3
from pathlib import Path
from typing import Any
from urllib.parse import quote

from .sqlite_snapshot import stable_sqlite_snapshot


def _readonly_connection(db_path: Path) -> sqlite3.Connection:
    normalized = str(db_path.resolve()).replace("\\", "/")
    connection = sqlite3.connect(
        f"file:{quote(normalized, safe='/:')}?mode=ro", uri=True, timeout=0.25
    )
    connection.execute("PRAGMA query_only=ON")
    return connection


def collect_memory_library_stats(db_path: str | Path) -> dict[str, Any]:
    """Count persistent-memory entities from one stable read-only SQLite snapshot.

    ``events`` are canonical original events and ``memory_records`` are derived
    governed memories. ``event_chunks`` are evidence projections and deliberately
    remain a separate metric so they cannot inflate the persistent-memory total.
    """
    db = Path(db_path)
    with stable_sqlite_snapshot(db) as stable_db:
        connection = _readonly_connection(stable_db)
        try:
            tables = {str(row[0]) for row in connection.execute(
                "SELECT name FROM sqlite_master WHERE type IN ('table', 'view')"
            )}

            def count(name: str) -> int:
                if name not in tables:
                    return 0
                row = connection.execute(f'SELECT count(*) FROM "{name}"').fetchone()
                return int(row[0] if row else 0)

            event_count = count("events")
            derived_memory_count = count("memory_records")
            evidence_chunk_count = count("event_chunks")
        finally:
            connection.close()
    return {
        "source": "stable_sqlite_snapshot",
        "event_count": event_count,
        "derived_memory_count": derived_memory_count,
        "persistent_memory_total": event_count + derived_memory_count,
        "evidence_chunk_count": evidence_chunk_count,
        "provenance": {
            "database": "pcltm_runtime_db",
            "snapshot": "stable read-only SQLite DB+WAL copy",
            "event_table": "events",
            "derived_memory_table": "memory_records",
            "evidence_table": "event_chunks",
            "counting_rule": "persistent_memory_total = events + memory_records; event_chunks excluded",
        },
    }


__all__ = ["collect_memory_library_stats"]
