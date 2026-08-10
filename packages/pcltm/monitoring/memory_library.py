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

    ``events`` are canonical original evidence. The user-facing durable-memory
    inventory is the governed ``memory_current`` projection, not the legacy
    ``memory_records`` candidate table. Lineage is counted only as a breakdown
    of active claims so it cannot be mistaken for another lifecycle stage.
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
            active_memory_count = 0
            active_event_derived_count = 0
            if "memory_current" in tables:
                active_memory_count = int(connection.execute(
                    "SELECT count(*) FROM memory_current WHERE lifecycle_state = 'active'"
                ).fetchone()[0])
            if {"memory_current", "memory_claim_versions"}.issubset(tables):
                active_event_derived_count = int(connection.execute(
                    """SELECT count(*) FROM memory_current mc
                       JOIN memory_claim_versions v
                         ON v.claim_version_id = mc.claim_version_id
                       WHERE mc.lifecycle_state = 'active'
                         AND v.lineage_kind = 'event_derived'"""
                ).fetchone()[0])
        finally:
            connection.close()
    return {
        "source": "stable_sqlite_snapshot",
        "event_count": event_count,
        "active_memory_count": active_memory_count,
        "active_event_derived_count": active_event_derived_count,
        "active_other_lineage_count": active_memory_count - active_event_derived_count,
        "derived_memory_count": derived_memory_count,
        "persistent_memory_total": event_count + derived_memory_count,
        "evidence_chunk_count": evidence_chunk_count,
        "provenance": {
            "database": "pcltm_runtime_db",
            "snapshot": "stable read-only SQLite DB+WAL copy",
            "event_table": "events",
            "active_table": "memory_current",
            "lineage_table": "memory_claim_versions",
            "derived_memory_table": "memory_records",
            "evidence_table": "event_chunks",
            "counting_rule": "persistent_memory_total = events + memory_records; event_chunks excluded",
            "active_counting_rule": "active_memory_count = memory_current where lifecycle_state = active; lineage is a breakdown of active claims",
        },
    }


__all__ = ["collect_memory_library_stats"]
