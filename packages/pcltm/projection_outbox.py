"""Transactional projection-outbox helpers."""

from __future__ import annotations

import sqlite3


TRANSCRIPT_PROJECTIONS = ("transcript_chunks", "transcript_fts")


def enqueue_event_projections(
    conn: sqlite3.Connection,
    *,
    event_id: int,
    aggregate_version: int,
    payload_sha256: str,
) -> None:
    """Enqueue rebuildable transcript projections in the caller transaction."""
    for projection_kind in TRANSCRIPT_PROJECTIONS:
        conn.execute(
            """
            INSERT OR IGNORE INTO projection_outbox (
                event_seq, projection_kind, aggregate_id, aggregate_version,
                payload_sha256, status
            ) VALUES (?, ?, ?, ?, ?, 'pending')
            """,
            (
                event_id,
                projection_kind,
                str(event_id),
                int(aggregate_version),
                payload_sha256,
            ),
        )
