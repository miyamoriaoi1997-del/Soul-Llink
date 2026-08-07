"""Transactional projection-outbox helpers."""

from __future__ import annotations

import sqlite3


TRANSCRIPT_PROJECTIONS = ("transcript_chunks", "transcript_fts")
MEMORY_PROJECTIONS = ("memory_fts", "memory_memfs")


def require_event_projection_authority(job: dict[str, object]) -> int:
    """Return the event id only for an internally consistent event job."""

    if type(job.get("authority_kind")) is not str or job.get("authority_kind") != "event":
        raise ValueError("transcript projection authority mismatch")
    raw_event_seq = job.get("event_seq")
    if raw_event_seq is None:
        raise ValueError("transcript projection authority mismatch")
    try:
        if type(raw_event_seq) is int:
            event_id = raw_event_seq
        elif (
            type(raw_event_seq) is str
            and raw_event_seq.isascii()
            and raw_event_seq.isdecimal()
            and raw_event_seq == str(int(raw_event_seq))
        ):
            event_id = int(raw_event_seq)
        else:
            raise ValueError
        if event_id <= 0:
            raise ValueError
        authority_id = job.get("authority_id")
        if (
            type(authority_id) is not str
            or not authority_id.isascii()
            or not authority_id.isdecimal()
            or authority_id != str(event_id)
        ):
            raise ValueError
    except (TypeError, ValueError) as exc:
        raise ValueError("transcript projection authority mismatch") from exc
    return event_id


def _enqueue(
    conn: sqlite3.Connection,
    *,
    event_id: int | None,
    authority_kind: str,
    authority_id: str,
    aggregate_id: str,
    aggregate_version: int,
    payload_sha256: str,
    projection_kinds: tuple[str, ...],
) -> None:
    for projection_kind in projection_kinds:
        existing = conn.execute(
            """SELECT event_seq, authority_kind, authority_id, payload_sha256
               FROM projection_outbox
               WHERE projection_kind=? AND aggregate_id=? AND aggregate_version=?""",
            (projection_kind, aggregate_id, int(aggregate_version)),
        ).fetchone()
        commitment = (event_id, authority_kind, authority_id, payload_sha256)
        if existing is not None:
            persisted = (
                existing["event_seq"], str(existing["authority_kind"]),
                str(existing["authority_id"]), str(existing["payload_sha256"]),
            )
            if persisted != commitment:
                raise ValueError("projection commitment conflict")
            continue
        try:
            conn.execute(
                """
                INSERT INTO projection_outbox (
                    event_seq, authority_kind, authority_id, projection_kind,
                    aggregate_id, aggregate_version, payload_sha256, status
                ) VALUES (?, ?, ?, ?, ?, ?, ?, 'pending')
                """,
                (
                    event_id, authority_kind, authority_id, projection_kind,
                    aggregate_id, int(aggregate_version), payload_sha256,
                ),
            )
        except sqlite3.IntegrityError:
            raced = conn.execute(
                """SELECT event_seq, authority_kind, authority_id, payload_sha256
                   FROM projection_outbox
                   WHERE projection_kind=? AND aggregate_id=? AND aggregate_version=?""",
                (projection_kind, aggregate_id, int(aggregate_version)),
            ).fetchone()
            if raced is None or (
                raced["event_seq"], str(raced["authority_kind"]),
                str(raced["authority_id"]), str(raced["payload_sha256"]),
            ) != commitment:
                raise ValueError("projection commitment conflict") from None


def enqueue_event_projections(
    conn: sqlite3.Connection,
    *,
    event_id: int,
    aggregate_version: int,
    payload_sha256: str,
) -> None:
    """Enqueue rebuildable transcript projections in the caller transaction."""
    _enqueue(
        conn,
        event_id=event_id,
        authority_kind="event",
        authority_id=str(event_id),
        aggregate_id=str(event_id),
        aggregate_version=aggregate_version,
        payload_sha256=payload_sha256,
        projection_kinds=TRANSCRIPT_PROJECTIONS,
    )


def enqueue_memory_projections(
    conn: sqlite3.Connection,
    *,
    event_id: int | None,
    aggregate_id: str,
    aggregate_version: int,
    payload_sha256: str,
    authority_kind: str = "event",
    authority_id: str | None = None,
) -> None:
    """Enqueue memory projections without materializing any projection."""
    if authority_id is None or not authority_id.strip():
        raise ValueError("memory projection authority_id is required")
    _enqueue(
        conn,
        event_id=event_id,
        authority_kind=authority_kind,
        authority_id=authority_id,
        aggregate_id=aggregate_id,
        aggregate_version=aggregate_version,
        payload_sha256=payload_sha256,
        projection_kinds=MEMORY_PROJECTIONS,
    )
