"""Bounded synchronous drain for transcript projections at host lifecycle boundaries."""

from __future__ import annotations

from datetime import UTC, datetime, timedelta
from typing import Any

from ..store import EventStore
from ..projection_outbox import TRANSCRIPT_PROJECTIONS, require_event_projection_authority
from .transcript_chunks import TranscriptChunkProjector


def _timestamp_pair() -> tuple[str, str]:
    now = datetime.now(UTC)
    return (
        now.isoformat().replace("+00:00", "Z"),
        (now + timedelta(minutes=1)).isoformat().replace("+00:00", "Z"),
    )


def _apply_fts_job(store: EventStore, job: dict[str, Any], *, worker_id: str, now: str) -> None:
    event_id = require_event_projection_authority(job)
    event = store.get_event(event_id)
    if event["payload_sha256"] != job["payload_sha256"]:
        raise ValueError("projection payload hash is stale")
    row = store._conn.execute(
        "SELECT content FROM event_fts WHERE rowid = ?",
        (event_id,),
    ).fetchone()
    if row is None or str(row["content"]) != str(event["content"]):
        raise ValueError("transcript FTS projection is missing or stale")
    acked = store.ack_projection_job(
        int(job["outbox_id"]), worker_id=worker_id,
        expected_attempt_count=int(job["attempt_count"]), now=now,
    )
    if not acked:
        raise RuntimeError("projection lease ownership lost")


def drain_transcript_projections(
    store: EventStore,
    *,
    worker_id: str = "hermes-history-sync",
    max_jobs: int = 10000,
) -> dict[str, int]:
    """Converge pending transcript projections without a resident worker.

    Hermes history synchronization is already a bounded lifecycle hook. Draining
    here makes newly ingested transcript evidence immediately recallable while
    preserving the transactional outbox as the failure/retry boundary.
    """
    if max_jobs <= 0:
        raise ValueError("max_jobs must be positive")
    result = {"transcript_chunks": 0, "transcript_fts": 0}
    chunk_projector = TranscriptChunkProjector(store, worker_id=worker_id)
    processed = 0
    while processed < max_jobs:
        now, lease_until = _timestamp_pair()
        chunk_result = chunk_projector.run_once(now=now, lease_until=lease_until)
        if chunk_result["failed"]:
            raise RuntimeError("transcript chunk projection failed")
        if chunk_result["applied"]:
            result["transcript_chunks"] += chunk_result["applied"]
            processed += chunk_result["applied"]
            continue

        jobs = store.claim_projection_jobs(
            worker_id=worker_id,
            projection_kind="transcript_fts",
            limit=min(100, max_jobs - processed),
            now=now,
            lease_until=lease_until,
        )
        if not jobs:
            break
        for job in jobs:
            _apply_fts_job(store, job, worker_id=worker_id, now=now)
            result["transcript_fts"] += 1
            processed += 1
    if processed >= max_jobs:
        pending = store._conn.execute(
            "SELECT COUNT(*) FROM projection_outbox WHERE status = 'pending'"
        ).fetchone()[0]
        if pending:
            raise RuntimeError(f"transcript projection drain limit reached with {pending} pending jobs")
    return result


def require_transcript_projections_applied(store: EventStore, *, event_id: int) -> None:
    """Fail closed unless every transcript projection for one event is applied."""
    rows = store._conn.execute(
        """
        SELECT projection_kind, status
        FROM projection_outbox
        WHERE event_seq = ?
        """,
        (int(event_id),),
    ).fetchall()
    statuses = {str(row["projection_kind"]): str(row["status"]) for row in rows}
    incomplete = {
        kind: statuses.get(kind, "missing")
        for kind in TRANSCRIPT_PROJECTIONS
        if statuses.get(kind) != "applied"
    }
    if incomplete:
        details = ", ".join(f"{kind}={status}" for kind, status in incomplete.items())
        raise RuntimeError(f"transcript projections are not converged for event {event_id}: {details}")
