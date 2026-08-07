"""Idempotent transcript-chunk materialization."""

from __future__ import annotations

from datetime import UTC, datetime, timedelta
from typing import Any

from ..store import EventStore
from ..projection_outbox import require_event_projection_authority
from ..transcript_chunker import CHUNKER_VERSION, chunk_transcript


class TranscriptChunkProjector:
    def __init__(
        self,
        store: EventStore,
        *,
        worker_id: str,
        max_chars: int = 1200,
        overlap_chars: int = 120,
    ) -> None:
        self.store = store
        self.worker_id = worker_id
        self.max_chars = max_chars
        self.overlap_chars = overlap_chars

    def run_once(self, *, now: str, lease_until: str) -> dict[str, int]:
        jobs = self.store.claim_projection_jobs(
            worker_id=self.worker_id,
            projection_kind="transcript_chunks",
            limit=1,
            now=now,
            lease_until=lease_until,
        )
        result = {"claimed": len(jobs), "applied": 0, "failed": 0}
        for job in jobs:
            try:
                self._apply(job, now=now)
                acked = self.store.ack_projection_job(
                    job["outbox_id"], worker_id=self.worker_id,
                    expected_attempt_count=int(job["attempt_count"]), now=now,
                )
                if not acked:
                    raise RuntimeError("projection lease ownership lost")
                result["applied"] += 1
            except Exception as exc:
                if str(exc) == "projection lease ownership lost":
                    result["failed"] += 1
                    continue
                retry_at = (datetime.fromisoformat(now.replace("Z", "+00:00")) + timedelta(minutes=1)).astimezone(UTC).isoformat().replace("+00:00", "Z")
                self.store.fail_projection_job(
                    job["outbox_id"], worker_id=self.worker_id, error=str(exc),
                    expected_attempt_count=int(job["attempt_count"]),
                    now=now, next_retry_at=retry_at,
                )
                result["failed"] += 1
        return result

    def _apply(self, job: dict[str, Any], *, now: str) -> None:
        event_id = require_event_projection_authority(job)
        event = self.store.get_event(event_id)
        if event["payload_sha256"] != job["payload_sha256"]:
            raise ValueError("projection payload hash is stale")
        chunks = chunk_transcript(
            str(event["content"]),
            max_chars=self.max_chars,
            overlap_chars=self.overlap_chars,
        )
        conn = self.store._conn
        try:
            conn.execute("BEGIN IMMEDIATE")
            conn.execute(
                "DELETE FROM event_chunks WHERE event_id = ? AND chunker_version = ?",
                (event_id, CHUNKER_VERSION),
            )
            conn.executemany(
                """
                INSERT INTO event_chunks (
                    event_id, chunk_ordinal, start_char, end_char, token_count,
                    chunk_text, chunk_sha256, chunker_version
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                """,
                [
                    (
                        event_id, chunk.ordinal, chunk.start_char, chunk.end_char,
                        len(chunk.text), chunk.text, chunk.sha256, chunk.chunker_version,
                    )
                    for chunk in chunks
                ],
            )
            conn.execute(
                """
                INSERT INTO runtime_watermarks (
                    projection_kind, applied_event_seq, schema_version,
                    producer_version, updated_at
                ) VALUES ('transcript_chunks', ?, 9, ?, ?)
                ON CONFLICT(projection_kind) DO UPDATE SET
                    applied_event_seq = MAX(runtime_watermarks.applied_event_seq, excluded.applied_event_seq),
                    schema_version = excluded.schema_version,
                    producer_version = excluded.producer_version,
                    updated_at = excluded.updated_at
                """,
                (event_id, CHUNKER_VERSION, now),
            )
            conn.commit()
        except BaseException:
            conn.rollback()
            raise
