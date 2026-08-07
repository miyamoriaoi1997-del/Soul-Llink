from __future__ import annotations

from pathlib import Path

from pcltm.projections.transcript_chunks import TranscriptChunkProjector
from pcltm.store import EventStore


def test_chunk_projector_materializes_idempotently_and_advances_watermark(tmp_path: Path) -> None:
    store = EventStore(tmp_path / "pcltm.db")
    try:
        event_id, _ = store.ingest_external_event(
            external_id="source:1", source_hash="hash", kind="chat_message",
            session_id="s", conversation_id="c", platform="desktop",
            role="user", source="test",
            content="第一段。\n\n第二段需要被精确索引。\n\n第三段。",
            category="raw_conversation", subcategory="user", inject_policy="retrieve_only",
        )
        projector = TranscriptChunkProjector(store, worker_id="worker-1", max_chars=12, overlap_chars=3)
        first = projector.run_once(now="2026-07-17T04:00:00Z", lease_until="2026-07-17T04:01:00Z")
        store._conn.execute(
            "UPDATE projection_outbox SET status='pending', applied_at=NULL WHERE projection_kind='transcript_chunks'"
        )
        store._conn.commit()
        second = projector.run_once(now="2026-07-17T04:02:00Z", lease_until="2026-07-17T04:03:00Z")
        chunks = store._conn.execute(
            "SELECT * FROM event_chunks WHERE event_id=? ORDER BY chunk_ordinal", (event_id,)
        ).fetchall()
        watermark = store._conn.execute(
            "SELECT * FROM runtime_watermarks WHERE projection_kind='transcript_chunks'"
        ).fetchone()
    finally:
        store.close()

    assert first == {"claimed": 1, "applied": 1, "failed": 0}
    assert second == {"claimed": 1, "applied": 1, "failed": 0}
    assert len(chunks) >= 3
    assert [row["chunk_ordinal"] for row in chunks] == list(range(len(chunks)))
    assert watermark["applied_event_seq"] == event_id


def test_chunk_projector_does_not_count_failed_ack_as_applied(tmp_path: Path, monkeypatch) -> None:
    store = EventStore(tmp_path / "pcltm.db")
    try:
        store.ingest_external_event(
            external_id="source:ack-false", source_hash="hash", kind="chat_message",
            session_id="s", conversation_id="c", platform="desktop",
            role="user", source="test", content="chunk payload",
            category="raw_conversation", subcategory="user", inject_policy="retrieve_only",
        )
        monkeypatch.setattr(store, "ack_projection_job", lambda *args, **kwargs: False)
        result = TranscriptChunkProjector(store, worker_id="worker-1").run_once(
            now="2026-07-17T04:00:00Z", lease_until="2026-07-17T04:01:00Z",
        )
    finally:
        store.close()

    assert result == {"claimed": 1, "applied": 0, "failed": 1}


def test_chunk_projector_rejects_noncanonical_event_authority_id(
    tmp_path: Path,
) -> None:
    store = EventStore(tmp_path / "pcltm.db")
    try:
        event_id = store.append_event(
            session_id="s", conversation_id="c", platform="desktop",
            role="user", source="test", content="must not project",
            category="raw_conversation", subcategory="user",
            inject_policy="retrieve_only",
        )
        # Simulate a pre-hardened/corrupt row so the projector's independent
        # fail-closed boundary remains covered even though fresh DB writes are gated.
        store._conn.execute("PRAGMA ignore_check_constraints = ON")
        store._conn.execute(
            """
            UPDATE projection_outbox SET authority_id = ?
            WHERE event_seq = ? AND projection_kind = 'transcript_chunks'
            """,
            (f"+{event_id}", event_id),
        )
        store._conn.commit()
        store._conn.execute("PRAGMA ignore_check_constraints = OFF")
        result = TranscriptChunkProjector(store, worker_id="worker-1").run_once(
            now="2026-07-17T04:00:00Z",
            lease_until="2026-07-17T04:01:00Z",
        )
        chunk_count = int(store._conn.execute(
            "SELECT count(*) FROM event_chunks WHERE event_id = ?", (event_id,),
        ).fetchone()[0])
        job = store._conn.execute(
            """
            SELECT status, last_error FROM projection_outbox
            WHERE event_seq = ? AND projection_kind = 'transcript_chunks'
            """,
            (event_id,),
        ).fetchone()
    finally:
        store.close()

    assert result == {"claimed": 1, "applied": 0, "failed": 1}
    assert chunk_count == 0
    assert tuple(job) == ("pending", "transcript projection authority mismatch")
