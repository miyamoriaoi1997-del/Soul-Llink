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
