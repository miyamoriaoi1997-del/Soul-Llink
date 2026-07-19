from __future__ import annotations

import sqlite3
from pathlib import Path

import pytest

import pcltm.projections.runtime as projection_runtime
from pcltm.projections.transcript_chunks import TranscriptChunkProjector
from pcltm.store import EventStore


def _apply_chunk_projection(store: EventStore) -> None:
    result = TranscriptChunkProjector(store, worker_id="test-chunks").run_once(
        now="2026-07-19T17:00:00Z", lease_until="2026-07-19T17:01:00Z",
    )
    assert result == {"claimed": 1, "applied": 1, "failed": 0}


def test_external_ingest_commits_event_and_projection_jobs_together(tmp_path: Path) -> None:
    db = tmp_path / "pcltm.db"
    store = EventStore(db)
    try:
        event_id, inserted = store.ingest_external_event(
            external_id="source:1",
            source_hash="source-hash",
            kind="chat_message",
            payload_metadata={"created_at": "2026-07-17T03:00:00Z"},
            session_id="session-1",
            conversation_id="conversation-1",
            platform="desktop",
            role="user",
            source="test",
            content="需要索引的原始事件",
            category="raw_conversation",
            subcategory="user",
            inject_policy="retrieve_only",
        )
    finally:
        store.close()

    assert inserted is True
    with sqlite3.connect(db) as conn:
        rows = conn.execute(
            """
            SELECT projection_kind, aggregate_id, aggregate_version, status
            FROM projection_outbox
            WHERE event_seq = ?
            ORDER BY projection_kind
            """,
            (event_id,),
        ).fetchall()
    assert rows == [
        ("transcript_chunks", str(event_id), 1, "pending"),
        ("transcript_fts", str(event_id), 1, "pending"),
    ]


def test_direct_append_commits_projection_jobs_in_same_transaction(tmp_path: Path) -> None:
    db = tmp_path / "pcltm.db"
    store = EventStore(db)
    try:
        event_id = store.append_event(
            session_id="s", conversation_id="c", platform="desktop",
            role="assistant", source="direct", content="direct event",
            category="raw_conversation", subcategory="assistant", inject_policy="retrieve_only",
        )
        jobs = store._conn.execute(
            "SELECT projection_kind FROM projection_outbox WHERE event_seq=? ORDER BY projection_kind",
            (event_id,),
        ).fetchall()
    finally:
        store.close()

    assert [row["projection_kind"] for row in jobs] == ["transcript_chunks", "transcript_fts"]


def test_failed_ingest_rolls_back_event_and_outbox(tmp_path: Path) -> None:
    db = tmp_path / "pcltm.db"
    store = EventStore(db)
    store._conn.execute(
        """
        CREATE TRIGGER fail_ingest_mapping
        BEFORE INSERT ON ingest_events
        BEGIN
            SELECT RAISE(ABORT, 'forced failure');
        END
        """
    )
    store._conn.commit()
    try:
        try:
            store.ingest_external_event(
                external_id="source:1",
                source_hash="source-hash",
                kind="chat_message",
                session_id="session-1",
                conversation_id="conversation-1",
                platform="desktop",
                role="user",
                source="test",
                content="事务必须回滚",
                category="raw_conversation",
                subcategory="user",
                inject_policy="retrieve_only",
            )
        except sqlite3.IntegrityError:
            pass
        else:
            raise AssertionError("forced ingest failure was not raised")
        counts = {
            "events": store._conn.execute("SELECT count(*) FROM events").fetchone()[0],
            "outbox": store._conn.execute("SELECT count(*) FROM projection_outbox").fetchone()[0],
            "fts": store._conn.execute("SELECT count(*) FROM event_fts").fetchone()[0],
        }
    finally:
        store.close()

    assert counts == {"events": 0, "outbox": 0, "fts": 0}


def test_claim_ack_and_expired_lease_recovery_are_idempotent(tmp_path: Path) -> None:
    db = tmp_path / "pcltm.db"
    store = EventStore(db)
    try:
        event_id, _ = store.ingest_external_event(
            external_id="source:1", source_hash="hash", kind="chat_message",
            session_id="s", conversation_id="c", platform="desktop",
            role="user", source="test", content="projection payload",
            category="raw_conversation", subcategory="user", inject_policy="retrieve_only",
        )
        first = store.claim_projection_jobs(
            worker_id="worker-1",
            projection_kind="transcript_chunks",
            limit=1,
            now="2026-07-17T03:00:00Z",
            lease_until="2026-07-17T03:01:00Z",
        )
        blocked = store.claim_projection_jobs(
            worker_id="worker-2",
            projection_kind="transcript_chunks",
            limit=1,
            now="2026-07-17T03:00:30Z",
            lease_until="2026-07-17T03:02:00Z",
        )
        recovered = store.claim_projection_jobs(
            worker_id="worker-2",
            projection_kind="transcript_chunks",
            limit=1,
            now="2026-07-17T03:01:01Z",
            lease_until="2026-07-17T03:02:00Z",
        )
        store.ack_projection_job(recovered[0]["outbox_id"], worker_id="worker-2", now="2026-07-17T03:01:02Z")
        store.ack_projection_job(recovered[0]["outbox_id"], worker_id="worker-2", now="2026-07-17T03:01:03Z")
        row = store._conn.execute(
            "SELECT status, applied_at FROM projection_outbox WHERE outbox_id = ?",
            (recovered[0]["outbox_id"],),
        ).fetchone()
    finally:
        store.close()

    assert len(first) == 1
    assert blocked == []
    assert len(recovered) == 1
    assert recovered[0]["event_seq"] == event_id
    assert row["status"] == "applied"
    assert row["applied_at"] == "2026-07-17T03:01:02Z"


def test_failed_projection_retries_then_moves_to_dead_letter(tmp_path: Path) -> None:
    store = EventStore(tmp_path / "pcltm.db")
    try:
        store.ingest_external_event(
            external_id="source:1", source_hash="hash", kind="chat_message",
            session_id="s", conversation_id="c", platform="desktop",
            role="user", source="test", content="projection payload",
            category="raw_conversation", subcategory="user", inject_policy="retrieve_only",
        )
        job = store.claim_projection_jobs(
            worker_id="worker-1", projection_kind="transcript_chunks", limit=1,
            now="2026-07-17T03:00:00Z", lease_until="2026-07-17T03:01:00Z",
        )[0]
        retrying = store.fail_projection_job(
            job["outbox_id"], worker_id="worker-1", error="temporary",
            now="2026-07-17T03:00:10Z", next_retry_at="2026-07-17T03:05:00Z",
            max_attempts=2,
        )
        reclaimed = store.claim_projection_jobs(
            worker_id="worker-2", projection_kind="transcript_chunks", limit=1,
            now="2026-07-17T03:05:01Z", lease_until="2026-07-17T03:06:00Z",
        )[0]
        dead = store.fail_projection_job(
            reclaimed["outbox_id"], worker_id="worker-2", error="permanent",
            now="2026-07-17T03:05:02Z", next_retry_at="2026-07-17T03:10:00Z",
            max_attempts=2,
        )
    finally:
        store.close()

    assert retrying["status"] == "pending"
    assert retrying["next_retry_at"] == "2026-07-17T03:05:00Z"
    assert dead["status"] == "dead_letter"
    assert dead["last_error"] == "permanent"


@pytest.mark.parametrize("damage", ("missing", "stale"))
def test_fts_worker_rebuilds_missing_or_stale_projection_before_ack(tmp_path: Path, damage: str) -> None:
    store = EventStore(tmp_path / "pcltm.db")
    try:
        event_id = store.append_event(
            session_id="s", conversation_id="c", platform="desktop",
            role="user", source="test", content="authoritative transcript",
            category="raw_conversation", subcategory="user", inject_policy="retrieve_only",
        )
        if damage == "missing":
            store._conn.execute("DELETE FROM event_fts WHERE rowid = ?", (event_id,))
        else:
            store._conn.execute("UPDATE event_fts SET content = ? WHERE rowid = ?", ("stale", event_id))
        store._conn.commit()

        projection_runtime.drain_transcript_projections(store, worker_id="fts-repair")
        indexed = store._conn.execute("SELECT content FROM event_fts WHERE rowid = ?", (event_id,)).fetchone()
        outbox = store._conn.execute(
            "SELECT status FROM projection_outbox WHERE event_seq = ? AND projection_kind = 'transcript_fts'",
            (event_id,),
        ).fetchone()
    finally:
        store.close()

    assert indexed["content"] == "authoritative transcript"
    assert outbox["status"] == "applied"


def test_fts_worker_retries_idempotently_after_materialization_before_ack_failure(tmp_path: Path, monkeypatch) -> None:
    store = EventStore(tmp_path / "pcltm.db")
    try:
        event_id = store.append_event(
            session_id="s", conversation_id="c", platform="desktop",
            role="user", source="test", content="retry-safe transcript",
            category="raw_conversation", subcategory="user", inject_policy="retrieve_only",
        )
        store._conn.execute("DELETE FROM event_fts WHERE rowid = ?", (event_id,))
        store._conn.commit()
        _apply_chunk_projection(store)
        real_ack = store.ack_projection_job
        failed = False

        def fail_once_after_materialization(*args, **kwargs):
            nonlocal failed
            if not failed:
                failed = True
                raise RuntimeError("ack transport failed")
            return real_ack(*args, **kwargs)

        monkeypatch.setattr(store, "ack_projection_job", fail_once_after_materialization)
        with pytest.raises(RuntimeError, match="ack transport failed"):
            projection_runtime.drain_transcript_projections(store, worker_id="fts-retry")
        first = store._conn.execute(
            "SELECT status FROM projection_outbox WHERE event_seq = ? AND projection_kind = 'transcript_fts'",
            (event_id,),
        ).fetchone()
        indexed_after_failure = store._conn.execute(
            "SELECT content FROM event_fts WHERE rowid = ?", (event_id,)
        ).fetchone()
        store._conn.execute(
            "UPDATE projection_outbox SET next_retry_at = NULL WHERE event_seq = ? AND projection_kind = 'transcript_fts'",
            (event_id,),
        )
        store._conn.commit()
        projection_runtime.drain_transcript_projections(store, worker_id="fts-retry")
        second = store._conn.execute(
            "SELECT status FROM projection_outbox WHERE event_seq = ? AND projection_kind = 'transcript_fts'",
            (event_id,),
        ).fetchone()
    finally:
        store.close()

    assert indexed_after_failure["content"] == "retry-safe transcript"
    assert first["status"] == "pending"
    assert second["status"] == "applied"


def test_fts_worker_write_failure_leaves_job_unacknowledged(tmp_path: Path, monkeypatch) -> None:
    store = EventStore(tmp_path / "pcltm.db")
    try:
        event_id = store.append_event(
            session_id="s", conversation_id="c", platform="desktop",
            role="user", source="test", content="write failure transcript",
            category="raw_conversation", subcategory="user", inject_policy="retrieve_only",
        )
        store._conn.execute("DELETE FROM event_fts WHERE rowid = ?", (event_id,))
        store._conn.commit()
        _apply_chunk_projection(store)
        monkeypatch.setattr(projection_runtime, "_materialize_fts", lambda *args, **kwargs: (_ for _ in ()).throw(OSError("disk full")))

        with pytest.raises(RuntimeError, match="transcript FTS projection failed"):
            projection_runtime.drain_transcript_projections(store, worker_id="fts-write-failure")
        outbox = store._conn.execute(
            "SELECT status, last_error FROM projection_outbox WHERE event_seq = ? AND projection_kind = 'transcript_fts'",
            (event_id,),
        ).fetchone()
    finally:
        store.close()

    assert outbox["status"] == "pending"
    assert "disk full" in outbox["last_error"]


def test_fts_worker_ack_false_after_real_lease_handoff_is_not_reported_applied(
    tmp_path: Path, monkeypatch
) -> None:
    db = tmp_path / "pcltm.db"
    store = EventStore(db)
    contender = EventStore(db)
    try:
        event_id = store.append_event(
            session_id="s", conversation_id="c", platform="desktop",
            role="user", source="test", content="lease handoff transcript",
            category="raw_conversation", subcategory="user", inject_policy="retrieve_only",
        )
        _apply_chunk_projection(store)
        real_ack = store.ack_projection_job

        def handoff_before_ack(outbox_id: int, *, worker_id: str, now: str) -> bool:
            claimed = contender.claim_projection_jobs(
                worker_id="fts-contender",
                projection_kind="transcript_fts",
                limit=1,
                now="2099-01-01T00:00:00Z",
                lease_until="2099-01-01T00:01:00Z",
            )
            assert [job["outbox_id"] for job in claimed] == [outbox_id]
            return real_ack(outbox_id, worker_id=worker_id, now=now)

        monkeypatch.setattr(store, "ack_projection_job", handoff_before_ack)
        with pytest.raises(RuntimeError, match="acknowledgement ownership lost"):
            projection_runtime.drain_transcript_projections(store, worker_id="fts-original")

        outbox = contender._conn.execute(
            "SELECT status, lease_owner FROM projection_outbox "
            "WHERE event_seq = ? AND projection_kind = 'transcript_fts'",
            (event_id,),
        ).fetchone()
    finally:
        store.close()
        contender.close()

    assert dict(outbox) == {"status": "processing", "lease_owner": "fts-contender"}
