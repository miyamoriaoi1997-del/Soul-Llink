from __future__ import annotations

import sqlite3
from pathlib import Path

from pcltm.store import EventStore


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
