from __future__ import annotations

import json
import sqlite3
from pathlib import Path

from pcltm.store import EventStore


def _disable_event_immutability_for_tamper_test(store: EventStore) -> None:
    store._conn.execute("DROP TRIGGER protect_events_update")
    store._conn.execute("DROP TRIGGER protect_events_delete")
    store._conn.commit()


def test_verify_chain_detects_payload_tampering(tmp_path: Path) -> None:
    db = tmp_path / "pcltm.db"
    store = EventStore(db)
    try:
        first = store.append_event(
            session_id="session-1", conversation_id="conversation-1",
            platform="desktop", role="user", source="test", content="第一条",
            category="raw_conversation", subcategory="user", inject_policy="retrieve_only",
        )
        second = store.append_event(
            session_id="session-1", conversation_id="conversation-1",
            platform="desktop", role="assistant", source="test", content="第二条",
            category="raw_conversation", subcategory="assistant", inject_policy="retrieve_only",
        )
        assert store.verify_event_chain() == {
            "ok": True,
            "checked": 2,
            "first_invalid_event_id": None,
            "reason": None,
        }
        _disable_event_immutability_for_tamper_test(store)
        store._conn.execute("UPDATE events SET content = '被篡改' WHERE event_id = ?", (first,))
        store._conn.commit()
        report = store.verify_event_chain()
    finally:
        store.close()

    assert report["ok"] is False
    assert report["first_invalid_event_id"] == first
    assert report["reason"] == "payload_sha256_mismatch"


def test_verify_chain_detects_deleted_tail_and_stripped_chain(tmp_path: Path) -> None:
    db = tmp_path / "pcltm.db"
    store = EventStore(db)
    try:
        first = store.append_event(
            session_id="s", conversation_id="c", platform="desktop",
            role="user", source="test", content="first",
            category="raw_conversation", subcategory="user", inject_policy="retrieve_only",
        )
        second = store.append_event(
            session_id="s", conversation_id="c", platform="desktop",
            role="assistant", source="test", content="second",
            category="raw_conversation", subcategory="assistant", inject_policy="retrieve_only",
        )
        _disable_event_immutability_for_tamper_test(store)
        store._conn.execute("DELETE FROM event_fts WHERE rowid = ?", (second,))
        store._conn.execute("DELETE FROM events WHERE event_id = ?", (second,))
        store._conn.commit()
        deleted = store.verify_event_chain()
        store._conn.execute(
            "UPDATE events SET payload_sha256='', previous_chain_hash=NULL, chain_hash='' WHERE event_id=?",
            (first,),
        )
        store._conn.commit()
        stripped = store.verify_event_chain()
    finally:
        store.close()

    assert deleted["ok"] is False
    assert deleted["reason"] == "chain_anchor_mismatch"
    assert stripped["ok"] is False
    assert stripped["reason"] == "missing_chain_envelope"


def test_verify_chain_detects_identity_and_visibility_tampering(tmp_path: Path) -> None:
    db = tmp_path / "pcltm.db"
    store = EventStore(db)
    try:
        event_id, _ = store.ingest_external_event(
            external_id="source:identity", source_hash="source-hash", kind="chat_message",
            payload_metadata={"created_at": "2026-07-17T00:00:00Z"},
            session_id="s", conversation_id="c", platform="desktop",
            role="user", source="test", content="evidence",
            category="raw_conversation", subcategory="user", inject_policy="retrieve_only",
        )
        assert store.verify_event_chain()["ok"] is True
        _disable_event_immutability_for_tamper_test(store)
        store._conn.execute(
            "UPDATE events SET visibility='always_inject', source_revision=99 WHERE event_id=?",
            (event_id,),
        )
        store._conn.commit()
        report = store.verify_event_chain()
    finally:
        store.close()

    assert report["ok"] is False
    assert report["first_invalid_event_id"] == event_id
    assert report["reason"] == "chain_hash_mismatch"


def test_numeric_source_timestamp_round_trips_through_revision_verification(tmp_path: Path) -> None:
    store = EventStore(tmp_path / "pcltm.db")
    try:
        event_id, _ = store.ingest_external_event(
            external_id="hermes-message:442", source_hash="hash", kind="chat_message",
            payload_metadata={"timestamp": 1782367349.4994028},
            session_id="s", conversation_id="c", platform="desktop",
            role="user", source="hermes", content="historical evidence",
            category="raw_conversation", subcategory="user", inject_policy="retrieve_only",
        )
        event = store.get_event(event_id)
        report = store.verify_event_chain()
    finally:
        store.close()

    assert event["source_created_at"] == "1782367349.4994028"
    assert report["ok"] is True


def test_verify_chain_rejects_revision_content_and_source_time_drift(tmp_path: Path) -> None:
    store = EventStore(tmp_path / "pcltm.db")
    try:
        event_id, _ = store.ingest_external_event(
            external_id="source:revision", source_hash="hash", kind="chat_message",
            payload_metadata={"created_at": "2026-07-17T00:00:00Z"},
            session_id="s", conversation_id="c", platform="desktop",
            role="user", source="test", content="evidence",
            category="raw_conversation", subcategory="user", inject_policy="retrieve_only",
        )
        assert store.verify_event_chain()["ok"] is True
        store._conn.execute("DROP TRIGGER protect_event_revisions_update")
        store._conn.execute(
            "UPDATE event_revisions SET content_sha256='drift' WHERE event_id=?",
            (event_id,),
        )
        store._conn.commit()
        content_drift = store.verify_event_chain()
        store._conn.execute(
            "UPDATE event_revisions SET content_sha256=(SELECT payload_sha256 FROM events WHERE event_id=?), payload_metadata=? WHERE event_id=?",
            (event_id, json.dumps({"created_at": "2099-01-01T00:00:00Z"}), event_id),
        )
        store._conn.commit()
        time_drift = store.verify_event_chain()
    finally:
        store.close()

    assert content_drift["ok"] is False
    assert content_drift["reason"] == "revision_content_hash_mismatch"
    assert time_drift["ok"] is False
    assert time_drift["reason"] == "revision_source_time_mismatch"


def _columns(conn: sqlite3.Connection, table: str) -> set[str]:
    return {str(row[1]) for row in conn.execute(f"PRAGMA table_info({table})")}


def test_fresh_store_has_evidence_ledger_schema(tmp_path: Path) -> None:
    db = tmp_path / "pcltm.db"
    store = EventStore(db)
    try:
        assert store.schema_version() >= 9
    finally:
        store.close()

    with sqlite3.connect(db) as conn:
        event_columns = _columns(conn, "events")
        tables = {
            str(row[0])
            for row in conn.execute(
                "SELECT name FROM sqlite_master WHERE type = 'table'"
            )
        }

        assert {
            "external_event_id",
            "turn_id",
            "parent_event_id",
            "source_created_at",
            "recorded_at",
            "payload_sha256",
            "previous_chain_hash",
            "chain_hash",
            "source_revision",
            "evidence_state",
            "redaction_policy",
            "visibility",
            "schema_version",
        } <= event_columns
        assert {
            "event_revisions",
            "event_governance",
            "event_chunks",
            "projection_outbox",
            "projection_generations",
            "runtime_watermarks",
        } <= tables


def test_appended_event_receives_verifiable_ledger_envelope(tmp_path: Path) -> None:
    db = tmp_path / "pcltm.db"
    store = EventStore(db)
    try:
        event_id = store.append_event(
            session_id="session-1",
            conversation_id="conversation-1",
            platform="desktop",
            role="user",
            source="test",
            content="永久原文证据",
            category="raw_conversation",
            subcategory="user",
            inject_policy="retrieve_only",
        )
        row = store.get_event(event_id)
    finally:
        store.close()

    assert row["recorded_at"]
    assert row["payload_sha256"]
    assert row["chain_hash"]
    assert row["source_revision"] == 1
    assert row["evidence_state"] == "active"
    assert row["visibility"] == "retrieve_only"
    assert row["schema_version"] >= 9


def _create_real_v8_database(path: Path) -> None:
    with sqlite3.connect(path) as conn:
        conn.executescript(
            """
            CREATE TABLE schema_migrations (version INTEGER PRIMARY KEY, applied_at TEXT);
            CREATE TABLE events (
                event_id INTEGER PRIMARY KEY AUTOINCREMENT,
                session_id TEXT NOT NULL, conversation_id TEXT NOT NULL,
                platform TEXT NOT NULL, role TEXT NOT NULL, source TEXT NOT NULL,
                content TEXT NOT NULL, persona_mode TEXT, route_bucket TEXT,
                model_hint TEXT, sensitivity TEXT NOT NULL DEFAULT 'normal',
                category TEXT NOT NULL DEFAULT 'unknown',
                subcategory TEXT NOT NULL DEFAULT 'unknown',
                inject_policy TEXT NOT NULL DEFAULT 'retrieve_only',
                classification_confidence REAL NOT NULL DEFAULT 0.0,
                classifier_version TEXT NOT NULL DEFAULT 'unknown',
                created_at TEXT NOT NULL
            );
            CREATE TABLE ingest_events (
                ingest_id INTEGER PRIMARY KEY AUTOINCREMENT,
                external_id TEXT NOT NULL UNIQUE, source_hash TEXT NOT NULL,
                kind TEXT NOT NULL, event_id INTEGER NOT NULL,
                attachments TEXT NOT NULL DEFAULT '[]',
                payload_metadata TEXT NOT NULL DEFAULT '{}', created_at TEXT NOT NULL
            );
            CREATE VIRTUAL TABLE event_fts USING fts5(content);
            """
        )
        conn.executemany(
            "INSERT INTO schema_migrations(version, applied_at) VALUES (?, '2026-01-01T00:00:00Z')",
            [(version,) for version in range(1, 9)],
        )
        conn.execute(
            """
            INSERT INTO events (
                event_id, session_id, conversation_id, platform, role, source,
                content, category, subcategory, inject_policy,
                classification_confidence, classifier_version, created_at
            ) VALUES (7, 's', 'c', 'desktop', 'user', 'hermes_state_db',
                      '旧版原文', 'raw_conversation', 'user', 'retrieve_only',
                      1.0, 'v8', '2026-01-02T00:00:00Z')
            """
        )
        conn.execute("INSERT INTO event_fts(rowid, content) VALUES (7, '旧版原文')")
        conn.execute(
            """
            INSERT INTO ingest_events (
                external_id, source_hash, kind, event_id, attachments,
                payload_metadata, created_at
            ) VALUES ('hermes-message:7', 'old-source-hash', 'hermes_message', 7,
                      '[]', '{"timestamp":"2026-01-02T00:00:00Z"}',
                      '2026-01-02T00:00:01Z')
            """
        )


def test_failed_v8_migration_rolls_back_schema_and_version(tmp_path: Path, monkeypatch) -> None:
    import pcltm.store as store_module

    db = tmp_path / "pcltm-v8.db"
    _create_real_v8_database(db)
    real_migration = store_module.ensure_evidence_ledger_schema

    def fail_after_migration(conn: sqlite3.Connection) -> None:
        real_migration(conn)
        raise RuntimeError("forced migration failure")

    monkeypatch.setattr(store_module, "ensure_evidence_ledger_schema", fail_after_migration)
    try:
        EventStore(db)
    except RuntimeError as exc:
        assert str(exc) == "forced migration failure"
    else:
        raise AssertionError("migration failure was not raised")

    with sqlite3.connect(db) as conn:
        versions = [row[0] for row in conn.execute("SELECT version FROM schema_migrations ORDER BY version")]
        columns = _columns(conn, "events")
        content = conn.execute("SELECT content FROM events WHERE event_id=7").fetchone()[0]

    assert versions == list(range(1, 9))
    assert "chain_hash" not in columns
    assert content == "旧版原文"


def test_real_v8_database_migrates_without_losing_evidence(tmp_path: Path) -> None:
    db = tmp_path / "pcltm-v8.db"
    _create_real_v8_database(db)

    store = EventStore(db)
    try:
        event = store.get_event(7)
        ingest = store.find_ingest_event("hermes-message:7")
        verification = store.verify_event_chain()
        fts = store.search_events("旧版原文", include_sensitive=True)
        revision = store._conn.execute(
            "SELECT * FROM event_revisions WHERE event_id=7 AND source_revision=1"
        ).fetchone()
        projection_jobs = store._conn.execute(
            """SELECT projection_kind, status FROM projection_outbox
               WHERE event_seq=7 ORDER BY projection_kind"""
        ).fetchall()
        chunks = store._conn.execute(
            "SELECT chunk_text, start_char, end_char FROM event_chunks WHERE event_id=7 ORDER BY chunk_ordinal"
        ).fetchall()
    finally:
        store.close()

    assert event["content"] == "旧版原文"
    assert event["external_event_id"] == "hermes-message:7"
    assert event["payload_sha256"]
    assert event["chain_hash"]
    assert ingest is not None and ingest["event_id"] == 7
    assert revision is not None and revision["source_hash"] == "old-source-hash"
    assert [(row["projection_kind"], row["status"]) for row in projection_jobs] == [
        ("transcript_chunks", "applied"),
        ("transcript_fts", "applied"),
    ]
    assert [(row["chunk_text"], row["start_char"], row["end_char"]) for row in chunks] == [
        ("旧版原文", 0, len("旧版原文")),
    ]
    assert [row["event_id"] for row in fts] == [7]
    assert verification["ok"] is True
    assert verification["checked"] == 1


def test_bootstrap_repairs_same_count_fts_content_drift(tmp_path: Path) -> None:
    db = tmp_path / "pcltm.db"
    store = EventStore(db)
    try:
        event_id = store.append_event(
            session_id="s", conversation_id="c", platform="desktop",
            role="user", source="test", content="权威全文内容",
            category="raw_conversation", subcategory="user", inject_policy="retrieve_only",
        )
        store._conn.execute("UPDATE event_fts SET content='漂移索引内容' WHERE rowid=?", (event_id,))
        store._conn.commit()
    finally:
        store.close()

    reopened = EventStore(db)
    try:
        authoritative = reopened.search_events("权威全文内容", include_sensitive=True)
        drift = reopened.search_events("漂移索引内容", include_sensitive=True)
    finally:
        reopened.close()

    assert [row["event_id"] for row in authoritative] == [event_id]
    assert drift == []


def test_schema_migration_is_idempotent_for_v9_database(tmp_path: Path) -> None:
    db = tmp_path / "pcltm.db"
    first = EventStore(db)
    try:
        event_id = first.append_event(
            session_id="session-1",
            conversation_id="conversation-1",
            platform="desktop",
            role="assistant",
            source="test",
            content="保留既有记录",
            category="raw_conversation",
            subcategory="assistant",
            inject_policy="retrieve_only",
        )
    finally:
        first.close()

    second = EventStore(db)
    second.close()
    third = EventStore(db)
    try:
        row = third.get_event(event_id)
        assert row["content"] == "保留既有记录"
        assert third.schema_version() >= 9
    finally:
        third.close()
