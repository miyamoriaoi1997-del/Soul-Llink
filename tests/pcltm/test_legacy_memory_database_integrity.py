from __future__ import annotations

import sqlite3
from pathlib import Path

import pytest

from pcltm.ledger_schema import ensure_evidence_ledger_schema
from pcltm.legacy_memory_promotion import (
    LegacyMemoryPromotionRequest,
    LegacyMemoryPromotionService,
    LegacyMemoryPromotionSpec,
)
from pcltm.memory_contracts import PersonaMode
from pcltm.store import EventStore


def _checks(store: EventStore) -> tuple[str, list[tuple[object, ...]]]:
    integrity = str(store._conn.execute("PRAGMA integrity_check").fetchone()[0])
    foreign_keys = [tuple(row) for row in store._conn.execute("PRAGMA foreign_key_check").fetchall()]
    return integrity, foreign_keys


def test_database_integrity_after_legacy_promotion(tmp_path: Path) -> None:
    store = EventStore(tmp_path / "authority.db")
    try:
        record_id, _ = store.add_memory_record(
            candidate_id="legacy-integrity-check",
            kind="memory_note",
            target_file="USER.md",
            content="legacy integrity check token",
            confidence=1.0,
            sensitivity="normal",
            status="approved",
        )
        LegacyMemoryPromotionService(store).promote(
            LegacyMemoryPromotionRequest((
                LegacyMemoryPromotionSpec(
                    record_id=record_id,
                    canonical_key="legacy:integrity:check",
                    target="profile",
                    memory_type="preference",
                    mode_scope=(PersonaMode.DAILY,),
                    injection_policy="allow",
                ),
            ))
        )
        integrity, foreign_keys = _checks(store)
    finally:
        store.close()

    assert integrity == "ok"
    assert foreign_keys == []


def test_database_integrity_after_legacy_outbox_schema_upgrade(tmp_path: Path) -> None:
    db = tmp_path / "legacy-outbox.db"
    baseline = EventStore(db)
    try:
        event_id = baseline.append_event(
            session_id="s", conversation_id="c", platform="desktop",
            source="chat", role="user", content="legacy outbox upgrade token",
            category="raw_conversation", subcategory="user",
            inject_policy="retrieve_only",
        )
        baseline._conn.execute("DROP INDEX IF EXISTS idx_projection_outbox_pending")
        baseline._conn.execute("ALTER TABLE projection_outbox RENAME TO projection_outbox_current")
        baseline._conn.execute(
            """
            CREATE TABLE projection_outbox (
                outbox_id INTEGER PRIMARY KEY AUTOINCREMENT,
                event_seq INTEGER NOT NULL,
                projection_kind TEXT NOT NULL,
                aggregate_id TEXT NOT NULL,
                aggregate_version INTEGER NOT NULL,
                payload_sha256 TEXT NOT NULL,
                status TEXT NOT NULL DEFAULT 'pending',
                attempt_count INTEGER NOT NULL DEFAULT 0,
                lease_owner TEXT, lease_until TEXT, next_retry_at TEXT,
                last_error TEXT,
                created_at TEXT NOT NULL DEFAULT '2026-07-29T00:00:00Z',
                applied_at TEXT,
                UNIQUE(projection_kind, aggregate_id, aggregate_version)
            )
            """
        )
        baseline._conn.execute(
            """
            INSERT INTO projection_outbox(
                outbox_id, event_seq, projection_kind, aggregate_id,
                aggregate_version, payload_sha256, status, attempt_count,
                lease_owner, lease_until, next_retry_at, last_error,
                created_at, applied_at
            )
            SELECT outbox_id, event_seq, projection_kind, aggregate_id,
                   aggregate_version, payload_sha256, status, attempt_count,
                   lease_owner, lease_until, next_retry_at, last_error,
                   created_at, applied_at
            FROM projection_outbox_current
            """
        )
        baseline._conn.execute("DROP TABLE projection_outbox_current")
        baseline._conn.commit()
    finally:
        baseline.close()

    upgraded = EventStore(db)
    try:
        rows = upgraded._conn.execute(
            """
            SELECT event_seq, authority_kind, authority_id
            FROM projection_outbox WHERE event_seq = ? ORDER BY projection_kind
            """,
            (event_id,),
        ).fetchall()
        integrity, foreign_keys = _checks(upgraded)
    finally:
        upgraded.close()

    assert [(row["event_seq"], row["authority_kind"], row["authority_id"]) for row in rows] == [
        (event_id, "event", str(event_id)),
        (event_id, "event", str(event_id)),
    ]
    assert integrity == "ok"
    assert foreign_keys == []


@pytest.mark.parametrize("authority_id", ["+1", "01", " 1", "1 ", "１", "١"])
def test_outbox_upgrade_rejects_noncanonical_existing_authority_identity(
    tmp_path: Path,
    authority_id: str,
) -> None:
    db = tmp_path / "noncanonical-outbox.db"
    baseline = EventStore(db)
    try:
        baseline._conn.execute("DROP INDEX IF EXISTS idx_projection_outbox_pending")
        baseline._conn.execute("ALTER TABLE projection_outbox RENAME TO projection_outbox_current")
        baseline._conn.execute(
            """
            CREATE TABLE projection_outbox (
                outbox_id INTEGER PRIMARY KEY AUTOINCREMENT,
                event_seq INTEGER,
                authority_kind TEXT,
                authority_id TEXT,
                projection_kind TEXT NOT NULL,
                aggregate_id TEXT NOT NULL,
                aggregate_version INTEGER NOT NULL,
                payload_sha256 TEXT NOT NULL,
                status TEXT NOT NULL DEFAULT 'pending',
                attempt_count INTEGER NOT NULL DEFAULT 0,
                lease_owner TEXT, lease_until TEXT, next_retry_at TEXT,
                last_error TEXT, created_at TEXT NOT NULL, applied_at TEXT,
                UNIQUE(projection_kind, aggregate_id, aggregate_version)
            )
            """
        )
        baseline._conn.execute(
            """
            INSERT INTO projection_outbox(
                event_seq, authority_kind, authority_id, projection_kind,
                aggregate_id, aggregate_version, payload_sha256, created_at
            ) VALUES (1, 'event', ?, 'transcript_chunks', ?, 1, ?,
                      '2026-07-29T00:00:00Z')
            """,
            (authority_id, f"invalid:{authority_id}", "a" * 64),
        )
        baseline._conn.execute("DROP TABLE projection_outbox_current")
        baseline._conn.commit()
    finally:
        baseline.close()

    conn = sqlite3.connect(db)
    conn.row_factory = sqlite3.Row
    try:
        conn.execute("BEGIN IMMEDIATE")
        with pytest.raises(sqlite3.IntegrityError, match="projection outbox authority invalid"):
            ensure_evidence_ledger_schema(conn)
        conn.rollback()
        preserved = conn.execute(
            "SELECT authority_id FROM projection_outbox"
        ).fetchone()[0]
    finally:
        conn.close()

    assert preserved == authority_id
