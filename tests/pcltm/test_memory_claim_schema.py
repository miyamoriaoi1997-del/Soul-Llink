from __future__ import annotations

import sqlite3
from pathlib import Path

import pytest

from pcltm.memory_schema import ensure_memory_claim_schema
from pcltm.store import EventStore


def _claim(conn: sqlite3.Connection, key: str = "pref:timezone") -> int:
    return int(
        conn.execute(
            "INSERT INTO memory_claims(canonical_key, target, memory_type) VALUES (?, ?, ?)",
            (key, "profile", "preference"),
        ).lastrowid
    )


def _version(conn: sqlite3.Connection, claim_id: int, version: int = 1) -> int:
    return int(
        conn.execute(
            """
            INSERT INTO memory_claim_versions(
                claim_id, version, content, content_sha256, confidence,
                sensitivity, injection_policy, mode_scope, lineage_kind, schema_version
            ) VALUES (?, ?, 'UTC+8', ?, 0.9, 'normal', 'allow', 'agent-wide',
                      'explicit_user_assertion', 1)
            """,
            (claim_id, version, "a" * 64),
        ).lastrowid
    )


def test_memory_claim_schema_creates_immutable_version_and_source_rows(tmp_path: Path) -> None:
    db = sqlite3.connect(tmp_path / "claims.db")
    try:
        ensure_memory_claim_schema(db)
        claim_id = _claim(db)
        version_id = _version(db, claim_id)
        source_id = int(
            db.execute(
                """
                INSERT INTO memory_claim_sources(
                    claim_version_id, source_kind, event_id, event_revision,
                    event_payload_sha256
                ) VALUES (?, 'event', 7, 1, ?)
                """,
                (version_id, "b" * 64),
            ).lastrowid
        )
        db.commit()
        assert db.execute("SELECT count(*) FROM memory_claim_versions").fetchone()[0] == 1
        assert db.execute("SELECT count(*) FROM memory_claim_sources").fetchone()[0] == 1
        assert source_id > 0
        with pytest.raises(sqlite3.IntegrityError, match="immutable"):
            db.execute("UPDATE memory_claim_versions SET content='changed' WHERE claim_version_id=?", (version_id,))
        db.rollback()
        with pytest.raises(sqlite3.IntegrityError, match="immutable"):
            db.execute("DELETE FROM memory_claim_sources WHERE claim_source_id=?", (source_id,))
        db.rollback()
    finally:
        db.close()


def test_memory_claim_schema_rejects_duplicate_versions_and_sources(tmp_path: Path) -> None:
    db = sqlite3.connect(tmp_path / "claims.db")
    try:
        ensure_memory_claim_schema(db)
        claim_id = _claim(db)
        version_id = _version(db, claim_id)
        with pytest.raises(sqlite3.IntegrityError):
            _version(db, claim_id)
        db.rollback()
        db.execute(
            "INSERT INTO memory_claim_sources(claim_version_id, source_kind, event_id, event_revision, event_payload_sha256) VALUES (?, 'event', 7, 1, ?)",
            (version_id, "b" * 64),
        )
        with pytest.raises(sqlite3.IntegrityError):
            db.execute(
                "INSERT INTO memory_claim_sources(claim_version_id, source_kind, event_id, event_revision, event_payload_sha256) VALUES (?, 'event', 7, 1, ?)",
                (version_id, "b" * 64),
            )
    finally:
        db.close()


def test_memory_claim_schema_requires_source_before_activation(tmp_path: Path) -> None:
    db = sqlite3.connect(tmp_path / "claims.db")
    try:
        ensure_memory_claim_schema(db)
        claim_id = _claim(db)
        version_id = _version(db, claim_id)
        with pytest.raises(sqlite3.IntegrityError, match="source"):
            db.execute(
                """
                INSERT INTO memory_governance_events(
                    claim_id, claim_version_id, action, previous_state, new_state,
                    actor, reason_code, policy_version
                ) VALUES (?, ?, 'activate', 'pending_review', 'active', 'test', 'write_allowed', 'memory-policy-v1')
                """,
                (claim_id, version_id),
            )
    finally:
        db.close()


def test_memory_claim_schema_rejects_invalid_governance_transition_triple(
    tmp_path: Path,
) -> None:
    db = sqlite3.connect(tmp_path / "invalid-transition.db")
    try:
        ensure_memory_claim_schema(db)
        claim_id = _claim(db)
        version_id = _version(db, claim_id)
        db.execute(
            "INSERT INTO memory_claim_sources(claim_version_id, source_kind, event_id, event_revision, event_payload_sha256) VALUES (?, 'event', 7, 1, ?)",
            (version_id, "b" * 64),
        )
        with pytest.raises(
            sqlite3.IntegrityError,
            match="invalid memory governance transition",
        ):
            db.execute(
                """
                INSERT INTO memory_governance_events(
                    claim_id, claim_version_id, action, previous_state, new_state,
                    actor, reason_code, policy_version
                ) VALUES (?, ?, 'retire', 'pending_review', 'active', 'test', 'x', 'v1')
                """,
                (claim_id, version_id),
            )
        with pytest.raises(
            sqlite3.IntegrityError,
            match="invalid memory governance transition",
        ):
            db.execute(
                """
                INSERT INTO memory_governance_events(
                    claim_id, claim_version_id, action, previous_state, new_state,
                    actor, reason_code, policy_version
                ) VALUES (?, ?, 'activate', NULL, 'active', 'test', 'x', 'v1')
                """,
                (claim_id, version_id),
            )
    finally:
        db.close()


def test_event_store_bootstrap_installs_claim_schema_atomically(tmp_path: Path) -> None:
    db = tmp_path / "pcltm.db"
    store = EventStore(db)
    store.close()
    with sqlite3.connect(db) as conn:
        names = {row[0] for row in conn.execute("SELECT name FROM sqlite_master WHERE type='table' AND name LIKE 'memory_%'")}
    assert {"memory_claims", "memory_claim_versions", "memory_claim_sources", "memory_governance_events", "memory_current"}.issubset(names)


def test_memory_claim_schema_rebuilds_old_projection_guard_format(
    tmp_path: Path,
) -> None:
    db = sqlite3.connect(tmp_path / "old-guard.db")
    try:
        db.execute(
            """
            CREATE TABLE memory_projection_guards(
                claim_id INTEGER PRIMARY KEY,
                outbox_id INTEGER NOT NULL,
                attempt_count INTEGER NOT NULL,
                worker_id TEXT NOT NULL
            )
            """
        )
        db.execute(
            "INSERT INTO memory_projection_guards VALUES (1, 2, 1, 'old-worker')"
        )
        ensure_memory_claim_schema(db)
        columns = {
            str(row[1])
            for row in db.execute(
                "PRAGMA table_info(memory_projection_guards)"
            ).fetchall()
        }
        count = int(db.execute(
            "SELECT count(*) FROM memory_projection_guards"
        ).fetchone()[0])
    finally:
        db.close()

    assert "memfs_root_id" in columns
    assert count == 0


def test_event_store_bootstrap_fault_rolls_back_new_schema_and_preserves_prior_rows(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    db = tmp_path / "prior.db"
    with sqlite3.connect(db) as prior:
        prior.execute("CREATE TABLE memory_records(record_id INTEGER PRIMARY KEY, content TEXT NOT NULL)")
        prior.execute("INSERT INTO memory_records(content) VALUES ('legacy')")
        prior.commit()

    import pcltm.ledger_schema as ledger_schema

    def fail_after_partial_schema(conn: sqlite3.Connection) -> None:
        conn.execute("CREATE TABLE memory_claims(claim_id INTEGER PRIMARY KEY)")
        raise RuntimeError("forced bootstrap fault")

    monkeypatch.setattr(ledger_schema, "ensure_memory_claim_schema", fail_after_partial_schema)
    with pytest.raises(RuntimeError, match="forced bootstrap fault"):
        EventStore(db)

    with sqlite3.connect(db) as check:
        assert check.execute("SELECT content FROM memory_records").fetchall() == [("legacy",)]
        assert check.execute("SELECT name FROM sqlite_master WHERE type='table' AND name='memory_claims'").fetchone() is None
        assert check.execute("SELECT name FROM sqlite_master WHERE type='table' AND name='events'").fetchone() is None


def test_prior_schema_fault_rolls_back_additive_schema(tmp_path: Path) -> None:
    db = tmp_path / "prior.db"
    conn = sqlite3.connect(db)
    try:
        conn.execute("CREATE TABLE memory_records(record_id INTEGER PRIMARY KEY, content TEXT NOT NULL)")
        conn.commit()
        conn.execute("BEGIN")
        ensure_memory_claim_schema(conn)
        conn.execute("CREATE TRIGGER fail_claim_migration BEFORE INSERT ON memory_claims BEGIN SELECT RAISE(ABORT, 'forced claim migration failure'); END")
        with pytest.raises(sqlite3.IntegrityError, match="forced claim migration failure"):
            conn.execute("INSERT INTO memory_claims(canonical_key, target, memory_type) VALUES ('x', 't', 'k')")
        conn.rollback()
    finally:
        conn.close()
    with sqlite3.connect(db) as check:
        assert check.execute("SELECT count(*) FROM memory_records").fetchone()[0] == 0
        assert check.execute("SELECT name FROM sqlite_master WHERE type='table' AND name='memory_claims'").fetchone() is None


def test_memory_claim_schema_rejects_non_hex_hashes(tmp_path: Path) -> None:
    db = sqlite3.connect(tmp_path / "hashes.db")
    try:
        ensure_memory_claim_schema(db)
        claim_id = _claim(db)
        for bad in ("g" * 64, " " * 64, "A" * 64):
            with pytest.raises(sqlite3.IntegrityError):
                db.execute(
                    """INSERT INTO memory_claim_versions(
                    claim_id, version, content, content_sha256, confidence,
                    sensitivity, injection_policy, mode_scope, lineage_kind, schema_version
                    ) VALUES (?, ?, 'x', ?, 0.9, 'normal', 'allow', 'agent-wide',
                              'explicit_user_assertion', 1)""",
                    (claim_id, 10 + len(bad), bad),
                )
            db.rollback()
        version_id = _version(db, claim_id)
        for column, value in (("event_payload_sha256", "g" * 64), ("legacy_content_sha256", " " * 64)):
            if column == "event_payload_sha256":
                sql = "INSERT INTO memory_claim_sources(claim_version_id, source_kind, event_id, event_revision, event_payload_sha256) VALUES (?, 'event', 7, 1, ?)"
            else:
                sql = "INSERT INTO memory_claim_sources(claim_version_id, source_kind, legacy_record_id, legacy_content_sha256) VALUES (?, 'legacy_record', 7, ?)"
            with pytest.raises(sqlite3.IntegrityError):
                db.execute(sql, (version_id, value))
            db.rollback()
    finally:
        db.close()


def test_memory_claim_schema_rejects_cross_claim_governance_and_current_rows(tmp_path: Path) -> None:
    db = sqlite3.connect(tmp_path / "cross-claim.db")
    try:
        ensure_memory_claim_schema(db)
        c1, c2 = _claim(db, "pref:timezone-1"), _claim(db, "pref:timezone-2")
        v1 = _version(db, c1, 1)
        v2 = _version(db, c2, 1)
        db.execute(
            "INSERT INTO memory_claim_sources(claim_version_id, source_kind, event_id, event_revision, event_payload_sha256) VALUES (?, 'event', 7, 1, ?)",
            (v2, "b" * 64),
        )
        db.execute(
            "INSERT INTO memory_claim_sources(claim_version_id, source_kind, event_id, event_revision, event_payload_sha256) VALUES (?, 'event', 7, 1, ?)",
            (v1, "c" * 64),
        )
        db.commit()
        with pytest.raises(sqlite3.IntegrityError):
            db.execute(
                "INSERT INTO memory_governance_events(claim_id, claim_version_id, action, previous_state, new_state, actor, reason_code, policy_version) VALUES (?, ?, 'activate', 'pending_review', 'active', 'test', 'x', 'v1')",
                (c1, v2),
            )
        db.rollback()
        db.execute(
            "INSERT INTO memory_governance_events(claim_id, claim_version_id, action, previous_state, new_state, actor, reason_code, policy_version) VALUES (?, ?, 'activate', 'pending_review', 'active', 'test', 'x', 'v1')",
            (c1, v1),
        )
        g = int(db.execute("SELECT last_insert_rowid()").fetchone()[0])
        with pytest.raises(sqlite3.IntegrityError):
            db.execute("INSERT INTO memory_current(claim_id, claim_version_id, memory_governance_id, lifecycle_state) VALUES (?, ?, ?, 'active')", (c1, v2, g))
        db.rollback()
        with pytest.raises(sqlite3.IntegrityError):
            db.execute("INSERT INTO memory_current(claim_id, claim_version_id, memory_governance_id, lifecycle_state) VALUES (?, ?, ?, 'retired')", (c1, v1, g))
    finally:
        db.close()
