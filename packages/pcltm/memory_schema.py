"""Additive SQLite schema for governed, immutable memory claims.

The helper only executes DDL on the caller-owned connection.  It deliberately
never commits, so EventStore (or another migration owner) controls the outer
transaction and can roll it back atomically.
"""

from __future__ import annotations

import sqlite3


_MEMORY_SCHEMA_SQL = """
CREATE TABLE IF NOT EXISTS memory_claims (
    claim_id INTEGER PRIMARY KEY AUTOINCREMENT,
    canonical_key TEXT NOT NULL UNIQUE CHECK (length(trim(canonical_key)) > 0),
    target TEXT NOT NULL CHECK (length(trim(target)) > 0),
    memory_type TEXT NOT NULL CHECK (length(trim(memory_type)) > 0),
    created_at TEXT NOT NULL DEFAULT (strftime('%Y-%m-%dT%H:%M:%fZ', 'now'))
);

CREATE TABLE IF NOT EXISTS memory_claim_versions (
    claim_version_id INTEGER PRIMARY KEY AUTOINCREMENT,
    claim_id INTEGER NOT NULL,
    version INTEGER NOT NULL CHECK (version > 0),
    content TEXT NOT NULL CHECK (length(content) > 0),
    content_sha256 TEXT NOT NULL CHECK (length(content_sha256) = 64 AND content_sha256 NOT GLOB '*[^0-9a-f]*'),
    confidence REAL NOT NULL CHECK (confidence >= 0.0 AND confidence <= 1.0),
    sensitivity TEXT NOT NULL CHECK (sensitivity IN ('normal', 'private', 'restricted', 'secret')),
    injection_policy TEXT NOT NULL CHECK (length(trim(injection_policy)) > 0),
    mode_scope TEXT NOT NULL CHECK (length(trim(mode_scope)) > 0),
    ttl_seconds INTEGER CHECK (ttl_seconds IS NULL OR ttl_seconds > 0),
    expires_at TEXT,
    lineage_kind TEXT NOT NULL CHECK (lineage_kind IN (
        'event_derived', 'explicit_user_assertion', 'system_governed_invariant',
        'legacy_governed', 'transient_task_state'
    )),
    schema_version INTEGER NOT NULL CHECK (schema_version > 0),
    created_at TEXT NOT NULL DEFAULT (strftime('%Y-%m-%dT%H:%M:%fZ', 'now')),
    UNIQUE (claim_id, version),
    FOREIGN KEY (claim_id) REFERENCES memory_claims(claim_id)
);

CREATE TABLE IF NOT EXISTS memory_claim_sources (
    claim_source_id INTEGER PRIMARY KEY AUTOINCREMENT,
    claim_version_id INTEGER NOT NULL,
    source_kind TEXT NOT NULL CHECK (source_kind IN ('event', 'legacy_record', 'system')),
    event_id INTEGER,
    event_revision INTEGER CHECK (event_revision IS NULL OR event_revision > 0),
    event_payload_sha256 TEXT CHECK (event_payload_sha256 IS NULL OR (length(event_payload_sha256) = 64 AND event_payload_sha256 NOT GLOB '*[^0-9a-f]*')),
    legacy_record_id INTEGER,
    legacy_content_sha256 TEXT CHECK (legacy_content_sha256 IS NULL OR (length(legacy_content_sha256) = 64 AND legacy_content_sha256 NOT GLOB '*[^0-9a-f]*')),
    created_at TEXT NOT NULL DEFAULT (strftime('%Y-%m-%dT%H:%M:%fZ', 'now')),
    UNIQUE (claim_version_id, source_kind, event_id, event_revision, legacy_record_id),
    CHECK (
        (source_kind = 'event' AND event_id IS NOT NULL AND event_revision IS NOT NULL
            AND event_payload_sha256 IS NOT NULL AND length(event_payload_sha256) = 64
            AND legacy_record_id IS NULL AND legacy_content_sha256 IS NULL)
        OR
        (source_kind = 'legacy_record' AND legacy_record_id IS NOT NULL
            AND legacy_content_sha256 IS NOT NULL AND length(legacy_content_sha256) = 64
            AND event_id IS NULL AND event_revision IS NULL AND event_payload_sha256 IS NULL)
        OR
        (source_kind = 'system' AND event_id IS NULL AND event_revision IS NULL
            AND event_payload_sha256 IS NULL AND legacy_record_id IS NULL
            AND legacy_content_sha256 IS NULL)
    ),
    FOREIGN KEY (claim_version_id) REFERENCES memory_claim_versions(claim_version_id)
);

CREATE TABLE IF NOT EXISTS memory_governance_events (
    memory_governance_id INTEGER PRIMARY KEY AUTOINCREMENT,
    claim_id INTEGER NOT NULL,
    claim_version_id INTEGER NOT NULL,
    action TEXT NOT NULL CHECK (action IN ('submit', 'activate', 'supersede', 'retire', 'expire', 'reject', 'quarantine')),
    previous_state TEXT CHECK (previous_state IS NULL OR previous_state IN (
        'pending_review', 'active', 'superseded', 'retired', 'expired', 'rejected', 'quarantined'
    )),
    new_state TEXT NOT NULL CHECK (new_state IN (
        'pending_review', 'active', 'superseded', 'retired', 'expired', 'rejected', 'quarantined'
    )),
    actor TEXT NOT NULL CHECK (length(trim(actor)) > 0),
    reason_code TEXT NOT NULL CHECK (length(trim(reason_code)) > 0),
    policy_version TEXT NOT NULL CHECK (length(trim(policy_version)) > 0),
    created_at TEXT NOT NULL DEFAULT (strftime('%Y-%m-%dT%H:%M:%fZ', 'now')),
    FOREIGN KEY (claim_id) REFERENCES memory_claims(claim_id),
    FOREIGN KEY (claim_version_id) REFERENCES memory_claim_versions(claim_version_id)
);

CREATE TABLE IF NOT EXISTS memory_transition_receipts (
    transition_receipt_id INTEGER PRIMARY KEY AUTOINCREMENT,
    idempotency_key TEXT NOT NULL UNIQUE CHECK (length(trim(idempotency_key)) > 0),
    request_sha256 TEXT NOT NULL CHECK (length(request_sha256) = 64 AND request_sha256 NOT GLOB '*[^0-9a-f]*'),
    claim_id INTEGER NOT NULL,
    claim_version INTEGER NOT NULL CHECK (claim_version > 0),
    action TEXT NOT NULL CHECK (action IN ('retire', 'expire')),
    memory_governance_id INTEGER NOT NULL,
    created_at TEXT NOT NULL DEFAULT (strftime('%Y-%m-%dT%H:%M:%fZ', 'now')),
    FOREIGN KEY (claim_id) REFERENCES memory_claims(claim_id),
    FOREIGN KEY (memory_governance_id) REFERENCES memory_governance_events(memory_governance_id)
);

CREATE TABLE IF NOT EXISTS memory_projection_guards (
    claim_id INTEGER PRIMARY KEY,
    outbox_id INTEGER NOT NULL,
    attempt_count INTEGER NOT NULL CHECK (attempt_count > 0),
    worker_id TEXT NOT NULL CHECK (length(trim(worker_id)) > 0),
    memfs_root_id TEXT NOT NULL CHECK (length(trim(memfs_root_id)) > 0),
    created_at TEXT NOT NULL DEFAULT (strftime('%Y-%m-%dT%H:%M:%fZ', 'now')),
    FOREIGN KEY (claim_id) REFERENCES memory_claims(claim_id),
    FOREIGN KEY (outbox_id) REFERENCES projection_outbox(outbox_id)
);

CREATE TABLE IF NOT EXISTS memory_current (
    claim_id INTEGER PRIMARY KEY,
    claim_version_id INTEGER NOT NULL,
    memory_governance_id INTEGER NOT NULL,
    lifecycle_state TEXT NOT NULL CHECK (lifecycle_state IN (
        'pending_review', 'active', 'superseded', 'retired', 'expired', 'rejected', 'quarantined'
    )),
    updated_at TEXT NOT NULL DEFAULT (strftime('%Y-%m-%dT%H:%M:%fZ', 'now')),
    FOREIGN KEY (claim_id) REFERENCES memory_claims(claim_id),
    FOREIGN KEY (claim_version_id) REFERENCES memory_claim_versions(claim_version_id),
    FOREIGN KEY (memory_governance_id) REFERENCES memory_governance_events(memory_governance_id)
);

CREATE INDEX IF NOT EXISTS idx_memory_claim_versions_claim
    ON memory_claim_versions(claim_id, version);
CREATE INDEX IF NOT EXISTS idx_memory_claim_sources_version
    ON memory_claim_sources(claim_version_id);
CREATE INDEX IF NOT EXISTS idx_memory_governance_claim
    ON memory_governance_events(claim_id, memory_governance_id);
CREATE UNIQUE INDEX IF NOT EXISTS uq_memory_claim_sources_event
    ON memory_claim_sources(claim_version_id, event_id, event_revision)
    WHERE source_kind = 'event';
CREATE UNIQUE INDEX IF NOT EXISTS uq_memory_claim_sources_legacy
    ON memory_claim_sources(claim_version_id, legacy_record_id)
    WHERE source_kind = 'legacy_record';
CREATE UNIQUE INDEX IF NOT EXISTS uq_memory_claim_sources_legacy_record
    ON memory_claim_sources(legacy_record_id)
    WHERE source_kind = 'legacy_record';
CREATE UNIQUE INDEX IF NOT EXISTS uq_memory_claim_sources_system
    ON memory_claim_sources(claim_version_id)
    WHERE source_kind = 'system';

CREATE TRIGGER IF NOT EXISTS protect_memory_claim_versions_update
BEFORE UPDATE ON memory_claim_versions BEGIN
    SELECT RAISE(ABORT, 'memory claim versions are immutable');
END;
CREATE TRIGGER IF NOT EXISTS protect_memory_claim_versions_delete
BEFORE DELETE ON memory_claim_versions BEGIN
    SELECT RAISE(ABORT, 'memory claim versions are immutable');
END;
CREATE TRIGGER IF NOT EXISTS protect_memory_claim_sources_update
BEFORE UPDATE ON memory_claim_sources BEGIN
    SELECT RAISE(ABORT, 'memory claim sources are immutable');
END;
CREATE TRIGGER IF NOT EXISTS protect_memory_claim_sources_delete
BEFORE DELETE ON memory_claim_sources BEGIN
    SELECT RAISE(ABORT, 'memory claim sources are immutable');
END;
CREATE TRIGGER IF NOT EXISTS protect_memory_governance_update
BEFORE UPDATE ON memory_governance_events BEGIN
    SELECT RAISE(ABORT, 'memory governance events are immutable');
END;
CREATE TRIGGER IF NOT EXISTS protect_memory_governance_delete
BEFORE DELETE ON memory_governance_events BEGIN
    SELECT RAISE(ABORT, 'memory governance events are immutable');
END;
CREATE TRIGGER IF NOT EXISTS memory_governance_transition_valid
BEFORE INSERT ON memory_governance_events
WHEN NOT COALESCE((
    (NEW.action = 'submit' AND NEW.previous_state IS NULL AND NEW.new_state = 'pending_review')
    OR (NEW.action = 'activate' AND NEW.previous_state = 'pending_review' AND NEW.new_state = 'active')
    OR (NEW.action = 'supersede' AND NEW.previous_state = 'active' AND NEW.new_state = 'superseded')
    OR (NEW.action = 'retire' AND NEW.previous_state = 'active' AND NEW.new_state = 'retired')
    OR (NEW.action = 'expire' AND NEW.previous_state = 'active' AND NEW.new_state = 'expired')
    OR (NEW.action = 'reject' AND NEW.previous_state = 'pending_review' AND NEW.new_state = 'rejected')
    OR (NEW.action = 'quarantine' AND NEW.previous_state = 'pending_review' AND NEW.new_state = 'quarantined')
), 0)
BEGIN
    SELECT RAISE(ABORT, 'invalid memory governance transition');
END;
CREATE TRIGGER IF NOT EXISTS memory_activation_requires_source
BEFORE INSERT ON memory_governance_events
WHEN NEW.new_state = 'active'
 AND NOT EXISTS (
     SELECT 1 FROM memory_claim_sources s
     WHERE s.claim_version_id = NEW.claim_version_id
 )
BEGIN
    SELECT RAISE(ABORT, 'active memory claim requires source');
END;

CREATE TRIGGER IF NOT EXISTS memory_governance_claim_version_match
BEFORE INSERT ON memory_governance_events
WHEN NOT EXISTS (
    SELECT 1 FROM memory_claim_versions v
    WHERE v.claim_version_id = NEW.claim_version_id AND v.claim_id = NEW.claim_id
)
BEGIN
    SELECT RAISE(ABORT, 'governance claim/version mismatch');
END;

CREATE TRIGGER IF NOT EXISTS memory_current_projection_guard_insert
BEFORE INSERT ON memory_current
WHEN EXISTS (SELECT 1 FROM memory_projection_guards WHERE claim_id = NEW.claim_id)
BEGIN
    SELECT RAISE(ABORT, 'memory current projection guarded');
END;
CREATE TRIGGER IF NOT EXISTS memory_current_projection_guard_update
BEFORE UPDATE ON memory_current
WHEN EXISTS (
    SELECT 1 FROM memory_projection_guards
    WHERE claim_id IN (OLD.claim_id, NEW.claim_id)
)
BEGIN
    SELECT RAISE(ABORT, 'memory current projection guarded');
END;
CREATE TRIGGER IF NOT EXISTS memory_current_projection_guard_delete
BEFORE DELETE ON memory_current
WHEN EXISTS (SELECT 1 FROM memory_projection_guards WHERE claim_id = OLD.claim_id)
BEGIN
    SELECT RAISE(ABORT, 'memory current projection guarded');
END;

CREATE TRIGGER IF NOT EXISTS memory_current_claim_version_match
BEFORE INSERT ON memory_current
WHEN NOT EXISTS (
    SELECT 1 FROM memory_claim_versions v
    WHERE v.claim_version_id = NEW.claim_version_id AND v.claim_id = NEW.claim_id
)
BEGIN
    SELECT RAISE(ABORT, 'current claim/version mismatch');
END;

CREATE TRIGGER IF NOT EXISTS memory_current_governance_match
BEFORE INSERT ON memory_current
WHEN NOT EXISTS (
    SELECT 1 FROM memory_governance_events g
    WHERE g.memory_governance_id = NEW.memory_governance_id
      AND g.claim_id = NEW.claim_id
      AND g.claim_version_id = NEW.claim_version_id
      AND g.new_state = NEW.lifecycle_state
)
BEGIN
    SELECT RAISE(ABORT, 'current governance mismatch');
END;

CREATE TRIGGER IF NOT EXISTS memory_current_claim_version_match_update
BEFORE UPDATE ON memory_current
WHEN NOT EXISTS (
    SELECT 1 FROM memory_claim_versions v
    WHERE v.claim_version_id = NEW.claim_version_id AND v.claim_id = NEW.claim_id
)
BEGIN
    SELECT RAISE(ABORT, 'current claim/version mismatch');
END;

CREATE TRIGGER IF NOT EXISTS memory_current_governance_match_update
BEFORE UPDATE ON memory_current
WHEN NOT EXISTS (
    SELECT 1 FROM memory_governance_events g
    WHERE g.memory_governance_id = NEW.memory_governance_id
      AND g.claim_id = NEW.claim_id
      AND g.claim_version_id = NEW.claim_version_id
      AND g.new_state = NEW.lifecycle_state
)
BEGIN
    SELECT RAISE(ABORT, 'current governance mismatch');
END;
"""


def _execute_script_without_implicit_commit(conn: sqlite3.Connection, script: str) -> None:
    statement = ""
    for line in script.splitlines(keepends=True):
        statement += line
        if sqlite3.complete_statement(statement):
            sql = statement.strip()
            if sql:
                conn.execute(sql)
            statement = ""
    if statement.strip():
        raise ValueError("incomplete memory schema statement")


def ensure_memory_claim_schema(conn: sqlite3.Connection) -> None:
    """Install the additive memory claim schema without taking transaction ownership."""
    guard_columns = {
        str(row[1])
        for row in conn.execute(
            "PRAGMA table_info(memory_projection_guards)"
        ).fetchall()
    }
    if guard_columns and "memfs_root_id" not in guard_columns:
        # Guards are ephemeral coordination state. Bootstrap owns an exclusive
        # migration transaction, so an old-format guard cannot represent a live
        # projector and must not be guessed into a root identity.
        conn.execute("DROP TABLE memory_projection_guards")
    _execute_script_without_implicit_commit(conn, _MEMORY_SCHEMA_SQL)
