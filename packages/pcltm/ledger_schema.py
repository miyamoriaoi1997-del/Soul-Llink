"""Schema migration helpers for the PCLTM evidence-first ledger."""

from __future__ import annotations

import sqlite3

from .evidence_chain import chain_hash, normalize_source_created_at, sha256_text
from .memory_schema import ensure_memory_claim_schema
from .transcript_chunker import chunk_transcript


EVENT_LEDGER_COLUMNS: dict[str, str] = {
    "external_event_id": "TEXT",
    "turn_id": "TEXT",
    "parent_event_id": "INTEGER",
    "source_created_at": "TEXT",
    "recorded_at": "TEXT NOT NULL DEFAULT ''",
    "payload_sha256": "TEXT NOT NULL DEFAULT ''",
    "previous_chain_hash": "TEXT",
    "chain_hash": "TEXT NOT NULL DEFAULT ''",
    "source_revision": "INTEGER NOT NULL DEFAULT 1",
    "evidence_state": "TEXT NOT NULL DEFAULT 'active'",
    "redaction_policy": "TEXT NOT NULL DEFAULT 'none'",
    "visibility": "TEXT NOT NULL DEFAULT 'retrieve_only'",
    "schema_version": "INTEGER NOT NULL DEFAULT 9",
}


def _ensure_columns(
    conn: sqlite3.Connection,
    table: str,
    columns: dict[str, str],
) -> None:
    existing = {str(row[1]) for row in conn.execute(f"PRAGMA table_info({table})")}
    for name, definition in columns.items():
        if name not in existing:
            conn.execute(f"ALTER TABLE {table} ADD COLUMN {name} {definition}")


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
        raise ValueError("incomplete schema statement")


def ensure_evidence_ledger_schema(conn: sqlite3.Connection) -> None:
    """Create the rebuildable evidence-ledger coordination schema.

    This migration is deliberately additive. Existing event content remains
    untouched; a later governed backfill populates hashes and chain values.
    """

    conn.create_function(
        "pcltm_sha256", 1,
        lambda value: sha256_text(str(value)),
        deterministic=True,
    )
    _ensure_columns(conn, "events", EVENT_LEDGER_COLUMNS)
    _execute_script_without_implicit_commit(
        conn,
        """
        CREATE TABLE IF NOT EXISTS event_revisions (
            revision_id INTEGER PRIMARY KEY AUTOINCREMENT,
            event_id INTEGER NOT NULL,
            source_revision INTEGER NOT NULL,
            source_hash TEXT NOT NULL,
            content_sha256 TEXT NOT NULL,
            payload_metadata TEXT NOT NULL DEFAULT '{}',
            created_at TEXT NOT NULL DEFAULT (strftime('%Y-%m-%dT%H:%M:%fZ', 'now')),
            UNIQUE(event_id, source_revision),
            FOREIGN KEY (event_id) REFERENCES events(event_id)
        );

        CREATE TABLE IF NOT EXISTS event_governance (
            governance_id INTEGER PRIMARY KEY AUTOINCREMENT,
            event_id INTEGER NOT NULL,
            action TEXT NOT NULL,
            previous_state TEXT,
            new_state TEXT NOT NULL,
            actor TEXT NOT NULL,
            reason TEXT NOT NULL,
            created_at TEXT NOT NULL DEFAULT (strftime('%Y-%m-%dT%H:%M:%fZ', 'now')),
            FOREIGN KEY (event_id) REFERENCES events(event_id)
        );

        CREATE TABLE IF NOT EXISTS event_chunks (
            chunk_id INTEGER PRIMARY KEY AUTOINCREMENT,
            event_id INTEGER NOT NULL,
            chunk_ordinal INTEGER NOT NULL,
            start_char INTEGER NOT NULL,
            end_char INTEGER NOT NULL,
            token_count INTEGER NOT NULL DEFAULT 0,
            chunk_text TEXT NOT NULL,
            chunk_sha256 TEXT NOT NULL,
            chunker_version TEXT NOT NULL,
            created_at TEXT NOT NULL DEFAULT (strftime('%Y-%m-%dT%H:%M:%fZ', 'now')),
            UNIQUE(event_id, chunk_ordinal, chunker_version),
            FOREIGN KEY (event_id) REFERENCES events(event_id)
        );

        CREATE TABLE IF NOT EXISTS projection_outbox (
            outbox_id INTEGER PRIMARY KEY AUTOINCREMENT,
            event_seq INTEGER,
            authority_kind TEXT NOT NULL
                CHECK (authority_kind IN ('event', 'legacy_record')),
            authority_id TEXT NOT NULL CHECK (
                length(authority_id) > 0
                AND authority_id NOT GLOB '*[^0-9]*'
                AND substr(authority_id, 1, 1) BETWEEN '1' AND '9'
            ),
            projection_kind TEXT NOT NULL,
            aggregate_id TEXT NOT NULL,
            aggregate_version INTEGER NOT NULL,
            payload_sha256 TEXT NOT NULL,
            status TEXT NOT NULL DEFAULT 'pending',
            attempt_count INTEGER NOT NULL DEFAULT 0,
            lease_owner TEXT,
            lease_until TEXT,
            next_retry_at TEXT,
            last_error TEXT,
            created_at TEXT NOT NULL DEFAULT (strftime('%Y-%m-%dT%H:%M:%fZ', 'now')),
            applied_at TEXT,
            UNIQUE(projection_kind, aggregate_id, aggregate_version)
        );

        CREATE TABLE IF NOT EXISTS projection_generations (
            generation_id INTEGER PRIMARY KEY AUTOINCREMENT,
            projection_kind TEXT NOT NULL,
            from_event_seq INTEGER NOT NULL,
            to_event_seq INTEGER NOT NULL,
            manifest_sha256 TEXT NOT NULL,
            record_count INTEGER NOT NULL,
            status TEXT NOT NULL DEFAULT 'building',
            created_at TEXT NOT NULL DEFAULT (strftime('%Y-%m-%dT%H:%M:%fZ', 'now')),
            verified_at TEXT,
            activated_at TEXT
        );

        CREATE TABLE IF NOT EXISTS runtime_watermarks (
            projection_kind TEXT PRIMARY KEY,
            applied_event_seq INTEGER NOT NULL DEFAULT 0,
            schema_version INTEGER NOT NULL,
            producer_version TEXT NOT NULL,
            updated_at TEXT NOT NULL DEFAULT (strftime('%Y-%m-%dT%H:%M:%fZ', 'now'))
        );

        CREATE TABLE IF NOT EXISTS event_chain_state (
            chain_name TEXT PRIMARY KEY,
            first_event_id INTEGER,
            last_event_id INTEGER,
            event_count INTEGER NOT NULL DEFAULT 0,
            tip_hash TEXT NOT NULL DEFAULT '',
            schema_version INTEGER NOT NULL,
            updated_at TEXT NOT NULL DEFAULT (strftime('%Y-%m-%dT%H:%M:%fZ', 'now'))
        );

        CREATE INDEX IF NOT EXISTS idx_events_external_revision
            ON events(source, external_event_id, source_revision);
        CREATE INDEX IF NOT EXISTS idx_events_turn
            ON events(turn_id, event_id);
        CREATE INDEX IF NOT EXISTS idx_events_source_created
            ON events(source_created_at, event_id);
        CREATE INDEX IF NOT EXISTS idx_events_payload_sha256
            ON events(payload_sha256);
        CREATE INDEX IF NOT EXISTS idx_event_chunks_event
            ON event_chunks(event_id, chunk_ordinal);
        CREATE INDEX IF NOT EXISTS idx_projection_outbox_pending
            ON projection_outbox(status, next_retry_at, outbox_id);
        """
    )
    _ensure_projection_outbox_authority(conn)
    _backfill_legacy_evidence(conn)
    _install_immutability_triggers(conn)
    _install_hash_validation_triggers(conn)
    ensure_memory_claim_schema(conn)
    _install_memory_source_integrity_triggers(conn)


def _ensure_projection_outbox_authority(conn: sqlite3.Connection) -> None:
    """Make memory-job provenance explicit without changing transcript semantics."""

    columns = {
        str(row[1]): row
        for row in conn.execute("PRAGMA table_info(projection_outbox)").fetchall()
    }
    table_row = conn.execute(
        "SELECT sql FROM sqlite_master WHERE type = 'table' AND name = 'projection_outbox'"
    ).fetchone()
    table_sql = str(table_row[0]) if table_row is not None else ""
    if (
        "authority_kind" in columns
        and "authority_id" in columns
        and int(columns["event_seq"][3]) == 0
        and "authority_id NOT GLOB '*[^0-9]*'" in table_sql
        and "substr(authority_id, 1, 1) BETWEEN '1' AND '9'" in table_sql
    ):
        return

    if "authority_kind" in columns and "authority_id" in columns:
        invalid = conn.execute(
            """
            SELECT 1 FROM projection_outbox
            WHERE authority_kind NOT IN ('event', 'legacy_record')
               OR authority_id IS NULL
               OR length(authority_id) = 0
               OR authority_id GLOB '*[^0-9]*'
               OR substr(authority_id, 1, 1) NOT BETWEEN '1' AND '9'
            LIMIT 1
            """
        ).fetchone()
    else:
        invalid = conn.execute(
            "SELECT 1 FROM projection_outbox WHERE event_seq IS NULL LIMIT 1"
        ).fetchone()
    if invalid is not None:
        raise sqlite3.IntegrityError("projection outbox authority invalid")

    conn.execute("DROP INDEX IF EXISTS idx_projection_outbox_pending")
    conn.execute("ALTER TABLE projection_outbox RENAME TO projection_outbox_legacy")
    conn.execute(
        """
        CREATE TABLE projection_outbox (
            outbox_id INTEGER PRIMARY KEY AUTOINCREMENT,
            event_seq INTEGER,
            authority_kind TEXT NOT NULL
                CHECK (authority_kind IN ('event', 'legacy_record')),
            authority_id TEXT NOT NULL CHECK (
                length(authority_id) > 0
                AND authority_id NOT GLOB '*[^0-9]*'
                AND substr(authority_id, 1, 1) BETWEEN '1' AND '9'
            ),
            projection_kind TEXT NOT NULL,
            aggregate_id TEXT NOT NULL,
            aggregate_version INTEGER NOT NULL,
            payload_sha256 TEXT NOT NULL,
            status TEXT NOT NULL DEFAULT 'pending',
            attempt_count INTEGER NOT NULL DEFAULT 0,
            lease_owner TEXT,
            lease_until TEXT,
            next_retry_at TEXT,
            last_error TEXT,
            created_at TEXT NOT NULL DEFAULT (strftime('%Y-%m-%dT%H:%M:%fZ', 'now')),
            applied_at TEXT,
            UNIQUE(projection_kind, aggregate_id, aggregate_version)
        )
        """
    )
    if "authority_kind" in columns and "authority_id" in columns:
        authority_select = """
               CASE
                   WHEN authority_kind = 'legacy_record' THEN 'legacy_record'
                   ELSE 'event'
               END,
               COALESCE(NULLIF(trim(authority_id), ''), CAST(event_seq AS TEXT)),"""
    else:
        authority_select = """
               'event',
               CAST(event_seq AS TEXT),"""
    conn.execute(
        f"""
        INSERT INTO projection_outbox(
            outbox_id, event_seq, authority_kind, authority_id, projection_kind,
            aggregate_id, aggregate_version, payload_sha256, status,
            attempt_count, lease_owner, lease_until, next_retry_at, last_error,
            created_at, applied_at
        )
        SELECT outbox_id, event_seq,
               {authority_select}
               projection_kind, aggregate_id, aggregate_version, payload_sha256,
               status, attempt_count, lease_owner, lease_until, next_retry_at,
               last_error, created_at, applied_at
        FROM projection_outbox_legacy
        """
    )
    conn.execute("DROP TABLE projection_outbox_legacy")
    conn.execute(
        "CREATE INDEX idx_projection_outbox_pending "
        "ON projection_outbox(status, next_retry_at, outbox_id)"
    )


def _install_immutability_triggers(conn: sqlite3.Connection) -> None:
    _execute_script_without_implicit_commit(
        conn,
        """
        CREATE TRIGGER IF NOT EXISTS protect_events_update
        BEFORE UPDATE ON events
        WHEN OLD.chain_hash <> ''
        BEGIN
            SELECT RAISE(ABORT, 'events are immutable');
        END;

        CREATE TRIGGER IF NOT EXISTS protect_events_delete
        BEFORE DELETE ON events
        BEGIN
            SELECT RAISE(ABORT, 'events are immutable');
        END;

        CREATE TRIGGER IF NOT EXISTS protect_event_revisions_update
        BEFORE UPDATE ON event_revisions
        BEGIN
            SELECT RAISE(ABORT, 'event revisions are immutable');
        END;

        CREATE TRIGGER IF NOT EXISTS protect_event_revisions_delete
        BEFORE DELETE ON event_revisions
        BEGIN
            SELECT RAISE(ABORT, 'event revisions are immutable');
        END;

        CREATE TRIGGER IF NOT EXISTS protect_event_governance_update
        BEFORE UPDATE ON event_governance
        BEGIN
            SELECT RAISE(ABORT, 'event governance is immutable');
        END;

        CREATE TRIGGER IF NOT EXISTS protect_event_governance_delete
        BEFORE DELETE ON event_governance
        BEGIN
            SELECT RAISE(ABORT, 'event governance is immutable');
        END;
        """,
    )


def _install_hash_validation_triggers(conn: sqlite3.Connection) -> None:
    """Reject malformed hashes on the append-only revision table."""
    _execute_script_without_implicit_commit(conn, """
        CREATE TRIGGER IF NOT EXISTS validate_event_revisions_hashes_insert
        BEFORE INSERT ON event_revisions
        WHEN length(NEW.content_sha256) <> 64 OR NEW.content_sha256 GLOB '*[^0-9a-f]*'
        BEGIN SELECT RAISE(ABORT, 'event revision hashes must be lowercase hex'); END;
    """)


def _install_memory_source_integrity_triggers(conn: sqlite3.Connection) -> None:
    """Bind memory sources to their persisted authority commitments."""
    _execute_script_without_implicit_commit(conn, """
        CREATE TRIGGER IF NOT EXISTS validate_memory_event_source_insert
        BEFORE INSERT ON memory_claim_sources
        WHEN NEW.source_kind = 'event' AND NOT EXISTS (
            SELECT 1 FROM events e
            WHERE e.event_id = NEW.event_id
              AND e.source_revision = NEW.event_revision
              AND e.payload_sha256 = NEW.event_payload_sha256
        )
        BEGIN
            SELECT RAISE(ABORT, 'event source commitment mismatch');
        END;

        CREATE TRIGGER IF NOT EXISTS validate_memory_legacy_source_insert
        BEFORE INSERT ON memory_claim_sources
        WHEN NEW.source_kind = 'legacy_record' AND NOT EXISTS (
            SELECT 1 FROM memory_records r
            WHERE r.record_id = NEW.legacy_record_id
              AND r.status = 'approved'
              AND pcltm_sha256(r.content) = NEW.legacy_content_sha256
        )
        BEGIN
            SELECT RAISE(ABORT, 'legacy source commitment mismatch');
        END;

        CREATE TRIGGER IF NOT EXISTS memory_legacy_source_projection_guard_update
        BEFORE UPDATE ON memory_records
        WHEN EXISTS (
            SELECT 1
            FROM memory_claim_sources s
            JOIN memory_claim_versions v
              ON v.claim_version_id = s.claim_version_id
            JOIN memory_projection_guards pg
              ON pg.claim_id = v.claim_id
            WHERE s.source_kind = 'legacy_record'
              AND s.legacy_record_id IN (OLD.record_id, NEW.record_id)
        )
        BEGIN
            SELECT RAISE(ABORT, 'legacy memory source projection guarded');
        END;

        CREATE TRIGGER IF NOT EXISTS memory_legacy_source_projection_guard_delete
        BEFORE DELETE ON memory_records
        WHEN EXISTS (
            SELECT 1
            FROM memory_claim_sources s
            JOIN memory_claim_versions v
              ON v.claim_version_id = s.claim_version_id
            JOIN memory_projection_guards pg
              ON pg.claim_id = v.claim_id
            WHERE s.source_kind = 'legacy_record'
              AND s.legacy_record_id = OLD.record_id
        )
        BEGIN
            SELECT RAISE(ABORT, 'legacy memory source projection guarded');
        END;
    """)


def _backfill_legacy_evidence(conn: sqlite3.Connection) -> None:
    """Deterministically enroll legacy events into the v9 evidence chain."""
    rows = conn.execute(
        """
        SELECT e.*,
               i.external_id AS ingest_external_id,
               i.source_hash AS ingest_source_hash,
               i.payload_metadata AS ingest_payload_metadata,
               r.source_hash AS revision_source_hash,
               r.payload_metadata AS revision_payload_metadata
        FROM events e
        LEFT JOIN ingest_events i ON i.event_id = e.event_id
        LEFT JOIN event_revisions r
          ON r.event_id = e.event_id AND r.source_revision = e.source_revision
        ORDER BY e.event_id ASC
        """
    ).fetchall()
    if not rows:
        return
    previous_hash: str | None = None
    first_event_id: int | None = None
    last_event_id: int | None = None
    count = 0
    migrated_events: list[tuple[int, str, str, int]] = []
    for row in rows:
        event_id = int(row["event_id"])
        content = str(row["content"])
        recorded_at = str(row["recorded_at"] or row["created_at"])
        payload_sha256 = sha256_text(content)
        has_chain_envelope = bool(row["payload_sha256"] and row["chain_hash"])
        external_id = row["external_event_id"] if has_chain_envelope else row["ingest_external_id"]
        source_revision = int(row["source_revision"] or 1)
        source_created_at = row["source_created_at"] if has_chain_envelope else None
        payload_metadata = row["revision_payload_metadata"] if has_chain_envelope else row["ingest_payload_metadata"]
        if source_created_at is None and payload_metadata:
            try:
                import json
                metadata = json.loads(payload_metadata)
                source_created_at = normalize_source_created_at(
                    metadata.get("timestamp") or metadata.get("created_at")
                )
            except (TypeError, ValueError):
                source_created_at = None
        visibility = row["visibility"] if has_chain_envelope else row["inject_policy"]
        source_hash = row["revision_source_hash"] if has_chain_envelope else row["ingest_source_hash"]
        resolved_chain_hash = chain_hash(
            previous_chain_hash=previous_hash,
            event_id=event_id,
            session_id=str(row["session_id"]),
            conversation_id=str(row["conversation_id"]),
            platform=str(row["platform"]),
            role=str(row["role"]),
            source=str(row["source"]),
            payload_sha256=payload_sha256,
            recorded_at=recorded_at,
            schema_version=int(row["schema_version"] or 9),
            external_event_id=external_id,
            source_revision=source_revision,
            source_created_at=source_created_at,
            turn_id=row["turn_id"],
            parent_event_id=row["parent_event_id"],
            sensitivity=str(row["sensitivity"]),
            category=str(row["category"]),
            subcategory=str(row["subcategory"]),
            visibility=str(visibility),
            source_hash=source_hash,
        )

        existing_payload_sha256 = str(row["payload_sha256"] or "")
        existing_previous_hash = str(row["previous_chain_hash"] or "")
        existing_chain_hash = str(row["chain_hash"] or "")
        expected_previous_hash = previous_hash or ""
        validations = (
            ("payload_sha256_mismatch", existing_payload_sha256, payload_sha256),
            ("previous_chain_hash_mismatch", existing_previous_hash, expected_previous_hash),
            ("chain_hash_mismatch", existing_chain_hash, resolved_chain_hash),
        )
        for reason, existing_value, expected_value in validations:
            if existing_value and existing_value != expected_value:
                raise RuntimeError(
                    f"legacy evidence chain validation failed at event {event_id}: {reason}"
                )

        if existing_payload_sha256 and existing_chain_hash:
            previous_hash = existing_chain_hash
            first_event_id = event_id if first_event_id is None else first_event_id
            last_event_id = event_id
            count += 1
            continue

        conn.execute(
            """
            UPDATE events
            SET external_event_id = ?, source_created_at = ?, recorded_at = ?,
                payload_sha256 = ?, previous_chain_hash = ?, chain_hash = ?,
                source_revision = 1, visibility = inject_policy, schema_version = 9
            WHERE event_id = ?
            """,
            (
                external_id, source_created_at, recorded_at, payload_sha256,
                previous_hash, resolved_chain_hash, event_id,
            ),
        )
        if external_id is not None:
            conn.execute(
                """
                INSERT OR IGNORE INTO event_revisions (
                    event_id, source_revision, source_hash, content_sha256,
                    payload_metadata
                ) VALUES (?, 1, ?, ?, ?)
                """,
                (
                    event_id, str(row["ingest_source_hash"] or "legacy-unknown"),
                    payload_sha256, str(row["ingest_payload_metadata"] or "{}"),
                ),
            )
        previous_hash = resolved_chain_hash
        first_event_id = event_id if first_event_id is None else first_event_id
        last_event_id = event_id
        count += 1
        migrated_events.append((event_id, content, payload_sha256, 1))
    for event_id, content, payload_sha256, source_revision in migrated_events:
        chunks = chunk_transcript(content)
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
        for projection_kind in ("transcript_chunks", "transcript_fts"):
            conn.execute(
                """
                INSERT OR IGNORE INTO projection_outbox (
                    event_seq, authority_kind, authority_id, projection_kind,
                    aggregate_id, aggregate_version, payload_sha256, status,
                    applied_at
                ) VALUES (?, 'event', ?, ?, ?, ?, ?, 'applied',
                          strftime('%Y-%m-%dT%H:%M:%fZ', 'now'))
                """,
                (
                    event_id, str(event_id), projection_kind, str(event_id),
                    source_revision, payload_sha256,
                ),
            )
    conn.execute(
        """
        INSERT INTO event_chain_state (
            chain_name, first_event_id, last_event_id, event_count,
            tip_hash, schema_version
        ) VALUES ('events-v1', ?, ?, ?, ?, 9)
        ON CONFLICT(chain_name) DO UPDATE SET
            first_event_id=excluded.first_event_id,
            last_event_id=excluded.last_event_id,
            event_count=excluded.event_count,
            tip_hash=excluded.tip_hash,
            schema_version=excluded.schema_version,
            updated_at=strftime('%Y-%m-%dT%H:%M:%fZ', 'now')
        """,
        (first_event_id, last_event_id, count, previous_hash or ""),
    )
