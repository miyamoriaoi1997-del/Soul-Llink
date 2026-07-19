"""SQLite-backed immutable event store for PCLTM."""

from __future__ import annotations

import re
import sqlite3
import json
from pathlib import Path
from typing import Any

from .classifier import EventClassifier
from .evidence_chain import chain_hash, normalize_source_created_at, sha256_text
from .ledger_schema import ensure_evidence_ledger_schema
from .projection_outbox import enqueue_event_projections
from .secret_policy import evaluate_memory_write, redact_secrets


_SECRET_ASSIGNMENT_RE = re.compile(
    r"(?i)\b([A-Z0-9_]*(?:SECRET|TOKEN|API[_-]?KEY|PASSWORD|PASS)[A-Z0-9_]*)\s*=\s*[^\s,;]+"
)

CURRENT_SCHEMA_VERSION = 9


class EventStore:
    """Persist raw conversation/tool events with persona-aware metadata."""

    def __init__(self, db_path: str | Path):
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._conn = sqlite3.connect(self.db_path)
        self._conn.row_factory = sqlite3.Row
        self._conn.execute("PRAGMA busy_timeout=5000")
        self._bootstrap()

    def _bootstrap(self) -> None:
        self._conn.execute("PRAGMA journal_mode=WAL")
        self._conn.execute("PRAGMA foreign_keys=ON")
        try:
            self._conn.execute("BEGIN IMMEDIATE")
            self._conn.execute(
                """
                CREATE TABLE IF NOT EXISTS schema_migrations (
                    version INTEGER PRIMARY KEY,
                    applied_at TEXT NOT NULL DEFAULT (strftime('%Y-%m-%dT%H:%M:%fZ', 'now'))
                )
                """
            )
            self._apply_schema()
            self._conn.commit()
        except BaseException:
            self._conn.rollback()
            self._conn.close()
            raise

    def _apply_schema(self) -> None:
        self._create_core_tables()
        self._create_short_term_tables()
        self._create_dac_tables()
        self._ensure_event_classification_columns()
        self._ensure_memory_record_columns()
        ensure_evidence_ledger_schema(self._conn)
        self._create_fts_tables()
        self._ensure_fts_populated()
        for version in range(1, CURRENT_SCHEMA_VERSION + 1):
            self._conn.execute("INSERT OR IGNORE INTO schema_migrations (version) VALUES (?)", (version,))

    def _create_core_tables(self) -> None:
        self._conn.execute(
            """
            CREATE TABLE IF NOT EXISTS events (
                event_id INTEGER PRIMARY KEY AUTOINCREMENT,
                session_id TEXT NOT NULL,
                conversation_id TEXT NOT NULL,
                platform TEXT NOT NULL,
                role TEXT NOT NULL,
                source TEXT NOT NULL,
                content TEXT NOT NULL,
                persona_mode TEXT,
                route_bucket TEXT,
                model_hint TEXT,
                sensitivity TEXT NOT NULL DEFAULT 'normal',
                category TEXT NOT NULL DEFAULT 'unknown',
                subcategory TEXT NOT NULL DEFAULT 'unknown',
                inject_policy TEXT NOT NULL DEFAULT 'retrieve_only',
                classification_confidence REAL NOT NULL DEFAULT 0.0,
                classifier_version TEXT NOT NULL DEFAULT 'unknown',
                created_at TEXT NOT NULL DEFAULT (strftime('%Y-%m-%dT%H:%M:%fZ', 'now'))
            )
            """
        )
        self._conn.execute(
            "CREATE INDEX IF NOT EXISTS idx_events_scope ON events (session_id, conversation_id, platform, source, persona_mode, sensitivity)"
        )
        self._conn.execute(
            """
            CREATE TABLE IF NOT EXISTS summary_nodes (
                node_id INTEGER PRIMARY KEY AUTOINCREMENT,
                depth INTEGER NOT NULL,
                summary TEXT NOT NULL,
                expand_hint TEXT,
                sensitivity TEXT NOT NULL DEFAULT 'normal',
                inject_policy TEXT NOT NULL DEFAULT 'retrieve_only',
                created_at TEXT NOT NULL DEFAULT (strftime('%Y-%m-%dT%H:%M:%fZ', 'now'))
            )
            """
        )
        self._conn.execute(
            """
            CREATE TABLE IF NOT EXISTS summary_event_edges (
                node_id INTEGER NOT NULL,
                event_id INTEGER NOT NULL,
                position INTEGER NOT NULL,
                PRIMARY KEY (node_id, event_id),
                FOREIGN KEY (node_id) REFERENCES summary_nodes(node_id),
                FOREIGN KEY (event_id) REFERENCES events(event_id)
            )
            """
        )
        self._conn.execute(
            """
            CREATE TABLE IF NOT EXISTS summary_node_edges (
                parent_node_id INTEGER NOT NULL,
                child_node_id INTEGER NOT NULL,
                position INTEGER NOT NULL,
                PRIMARY KEY (parent_node_id, child_node_id),
                FOREIGN KEY (parent_node_id) REFERENCES summary_nodes(node_id),
                FOREIGN KEY (child_node_id) REFERENCES summary_nodes(node_id)
            )
            """
        )
        self._conn.execute(
            """
            CREATE TABLE IF NOT EXISTS memory_records (
                record_id INTEGER PRIMARY KEY AUTOINCREMENT,
                candidate_id TEXT NOT NULL UNIQUE,
                kind TEXT NOT NULL,
                target_file TEXT NOT NULL,
                content TEXT NOT NULL,
                confidence REAL NOT NULL,
                sensitivity TEXT NOT NULL,
                source_event_ids TEXT NOT NULL,
                source_node_ids TEXT NOT NULL,
                status TEXT NOT NULL DEFAULT 'pending',
                reviewer TEXT,
                reviewed_at TEXT,
                decision_reason TEXT,
                patch_suggestion TEXT,
                metadata TEXT NOT NULL DEFAULT '{}',
                created_at TEXT NOT NULL DEFAULT (strftime('%Y-%m-%dT%H:%M:%fZ', 'now'))
            )
            """
        )
        self._conn.execute(
            """
            CREATE TABLE IF NOT EXISTS ingest_events (
                ingest_id INTEGER PRIMARY KEY AUTOINCREMENT,
                external_id TEXT NOT NULL UNIQUE,
                source_hash TEXT NOT NULL,
                kind TEXT NOT NULL,
                event_id INTEGER NOT NULL,
                attachments TEXT NOT NULL DEFAULT '[]',
                payload_metadata TEXT NOT NULL DEFAULT '{}',
                created_at TEXT NOT NULL DEFAULT (strftime('%Y-%m-%dT%H:%M:%fZ', 'now'))
            )
            """
        )

    def _create_short_term_tables(self) -> None:
        self._conn.execute(
            """
            CREATE TABLE IF NOT EXISTS short_term_events (
                short_event_id INTEGER PRIMARY KEY AUTOINCREMENT,
                session_id TEXT NOT NULL,
                conversation_id TEXT NOT NULL,
                platform TEXT NOT NULL,
                role TEXT NOT NULL,
                source TEXT NOT NULL,
                content TEXT NOT NULL,
                persona_mode TEXT,
                route_bucket TEXT,
                model_hint TEXT,
                sensitivity TEXT NOT NULL DEFAULT 'normal',
                category TEXT NOT NULL DEFAULT 'unknown',
                subcategory TEXT NOT NULL DEFAULT 'unknown',
                inject_policy TEXT NOT NULL DEFAULT 'no_memory',
                retention_policy TEXT NOT NULL DEFAULT 'ttl',
                ttl_hours INTEGER NOT NULL DEFAULT 72,
                created_at TEXT NOT NULL DEFAULT (strftime('%Y-%m-%dT%H:%M:%fZ', 'now'))
            )
            """
        )
        self._conn.execute(
            "CREATE INDEX IF NOT EXISTS idx_short_term_events_ttl ON short_term_events (created_at, ttl_hours, source, persona_mode)"
        )

    def _create_dac_tables(self) -> None:
        self._conn.execute(
            """
            CREATE TABLE IF NOT EXISTS dac_raw_messages (
                raw_id INTEGER PRIMARY KEY AUTOINCREMENT,
                session_id TEXT NOT NULL,
                turn_id TEXT DEFAULT '',
                role TEXT NOT NULL,
                content TEXT NOT NULL,
                persona_mode TEXT DEFAULT '',
                source_platform TEXT DEFAULT '',
                sequence INTEGER NOT NULL DEFAULT 0,
                token_count INTEGER DEFAULT 0,
                sensitivity TEXT NOT NULL DEFAULT 'normal',
                inject_policy TEXT NOT NULL DEFAULT 'context_only',
                metadata TEXT NOT NULL DEFAULT '{}',
                created_at REAL NOT NULL
            )
            """
        )
        self._conn.execute(
            "CREATE INDEX IF NOT EXISTS idx_dac_raw_session_sequence ON dac_raw_messages (session_id, sequence, raw_id)"
        )
        self._conn.execute(
            """
            CREATE TABLE IF NOT EXISTS dac_summary_nodes (
                node_id INTEGER PRIMARY KEY AUTOINCREMENT,
                session_id TEXT NOT NULL,
                node_type TEXT NOT NULL DEFAULT 'summary',
                depth INTEGER NOT NULL DEFAULT 0,
                summary TEXT NOT NULL,
                token_count INTEGER DEFAULT 0,
                source_token_count INTEGER DEFAULT 0,
                source_type TEXT NOT NULL DEFAULT 'messages',
                source_ids TEXT NOT NULL DEFAULT '[]',
                persona_mode TEXT DEFAULT '',
                inject_policy TEXT NOT NULL DEFAULT 'retrieve_only',
                sensitivity TEXT NOT NULL DEFAULT 'normal',
                metadata TEXT NOT NULL DEFAULT '{}',
                created_at REAL NOT NULL,
                earliest_at REAL,
                latest_at REAL,
                expand_hint TEXT DEFAULT '',
                status TEXT NOT NULL DEFAULT 'active'
            )
            """
        )
        self._conn.execute(
            "CREATE INDEX IF NOT EXISTS idx_dac_nodes_session_depth ON dac_summary_nodes (session_id, depth, created_at)"
        )
        self._conn.execute(
            "CREATE INDEX IF NOT EXISTS idx_dac_nodes_status ON dac_summary_nodes (status, inject_policy, sensitivity)"
        )
        self._conn.execute(
            """
            CREATE TABLE IF NOT EXISTS dac_context_snapshots (
                snapshot_id INTEGER PRIMARY KEY AUTOINCREMENT,
                session_id TEXT NOT NULL,
                turn_id TEXT DEFAULT '',
                mode TEXT DEFAULT '',
                snapshot_type TEXT DEFAULT '',
                budget_tokens INTEGER DEFAULT 0,
                selected_node_ids TEXT NOT NULL DEFAULT '[]',
                selected_raw_ids TEXT NOT NULL DEFAULT '[]',
                fresh_tail_count INTEGER DEFAULT 0,
                metadata TEXT NOT NULL DEFAULT '{}',
                created_at REAL NOT NULL
            )
            """
        )
        snapshot_columns = {row[1] for row in self._conn.execute("PRAGMA table_info(dac_context_snapshots)")}
        snapshot_schema = {
            "turn_id": "TEXT DEFAULT ''",
            "mode": "TEXT DEFAULT ''",
            "snapshot_type": "TEXT DEFAULT ''",
            "budget_tokens": "INTEGER DEFAULT 0",
            "selected_node_ids": "TEXT NOT NULL DEFAULT '[]'",
            "selected_raw_ids": "TEXT NOT NULL DEFAULT '[]'",
            "fresh_tail_count": "INTEGER DEFAULT 0",
            "metadata": "TEXT NOT NULL DEFAULT '{}'",
            "created_at": "REAL NOT NULL DEFAULT 0",
        }
        for column, definition in snapshot_schema.items():
            if column not in snapshot_columns:
                self._conn.execute(f"ALTER TABLE dac_context_snapshots ADD COLUMN {column} {definition}")
        self._conn.execute(
            "CREATE INDEX IF NOT EXISTS idx_dac_snapshots_session ON dac_context_snapshots (session_id, created_at)"
        )
        self._conn.execute(
            "CREATE VIRTUAL TABLE IF NOT EXISTS dac_summary_nodes_fts USING fts5(summary)"
        )

    def _ensure_event_classification_columns(self) -> None:
        existing = {
            row["name"]
            for row in self._conn.execute("PRAGMA table_info(events)").fetchall()
        }
        columns = {
            "category": "TEXT NOT NULL DEFAULT 'unknown'",
            "subcategory": "TEXT NOT NULL DEFAULT 'unknown'",
            "inject_policy": "TEXT NOT NULL DEFAULT 'retrieve_only'",
            "classification_confidence": "REAL NOT NULL DEFAULT 0.0",
            "classifier_version": "TEXT NOT NULL DEFAULT 'unknown'",
        }
        for column, definition in columns.items():
            if column not in existing:
                self._conn.execute(f"ALTER TABLE events ADD COLUMN {column} {definition}")

    def _ensure_memory_record_columns(self) -> None:
        existing = {
            row["name"]
            for row in self._conn.execute("PRAGMA table_info(memory_records)").fetchall()
        }
        if "metadata" not in existing:
            self._conn.execute("ALTER TABLE memory_records ADD COLUMN metadata TEXT NOT NULL DEFAULT '{}'")

    def _create_fts_tables(self) -> None:
        self._conn.execute("CREATE VIRTUAL TABLE IF NOT EXISTS event_fts USING fts5(content)")
        self._conn.execute("CREATE VIRTUAL TABLE IF NOT EXISTS summary_fts USING fts5(summary)")

    def _ensure_fts_populated(self) -> None:
        event_count = self._conn.execute("SELECT count(*) AS n FROM events").fetchone()["n"]
        event_fts_count = self._conn.execute("SELECT count(*) AS n FROM event_fts").fetchone()["n"]
        summary_count = self._conn.execute("SELECT count(*) AS n FROM summary_nodes").fetchone()["n"]
        summary_fts_count = self._conn.execute("SELECT count(*) AS n FROM summary_fts").fetchone()["n"]
        if not self._fts_is_consistent():
            self.rebuild_fts()

    def _fts_is_consistent(self) -> bool:
        event_mismatch = self._conn.execute(
            """
            SELECT 1
            FROM events e
            LEFT JOIN event_fts f ON f.rowid = e.event_id
            WHERE f.rowid IS NULL OR f.content <> e.content
            LIMIT 1
            """
        ).fetchone()
        extra_event_fts = self._conn.execute(
            """
            SELECT 1 FROM event_fts f
            LEFT JOIN events e ON e.event_id = f.rowid
            WHERE e.event_id IS NULL
            LIMIT 1
            """
        ).fetchone()
        summary_mismatch = self._conn.execute(
            """
            SELECT 1
            FROM summary_nodes s
            LEFT JOIN summary_fts f ON f.rowid = s.node_id
            WHERE f.rowid IS NULL OR f.summary <> s.summary
            LIMIT 1
            """
        ).fetchone()
        extra_summary_fts = self._conn.execute(
            """
            SELECT 1 FROM summary_fts f
            LEFT JOIN summary_nodes s ON s.node_id = f.rowid
            WHERE s.node_id IS NULL
            LIMIT 1
            """
        ).fetchone()
        return not any((event_mismatch, extra_event_fts, summary_mismatch, extra_summary_fts))

    def rebuild_fts(self) -> dict[str, int]:
        self._conn.execute("DELETE FROM event_fts")
        self._conn.execute("DELETE FROM summary_fts")
        events = self._conn.execute("SELECT event_id, content FROM events ORDER BY event_id ASC").fetchall()
        for row in events:
            self._conn.execute("INSERT INTO event_fts(rowid, content) VALUES (?, ?)", (row["event_id"], row["content"]))
        summaries = self._conn.execute("SELECT node_id, summary FROM summary_nodes ORDER BY node_id ASC").fetchall()
        for row in summaries:
            self._conn.execute("INSERT INTO summary_fts(rowid, summary) VALUES (?, ?)", (row["node_id"], row["summary"]))
        return {"event_rows": len(events), "summary_rows": len(summaries)}

    def fts_counts(self) -> dict[str, int]:
        return {
            "events": int(self._conn.execute("SELECT count(*) AS n FROM events").fetchone()["n"]),
            "event_fts": int(self._conn.execute("SELECT count(*) AS n FROM event_fts").fetchone()["n"]),
            "summaries": int(self._conn.execute("SELECT count(*) AS n FROM summary_nodes").fetchone()["n"]),
            "summary_fts": int(self._conn.execute("SELECT count(*) AS n FROM summary_fts").fetchone()["n"]),
        }

    def schema_version(self) -> int:
        row = self._conn.execute("SELECT max(version) AS version FROM schema_migrations").fetchone()
        return int(row["version"] or 0)

    def claim_projection_jobs(
        self,
        *,
        worker_id: str,
        projection_kind: str,
        limit: int,
        now: str,
        lease_until: str,
    ) -> list[dict[str, Any]]:
        """Atomically claim pending or expired projection jobs."""
        try:
            self._conn.execute("BEGIN IMMEDIATE")
            rows = self._conn.execute(
                """
                SELECT * FROM projection_outbox
                WHERE projection_kind = ?
                  AND (
                    (status = 'pending' AND (next_retry_at IS NULL OR next_retry_at <= ?))
                    OR (status = 'processing' AND lease_until < ?)
                  )
                ORDER BY outbox_id ASC
                LIMIT ?
                """,
                (projection_kind, now, now, max(1, min(int(limit), 100))),
            ).fetchall()
            claimed: list[dict[str, Any]] = []
            for row in rows:
                self._conn.execute(
                    """
                    UPDATE projection_outbox
                    SET status = 'processing', lease_owner = ?, lease_until = ?,
                        attempt_count = attempt_count + 1
                    WHERE outbox_id = ?
                    """,
                    (worker_id, lease_until, row["outbox_id"]),
                )
                value = dict(row)
                value.update(
                    {
                        "status": "processing",
                        "lease_owner": worker_id,
                        "lease_until": lease_until,
                        "attempt_count": int(row["attempt_count"]) + 1,
                    }
                )
                claimed.append(value)
            self._conn.commit()
            return claimed
        except BaseException:
            self._conn.rollback()
            raise

    def ack_projection_job(self, outbox_id: int, *, worker_id: str, now: str) -> bool:
        """Idempotently mark a worker-owned projection job applied."""
        cur = self._conn.execute(
            """
            UPDATE projection_outbox
            SET status = 'applied', applied_at = COALESCE(applied_at, ?),
                lease_owner = NULL, lease_until = NULL, last_error = NULL
            WHERE outbox_id = ?
              AND ((status = 'processing' AND lease_owner = ?) OR status = 'applied')
            """,
            (now, int(outbox_id), worker_id),
        )
        self._conn.commit()
        return bool(cur.rowcount)

    def fail_projection_job(
        self,
        outbox_id: int,
        *,
        worker_id: str,
        error: str,
        now: str,
        next_retry_at: str,
        max_attempts: int = 5,
    ) -> dict[str, Any]:
        """Release a failed job for retry or move it to dead-letter."""
        row = self._conn.execute(
            "SELECT * FROM projection_outbox WHERE outbox_id = ?",
            (int(outbox_id),),
        ).fetchone()
        if row is None:
            raise KeyError(f"projection job not found: {outbox_id}")
        if row["status"] != "processing" or row["lease_owner"] != worker_id:
            raise RuntimeError("projection job is not owned by this worker")
        status = "dead_letter" if int(row["attempt_count"]) >= max(1, int(max_attempts)) else "pending"
        retry_at = None if status == "dead_letter" else next_retry_at
        self._conn.execute(
            """
            UPDATE projection_outbox
            SET status = ?, next_retry_at = ?, last_error = ?,
                lease_owner = NULL, lease_until = NULL
            WHERE outbox_id = ? AND status = 'processing' AND lease_owner = ?
            """,
            (status, retry_at, error, int(outbox_id), worker_id),
        )
        self._conn.commit()
        updated = self._conn.execute(
            "SELECT * FROM projection_outbox WHERE outbox_id = ?",
            (int(outbox_id),),
        ).fetchone()
        return dict(updated)

    def verify_event_chain(self) -> dict[str, Any]:
        """Verify payload hashes and the linked event chain in event order."""
        previous_chain_hash: str | None = None
        checked = 0
        rows = self._conn.execute(
            """
            SELECT e.event_id, e.session_id, e.conversation_id, e.platform,
                   e.role, e.source, e.content, e.recorded_at,
                   e.payload_sha256, e.previous_chain_hash, e.chain_hash,
                   e.schema_version, e.external_event_id, e.source_revision,
                   e.source_created_at, e.turn_id, e.parent_event_id,
                   e.sensitivity, e.category, e.subcategory, e.visibility,
                   r.source_hash, r.content_sha256 AS revision_content_sha256,
                   r.payload_metadata AS revision_payload_metadata
            FROM events e
            LEFT JOIN event_revisions r
              ON r.event_id = e.event_id AND r.source_revision = e.source_revision
            ORDER BY e.event_id ASC
            """
        ).fetchall()
        anchor = self._conn.execute(
            "SELECT * FROM event_chain_state WHERE chain_name = 'events-v1'"
        ).fetchone()
        if not rows:
            if anchor is None or int(anchor["event_count"]) == 0:
                return {"ok": True, "checked": 0, "first_invalid_event_id": None, "reason": None}
            return {"ok": False, "checked": 0, "first_invalid_event_id": None, "reason": "chain_anchor_mismatch"}
        for row in rows:
            event_id = int(row["event_id"])
            checked += 1
            if not row["payload_sha256"] or not row["chain_hash"]:
                return {
                    "ok": False,
                    "checked": checked,
                    "first_invalid_event_id": event_id,
                    "reason": "missing_chain_envelope",
                }
            payload_sha256 = sha256_text(str(row["content"]))
            if row["revision_content_sha256"] is not None and row["revision_content_sha256"] != payload_sha256:
                return {
                    "ok": False,
                    "checked": checked,
                    "first_invalid_event_id": event_id,
                    "reason": "revision_content_hash_mismatch",
                }
            if row["revision_payload_metadata"] is not None:
                try:
                    revision_metadata = json.loads(row["revision_payload_metadata"])
                except (TypeError, ValueError):
                    return {
                        "ok": False,
                        "checked": checked,
                        "first_invalid_event_id": event_id,
                        "reason": "revision_metadata_invalid",
                    }
                revision_time = normalize_source_created_at(
                    revision_metadata.get("timestamp") or revision_metadata.get("created_at")
                )
                if revision_time != normalize_source_created_at(row["source_created_at"]):
                    return {
                        "ok": False,
                        "checked": checked,
                        "first_invalid_event_id": event_id,
                        "reason": "revision_source_time_mismatch",
                    }
            if payload_sha256 != row["payload_sha256"]:
                return {
                    "ok": False,
                    "checked": checked,
                    "first_invalid_event_id": event_id,
                    "reason": "payload_sha256_mismatch",
                }
            if row["previous_chain_hash"] != previous_chain_hash:
                return {
                    "ok": False,
                    "checked": checked,
                    "first_invalid_event_id": event_id,
                    "reason": "previous_chain_hash_mismatch",
                }
            expected = chain_hash(
                previous_chain_hash=previous_chain_hash,
                event_id=event_id,
                session_id=str(row["session_id"]),
                conversation_id=str(row["conversation_id"]),
                platform=str(row["platform"]),
                role=str(row["role"]),
                source=str(row["source"]),
                payload_sha256=payload_sha256,
                recorded_at=str(row["recorded_at"]),
                schema_version=int(row["schema_version"]),
                external_event_id=row["external_event_id"],
                source_revision=int(row["source_revision"]),
                source_created_at=row["source_created_at"],
                turn_id=row["turn_id"],
                parent_event_id=row["parent_event_id"],
                sensitivity=str(row["sensitivity"]),
                category=str(row["category"]),
                subcategory=str(row["subcategory"]),
                visibility=str(row["visibility"]),
                source_hash=row["source_hash"],
            )
            if expected != row["chain_hash"]:
                return {
                    "ok": False,
                    "checked": checked,
                    "first_invalid_event_id": event_id,
                    "reason": "chain_hash_mismatch",
                }
            previous_chain_hash = str(row["chain_hash"])
        if anchor is None:
            return {
                "ok": False,
                "checked": checked,
                "first_invalid_event_id": None,
                "reason": "missing_chain_anchor",
            }
        if (
            int(anchor["first_event_id"]) != int(rows[0]["event_id"])
            or int(anchor["last_event_id"]) != int(rows[-1]["event_id"])
            or int(anchor["event_count"]) != len(rows)
            or str(anchor["tip_hash"]) != previous_chain_hash
        ):
            return {
                "ok": False,
                "checked": checked,
                "first_invalid_event_id": None,
                "reason": "chain_anchor_mismatch",
            }
        return {
            "ok": True,
            "checked": checked,
            "first_invalid_event_id": None,
            "reason": None,
        }

    def close(self) -> None:
        self._conn.close()

    def record_ingest_event(
        self,
        *,
        external_id: str,
        source_hash: str,
        kind: str,
        event_id: int,
        attachments: list[dict[str, Any]] | None = None,
        payload_metadata: dict[str, Any] | None = None,
    ) -> int:
        cur = self._conn.execute(
            "SELECT ingest_id FROM ingest_events WHERE external_id = ?",
            (external_id,),
        ).fetchone()
        if cur is not None:
            return int(cur["ingest_id"])
        insert = self._conn.execute(
            """
            INSERT INTO ingest_events (external_id, source_hash, kind, event_id, attachments, payload_metadata)
            VALUES (?, ?, ?, ?, ?, ?)
            """,
            (
                external_id,
                source_hash,
                kind,
                event_id,
                json.dumps(attachments or []),
                json.dumps(payload_metadata or {}),
            ),
        )
        self._conn.commit()
        return int(insert.lastrowid)

    def get_ingest_event(self, external_id: str) -> dict[str, Any]:
        row = self._conn.execute("SELECT * FROM ingest_events WHERE external_id = ?", (external_id,)).fetchone()
        if row is None:
            raise KeyError(f"ingest event not found: {external_id}")
        data = dict(row)
        data["attachments"] = json.loads(data["attachments"])
        data["payload_metadata"] = json.loads(data["payload_metadata"])
        return data

    def find_ingest_event(self, external_id: str) -> dict[str, Any] | None:
        row = self._conn.execute("SELECT * FROM ingest_events WHERE external_id = ?", (external_id,)).fetchone()
        if row is None:
            return None
        data = dict(row)
        data["attachments"] = json.loads(data["attachments"])
        data["payload_metadata"] = json.loads(data["payload_metadata"])
        return data

    def _insert_event_row(
        self,
        *,
        session_id: str,
        conversation_id: str,
        platform: str,
        role: str,
        source: str,
        content: str,
        persona_mode: str | None = None,
        route_bucket: str | None = None,
        model_hint: str | None = None,
        sensitivity: str = "normal",
        category: str | None = None,
        subcategory: str | None = None,
        inject_policy: str | None = None,
        classification_confidence: float | None = None,
        classifier_version: str | None = None,
        external_event_id: str | None = None,
        source_revision: int = 1,
        source_created_at: str | None = None,
        turn_id: str | None = None,
        parent_event_id: int | None = None,
        source_hash: str | None = None,
    ) -> int:
        """Insert one durable event and its FTS row without committing."""
        classification = EventClassifier().classify(
            role=role,
            source=source,
            content=content,
            persona_mode=persona_mode,
            sensitivity=sensitivity,
        )
        resolved_sensitivity = classification.sensitivity if sensitivity == "normal" else sensitivity
        cur = self._conn.execute(
            """
            INSERT INTO events (
                session_id, conversation_id, platform, role, source, content,
                persona_mode, route_bucket, model_hint, sensitivity,
                category, subcategory, inject_policy, classification_confidence, classifier_version
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                session_id, conversation_id, platform, role, source, content,
                persona_mode, route_bucket, model_hint, resolved_sensitivity,
                category or classification.category,
                subcategory or classification.subcategory,
                inject_policy or classification.inject_policy,
                classification_confidence if classification_confidence is not None else classification.confidence,
                classifier_version or classification.classifier_version,
            ),
        )
        event_id = int(cur.lastrowid)
        recorded_at = str(
            self._conn.execute(
                "SELECT created_at FROM events WHERE event_id = ?", (event_id,)
            ).fetchone()["created_at"]
        )
        payload_sha256 = sha256_text(content)
        previous_row = self._conn.execute(
            "SELECT chain_hash FROM events WHERE event_id < ? AND chain_hash <> '' ORDER BY event_id DESC LIMIT 1",
            (event_id,),
        ).fetchone()
        previous_chain_hash = str(previous_row["chain_hash"]) if previous_row is not None else None
        resolved_chain_hash = chain_hash(
            previous_chain_hash=previous_chain_hash,
            event_id=event_id,
            session_id=session_id,
            conversation_id=conversation_id,
            platform=platform,
            role=role,
            source=source,
            payload_sha256=payload_sha256,
            recorded_at=recorded_at,
            schema_version=CURRENT_SCHEMA_VERSION,
            external_event_id=external_event_id,
            source_revision=int(source_revision),
            source_created_at=source_created_at,
            turn_id=turn_id,
            parent_event_id=parent_event_id,
            sensitivity=resolved_sensitivity,
            category=category or classification.category,
            subcategory=subcategory or classification.subcategory,
            visibility=inject_policy or classification.inject_policy,
            source_hash=source_hash,
        )
        self._conn.execute(
            """
            UPDATE events
            SET external_event_id = ?, turn_id = ?, parent_event_id = ?,
                source_created_at = ?, recorded_at = ?, payload_sha256 = ?,
                previous_chain_hash = ?, chain_hash = ?, source_revision = ?,
                visibility = ?, schema_version = ?
            WHERE event_id = ?
            """,
            (
                external_event_id,
                turn_id,
                parent_event_id,
                source_created_at,
                recorded_at,
                payload_sha256,
                previous_chain_hash,
                resolved_chain_hash,
                int(source_revision),
                inject_policy or classification.inject_policy,
                CURRENT_SCHEMA_VERSION,
                event_id,
            ),
        )
        self._conn.execute("INSERT INTO event_fts(rowid, content) VALUES (?, ?)", (event_id, content))
        anchor = self._conn.execute(
            "SELECT first_event_id, event_count FROM event_chain_state WHERE chain_name = 'events-v1'"
        ).fetchone()
        first_event_id = event_id if anchor is None or anchor["first_event_id"] is None else int(anchor["first_event_id"])
        event_count = 1 if anchor is None else int(anchor["event_count"]) + 1
        self._conn.execute(
            """
            INSERT INTO event_chain_state (
                chain_name, first_event_id, last_event_id, event_count,
                tip_hash, schema_version
            ) VALUES ('events-v1', ?, ?, ?, ?, ?)
            ON CONFLICT(chain_name) DO UPDATE SET
                last_event_id = excluded.last_event_id,
                event_count = excluded.event_count,
                tip_hash = excluded.tip_hash,
                schema_version = excluded.schema_version,
                updated_at = strftime('%Y-%m-%dT%H:%M:%fZ', 'now')
            """,
            (
                first_event_id,
                event_id,
                event_count,
                resolved_chain_hash,
                CURRENT_SCHEMA_VERSION,
            ),
        )
        return event_id

    def ingest_external_event(
        self,
        *,
        external_id: str,
        source_hash: str,
        kind: str,
        attachments: list[dict[str, Any]] | None = None,
        payload_metadata: dict[str, Any] | None = None,
        **event: Any,
    ) -> tuple[int, bool]:
        """Atomically ingest one externally identified event.

        Returns ``(event_id, inserted)``. The unique external ID makes live
        sync and historical replay converge without duplicate event rows.
        """
        existing = self.find_ingest_event(external_id)
        if existing is not None:
            return int(existing["event_id"]), False
        try:
            self._conn.execute("BEGIN IMMEDIATE")
            existing_row = self._conn.execute(
                "SELECT event_id FROM ingest_events WHERE external_id = ?", (external_id,)
            ).fetchone()
            if existing_row is not None:
                self._conn.rollback()
                return int(existing_row["event_id"]), False
            event_id = self._insert_event_row(
                **event,
                external_event_id=external_id,
                source_created_at=normalize_source_created_at(
                    (payload_metadata or {}).get("timestamp") or (payload_metadata or {}).get("created_at")
                ),
                source_hash=source_hash,
            )
            self._conn.execute(
                """
                INSERT INTO event_revisions (
                    event_id, source_revision, source_hash, content_sha256, payload_metadata
                ) VALUES (?, 1, ?, ?, ?)
                """,
                (
                    event_id,
                    source_hash,
                    sha256_text(str(event["content"])),
                    json.dumps(payload_metadata or {}, ensure_ascii=False),
                ),
            )
            self._conn.execute(
                """
                INSERT INTO ingest_events (external_id, source_hash, kind, event_id, attachments, payload_metadata)
                VALUES (?, ?, ?, ?, ?, ?)
                """,
                (
                    external_id, source_hash, kind, event_id,
                    json.dumps(attachments or [], ensure_ascii=False),
                    json.dumps(payload_metadata or {}, ensure_ascii=False),
                ),
            )
            event_row = self._conn.execute(
                "SELECT payload_sha256, source_revision FROM events WHERE event_id = ?",
                (event_id,),
            ).fetchone()
            enqueue_event_projections(
                self._conn,
                event_id=event_id,
                aggregate_version=int(event_row["source_revision"]),
                payload_sha256=str(event_row["payload_sha256"]),
            )
            self._conn.commit()
            return event_id, True
        except BaseException:
            self._conn.rollback()
            raise

    def upsert_external_event(self, *, external_id: str, source_hash: str, kind: str, attachments: list[dict[str, Any]] | None = None, payload_metadata: dict[str, Any] | None = None, **event: Any) -> tuple[int, str]:
        """Atomically append a new immutable revision for changed source data."""
        existing = self.find_ingest_event(external_id)
        if existing is None:
            event_id, inserted = self.ingest_external_event(
                external_id=external_id,
                source_hash=source_hash,
                kind=kind,
                attachments=attachments,
                payload_metadata=payload_metadata,
                **event,
            )
            return event_id, "inserted" if inserted else "existing"
        if existing["source_hash"] == source_hash:
            return int(existing["event_id"]), "existing"
        try:
            self._conn.execute("BEGIN IMMEDIATE")
            current = self._conn.execute(
                "SELECT source_hash, event_id FROM ingest_events WHERE external_id = ?",
                (external_id,),
            ).fetchone()
            if current is None:
                self._conn.rollback()
                return self.upsert_external_event(
                    external_id=external_id,
                    source_hash=source_hash,
                    kind=kind,
                    attachments=attachments,
                    payload_metadata=payload_metadata,
                    **event,
                )
            if current["source_hash"] == source_hash:
                self._conn.rollback()
                return int(current["event_id"]), "existing"
            previous_event_id = int(current["event_id"])
            previous_event = self._conn.execute(
                "SELECT source_revision FROM events WHERE event_id = ?",
                (previous_event_id,),
            ).fetchone()
            source_revision = int(previous_event["source_revision"] or 1) + 1
            event_id = self._insert_event_row(
                **event,
                external_event_id=external_id,
                source_revision=source_revision,
                source_created_at=normalize_source_created_at(
                    (payload_metadata or {}).get("timestamp") or (payload_metadata or {}).get("created_at")
                ),
                source_hash=source_hash,
            )
            self._conn.execute(
                """
                INSERT INTO event_revisions (
                    event_id, source_revision, source_hash, content_sha256, payload_metadata
                ) VALUES (?, ?, ?, ?, ?)
                """,
                (
                    event_id,
                    source_revision,
                    source_hash,
                    sha256_text(str(event["content"])),
                    json.dumps(payload_metadata or {}, ensure_ascii=False),
                ),
            )
            self._conn.execute(
                """
                INSERT INTO event_governance (
                    event_id, action, previous_state, new_state, actor, reason
                ) VALUES (?, 'supersede', 'active', 'superseded', 'source_sync', ?)
                """,
                (previous_event_id, f"superseded by event {event_id}"),
            )
            self._conn.execute(
                """
                UPDATE ingest_events
                SET source_hash = ?, kind = ?, event_id = ?, attachments = ?, payload_metadata = ?
                WHERE external_id = ?
                """,
                (
                    source_hash,
                    kind,
                    event_id,
                    json.dumps(attachments or [], ensure_ascii=False),
                    json.dumps(payload_metadata or {}, ensure_ascii=False),
                    external_id,
                ),
            )
            event_row = self._conn.execute(
                "SELECT payload_sha256 FROM events WHERE event_id = ?",
                (event_id,),
            ).fetchone()
            enqueue_event_projections(
                self._conn,
                event_id=event_id,
                aggregate_version=source_revision,
                payload_sha256=str(event_row["payload_sha256"]),
            )
            self._conn.commit()
            return event_id, "updated"
        except BaseException:
            self._conn.rollback()
            raise

    def append_event(
        self,
        *,
        session_id: str,
        conversation_id: str,
        platform: str,
        role: str,
        source: str,
        content: str,
        persona_mode: str | None = None,
        route_bucket: str | None = None,
        model_hint: str | None = None,
        sensitivity: str = "normal",
        category: str | None = None,
        subcategory: str | None = None,
        inject_policy: str | None = None,
        classification_confidence: float | None = None,
        classifier_version: str | None = None,
    ) -> int:
        classification = EventClassifier().classify(
            role=role, source=source, content=content,
            persona_mode=persona_mode, sensitivity=sensitivity,
        )
        if classification.inject_policy in {"drop", "no_memory"} and category is None and subcategory is None and inject_policy is None:
            return self.append_short_term_event(
                session_id=session_id, conversation_id=conversation_id,
                platform=platform, role=role, source=source, content=content,
                persona_mode=persona_mode, route_bucket=route_bucket,
                model_hint=model_hint,
                sensitivity=classification.sensitivity if sensitivity == "normal" else sensitivity,
                category=classification.category, subcategory=classification.subcategory,
                inject_policy=classification.inject_policy,
            ) * 0
        try:
            self._conn.execute("BEGIN IMMEDIATE")
            event_id = self._insert_event_row(
                session_id=session_id, conversation_id=conversation_id,
                platform=platform, role=role, source=source, content=content,
                persona_mode=persona_mode, route_bucket=route_bucket,
                model_hint=model_hint, sensitivity=sensitivity, category=category,
                subcategory=subcategory, inject_policy=inject_policy,
                classification_confidence=classification_confidence,
                classifier_version=classifier_version,
            )
            event_row = self._conn.execute(
                "SELECT payload_sha256, source_revision FROM events WHERE event_id = ?",
                (event_id,),
            ).fetchone()
            enqueue_event_projections(
                self._conn,
                event_id=event_id,
                aggregate_version=int(event_row["source_revision"]),
                payload_sha256=str(event_row["payload_sha256"]),
            )
            self._conn.commit()
            return event_id
        except BaseException:
            self._conn.rollback()
            raise

    def append_short_term_event(
        self,
        *,
        session_id: str,
        conversation_id: str,
        platform: str,
        role: str,
        source: str,
        content: str,
        persona_mode: str | None = None,
        route_bucket: str | None = None,
        model_hint: str | None = None,
        sensitivity: str = "normal",
        category: str = "unknown",
        subcategory: str = "unknown",
        inject_policy: str = "no_memory",
        retention_policy: str = "ttl",
        ttl_hours: int = 72,
        created_at: str | None = None,
    ) -> int:
        cur = self._conn.execute(
            """
            INSERT INTO short_term_events (
                session_id, conversation_id, platform, role, source, content,
                persona_mode, route_bucket, model_hint, sensitivity,
                category, subcategory, inject_policy, retention_policy, ttl_hours, created_at
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, COALESCE(?, strftime('%Y-%m-%dT%H:%M:%fZ', 'now')))
            """,
            (
                session_id,
                conversation_id,
                platform,
                role,
                source,
                content,
                persona_mode,
                route_bucket,
                model_hint,
                sensitivity,
                category,
                subcategory,
                inject_policy,
                retention_policy,
                int(ttl_hours),
                created_at,
            ),
        )
        self._conn.commit()
        return int(cur.lastrowid)

    def list_short_term_events(self, *, limit: int = 50) -> list[dict[str, Any]]:
        rows = self._conn.execute(
            "SELECT * FROM short_term_events ORDER BY short_event_id DESC LIMIT ?",
            (max(1, min(int(limit), 500)),),
        ).fetchall()
        return [dict(row) for row in rows]

    def prune_short_term_events(self, *, now: str | None = None) -> dict[str, int]:
        now_expr = "?" if now is not None else "strftime('%Y-%m-%dT%H:%M:%fZ', 'now')"
        params = [now] if now is not None else []
        cur = self._conn.execute(
            f"""
            DELETE FROM short_term_events
            WHERE retention_policy = 'ttl'
              AND datetime(created_at, '+' || ttl_hours || ' hours') <= datetime({now_expr})
            """,
            params,
        )
        self._conn.commit()
        return {"deleted": int(cur.rowcount if cur.rowcount is not None else 0)}

    def get_event(self, event_id: int) -> dict[str, Any]:
        row = self._conn.execute(
            "SELECT * FROM events WHERE event_id = ?",
            (event_id,),
        ).fetchone()
        if row is None:
            raise KeyError(f"event not found: {event_id}")
        return dict(row)

    def search_events(
        self,
        query: str,
        *,
        session_id: str | None = None,
        conversation_id: str | None = None,
        platform: str | None = None,
        persona_mode: str | None = None,
        source: str | None = None,
        limit: int = 10,
        include_sensitive: bool = False,
    ) -> list[dict[str, Any]]:
        """Search raw events with explicit scope filters and safe snippets."""
        clauses = ["event_fts MATCH ?"]
        params: list[Any] = [self._fts_query(query)]
        for column, value in (
            ("session_id", session_id),
            ("conversation_id", conversation_id),
            ("platform", platform),
            ("persona_mode", persona_mode),
            ("source", source),
        ):
            if value is not None:
                clauses.append(f"e.{column} = ?")
                params.append(value)
        if not include_sensitive:
            clauses.append("e.sensitivity NOT IN ('private', 'secret', 'restricted')")
        params.append(max(1, min(int(limit), 100)))
        try:
            rows = self._conn.execute(
                f"""
                SELECT e.* FROM event_fts
                JOIN events e ON e.event_id = event_fts.rowid
                WHERE {' AND '.join(clauses)}
                ORDER BY e.event_id DESC
                LIMIT ?
                """,
                params,
            ).fetchall()
        except sqlite3.OperationalError:
            return self._search_events_like(
                query,
                session_id=session_id,
                conversation_id=conversation_id,
                platform=platform,
                persona_mode=persona_mode,
                source=source,
                limit=limit,
                include_sensitive=include_sensitive,
            )
        if not rows:
            like_rows = self._search_events_like(
                query,
                session_id=session_id,
                conversation_id=conversation_id,
                platform=platform,
                persona_mode=persona_mode,
                source=source,
                limit=limit,
                include_sensitive=include_sensitive,
            )
            if len(query) > 1 and any("\u4e00" <= char <= "\u9fff" for char in query):
                return like_rows
        return [self._result_from_row(row) for row in rows]

    def _search_events_like(
        self,
        query: str,
        *,
        session_id: str | None = None,
        conversation_id: str | None = None,
        platform: str | None = None,
        persona_mode: str | None = None,
        source: str | None = None,
        limit: int = 10,
        include_sensitive: bool = False,
    ) -> list[dict[str, Any]]:
        clauses = ["content LIKE ?"]
        params: list[Any] = [f"%{query}%"]
        for column, value in (
            ("session_id", session_id),
            ("conversation_id", conversation_id),
            ("platform", platform),
            ("persona_mode", persona_mode),
            ("source", source),
        ):
            if value is not None:
                clauses.append(f"{column} = ?")
                params.append(value)
        if not include_sensitive:
            clauses.append("sensitivity NOT IN ('private', 'secret', 'restricted')")
        params.append(max(1, min(int(limit), 100)))
        rows = self._conn.execute(
            f"""
            SELECT * FROM events
            WHERE {' AND '.join(clauses)}
            ORDER BY event_id DESC
            LIMIT ?
            """,
            params,
        ).fetchall()
        return [self._result_from_row(row) for row in rows]

    def search_summaries(
        self,
        query: str,
        *,
        limit: int = 10,
        include_sensitive: bool = False,
    ) -> list[dict[str, Any]]:
        clauses = ["summary_fts MATCH ?"]
        params: list[Any] = [self._fts_query(query)]
        if not include_sensitive:
            clauses.append("s.sensitivity NOT IN ('private', 'secret', 'restricted')")
        params.append(max(1, min(int(limit), 100)))
        try:
            rows = self._conn.execute(
                f"""
                SELECT s.* FROM summary_fts
                JOIN summary_nodes s ON s.node_id = summary_fts.rowid
                WHERE {' AND '.join(clauses)}
                ORDER BY s.node_id DESC
                LIMIT ?
                """,
                params,
            ).fetchall()
        except sqlite3.OperationalError:
            return self._search_summaries_like(query, limit=limit, include_sensitive=include_sensitive)
        return [dict(row) for row in rows]

    def _search_summaries_like(
        self,
        query: str,
        *,
        limit: int = 10,
        include_sensitive: bool = False,
    ) -> list[dict[str, Any]]:
        clauses = ["summary LIKE ?"]
        params: list[Any] = [f"%{query}%"]
        if not include_sensitive:
            clauses.append("sensitivity NOT IN ('private', 'secret', 'restricted')")
        params.append(max(1, min(int(limit), 100)))
        rows = self._conn.execute(
            f"""
            SELECT * FROM summary_nodes
            WHERE {' AND '.join(clauses)}
            ORDER BY node_id DESC
            LIMIT ?
            """,
            params,
        ).fetchall()
        return [dict(row) for row in rows]

    def list_events(
        self,
        *,
        session_id: str | None = None,
        conversation_id: str | None = None,
        platform: str | None = None,
        persona_mode: str | None = None,
        source: str | None = None,
        limit: int = 100,
    ) -> list[dict[str, Any]]:
        clauses = []
        params: list[Any] = []
        for column, value in (
            ("session_id", session_id),
            ("conversation_id", conversation_id),
            ("platform", platform),
            ("persona_mode", persona_mode),
            ("source", source),
        ):
            if value is not None:
                clauses.append(f"{column} = ?")
                params.append(value)
        params.append(max(1, min(int(limit), 500)))
        where = f"WHERE {' AND '.join(clauses)}" if clauses else ""
        rows = self._conn.execute(
            f"SELECT * FROM events {where} ORDER BY event_id ASC LIMIT ?",
            params,
        ).fetchall()
        return [dict(row) for row in rows]

    def list_memory_records(self, *, status: str | None = None, target_file: str | None = None) -> list[dict[str, Any]]:
        clauses = []
        params: list[Any] = []
        if status is not None:
            clauses.append("status = ?")
            params.append(status)
        if target_file is not None:
            clauses.append("target_file = ?")
            params.append(target_file)
        where = f"WHERE {' AND '.join(clauses)}" if clauses else ""
        rows = self._conn.execute(
            f"SELECT * FROM memory_records {where} ORDER BY record_id ASC",
            params,
        ).fetchall()
        return [self._memory_record_from_row(row) for row in rows]

    def add_memory_record(
        self,
        *,
        candidate_id: str,
        kind: str,
        target_file: str,
        content: str,
        confidence: float,
        sensitivity: str,
        source_event_ids: list[int] | None = None,
        source_node_ids: list[int] | None = None,
        status: str = "approved",
        reviewer: str | None = None,
        decision_reason: str | None = None,
        patch_suggestion: str | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> tuple[int, bool]:
        metadata = dict(metadata or {})
        decision = evaluate_memory_write(content, target_file=target_file)
        if decision.action == "reject":
            content = decision.sanitized_content or "[REJECTED_SECRET_MEMORY]"
            sensitivity = "secret"
            status = "rejected"
            decision_reason = decision_reason or decision.reason
        else:
            content = decision.sanitized_content or content
            sensitivity = decision.sensitivity if sensitivity == "normal" else sensitivity
        metadata.update(decision.metadata)
        content = redact_secrets(content)
        cur = self._conn.execute(
            "SELECT record_id FROM memory_records WHERE candidate_id = ?",
            (candidate_id,),
        ).fetchone()
        if cur is not None:
            return int(cur["record_id"]), False
        insert = self._conn.execute(
            """
            INSERT INTO memory_records (
                candidate_id, kind, target_file, content, confidence, sensitivity,
                source_event_ids, source_node_ids, status, reviewer, reviewed_at,
                decision_reason, patch_suggestion, metadata
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, strftime('%Y-%m-%dT%H:%M:%fZ', 'now'), ?, ?, ?)
            """,
            (
                candidate_id,
                kind,
                target_file,
                content,
                confidence,
                sensitivity,
                json.dumps(source_event_ids or []),
                json.dumps(source_node_ids or []),
                status,
                reviewer,
                decision_reason,
                patch_suggestion,
                json.dumps(metadata or {}, ensure_ascii=False),
            ),
        )
        self._conn.commit()
        return int(insert.lastrowid), True

    def enqueue_candidate(self, candidate: dict[str, Any]) -> int:
        candidate = dict(candidate)
        decision = evaluate_memory_write(str(candidate.get("content") or ""), target_file=str(candidate.get("target_file") or ""))
        if decision.action == "reject":
            candidate["content"] = decision.sanitized_content or "[REJECTED_SECRET_MEMORY]"
            candidate["sensitivity"] = "secret"
        else:
            candidate["content"] = redact_secrets(decision.sanitized_content or str(candidate.get("content") or ""))
            if candidate.get("sensitivity") == "normal":
                candidate["sensitivity"] = decision.sensitivity
        cur = self._conn.execute(
            "SELECT record_id FROM memory_records WHERE candidate_id = ?",
            (candidate["candidate_id"],),
        ).fetchone()
        if cur is not None:
            return int(cur["record_id"])
        insert = self._conn.execute(
            """
            INSERT INTO memory_records (
                candidate_id, kind, target_file, content, confidence, sensitivity,
                source_event_ids, source_node_ids, status
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, 'pending')
            """,
            (
                candidate["candidate_id"],
                candidate["kind"],
                candidate["target_file"],
                candidate["content"],
                candidate["confidence"],
                candidate["sensitivity"],
                json.dumps(candidate["source_event_ids"]),
                json.dumps(candidate["source_node_ids"]),
            ),
        )
        self._conn.commit()
        return int(insert.lastrowid)

    def list_candidate_queue(self, *, status: str | None = None) -> list[dict[str, Any]]:
        clauses = []
        params: list[Any] = []
        if status is not None:
            clauses.append("status = ?")
            params.append(status)
        where = f"WHERE {' AND '.join(clauses)}" if clauses else ""
        rows = self._conn.execute(
            f"SELECT * FROM memory_records {where} ORDER BY record_id ASC",
            params,
        ).fetchall()
        return [self._memory_record_from_row(row) for row in rows]

    def review_candidate(
        self,
        record_id: int,
        *,
        decision: str,
        reviewer: str,
        decision_reason: str,
    ) -> dict[str, Any]:
        if decision not in {"approved", "rejected"}:
            raise ValueError("decision must be approved or rejected")
        row = self._conn.execute("SELECT * FROM memory_records WHERE record_id = ?", (record_id,)).fetchone()
        if row is None:
            raise KeyError(f"memory record not found: {record_id}")
        patch_suggestion = None
        content_update_sql = ""
        content_update_params: tuple[Any, ...] = ()
        if decision == "approved":
            policy = evaluate_memory_write(str(row["content"] or ""), target_file=str(row["target_file"] or ""))
            if policy.action == "reject":
                decision = "rejected"
                decision_reason = decision_reason or policy.reason
            else:
                safe_content = redact_secrets(policy.sanitized_content or str(row["content"] or ""))
                patch_suggestion = f"Append to {row['target_file']}:\n{safe_content}"
                content_update_sql = ", content = ?, sensitivity = ?"
                content_update_params = (safe_content, policy.sensitivity)
        self._conn.execute(
            f"""
            UPDATE memory_records
            SET status = ?, reviewer = ?, reviewed_at = strftime('%Y-%m-%dT%H:%M:%fZ', 'now'),
                decision_reason = ?, patch_suggestion = ?{content_update_sql}
            WHERE record_id = ?
            """,
            (decision, reviewer, decision_reason, patch_suggestion, *content_update_params, record_id),
        )
        self._conn.commit()
        updated = self._conn.execute("SELECT * FROM memory_records WHERE record_id = ?", (record_id,)).fetchone()
        return self._memory_record_from_row(updated)

    def _memory_record_from_row(self, row: sqlite3.Row) -> dict[str, Any]:
        record = dict(row)
        record["source_event_ids"] = json.loads(record["source_event_ids"])
        record["source_node_ids"] = json.loads(record["source_node_ids"])
        record["metadata"] = json.loads(record.get("metadata") or "{}")
        return record

    def create_summary_from_events(
        self,
        source_event_ids: list[int],
        *,
        summarizer: Any,
        max_chars: int = 1200,
        depth: int = 0,
        expand_hint: str | None = None,
        inject_policy: str = "retrieve_only",
    ) -> int:
        """Create a summary node from raw events using a bounded fallback summarizer."""
        events = [self.get_event(event_id) for event_id in source_event_ids]
        summary = summarizer.summarize(events, max_chars=max_chars)
        return self.create_summary_node(
            depth=depth,
            summary=summary,
            source_event_ids=source_event_ids,
            expand_hint=expand_hint,
            inject_policy=inject_policy,
        )

    def create_summary_node(
        self,
        *,
        depth: int,
        summary: str,
        source_event_ids: list[int] | None = None,
        source_node_ids: list[int] | None = None,
        expand_hint: str | None = None,
        sensitivity: str | None = None,
        inject_policy: str = "retrieve_only",
    ) -> int:
        """Create a summary DAG node without modifying raw events."""
        source_event_ids = source_event_ids or []
        source_node_ids = source_node_ids or []
        resolved_sensitivity = sensitivity or self._max_source_sensitivity(source_event_ids, source_node_ids)
        summary = redact_secrets(summary)
        cur = self._conn.execute(
            """
            INSERT INTO summary_nodes (depth, summary, expand_hint, sensitivity, inject_policy)
            VALUES (?, ?, ?, ?, ?)
            """,
            (depth, summary, expand_hint, resolved_sensitivity, inject_policy),
        )
        node_id = int(cur.lastrowid)
        for position, event_id in enumerate(source_event_ids):
            self._conn.execute(
                "INSERT INTO summary_event_edges (node_id, event_id, position) VALUES (?, ?, ?)",
                (node_id, event_id, position),
            )
        for position, child_node_id in enumerate(source_node_ids):
            self._conn.execute(
                "INSERT INTO summary_node_edges (parent_node_id, child_node_id, position) VALUES (?, ?, ?)",
                (node_id, child_node_id, position),
            )
        self._conn.commit()
        self._conn.execute("INSERT INTO summary_fts(rowid, summary) VALUES (?, ?)", (node_id, summary))
        self._conn.commit()
        return node_id

    def get_summary_node(self, node_id: int) -> dict[str, Any]:
        row = self._conn.execute(
            "SELECT * FROM summary_nodes WHERE node_id = ?",
            (node_id,),
        ).fetchone()
        if row is None:
            raise KeyError(f"summary node not found: {node_id}")
        data = dict(row)
        data["source_event_ids"] = self._source_event_ids(node_id)
        data["source_node_ids"] = self._source_node_ids(node_id)
        return data

    def expand_summary(self, node_id: int) -> dict[str, Any]:
        """Expand a summary node to immediate child nodes and reachable raw events."""
        node = self.get_summary_node(node_id)
        child_nodes = [self.get_summary_node(child_id) for child_id in node["source_node_ids"]]
        event_ids: list[int] = list(node["source_event_ids"])
        for child_id in node["source_node_ids"]:
            event_ids.extend(self._collect_event_ids(child_id))
        seen: set[int] = set()
        unique_event_ids = []
        for event_id in event_ids:
            if event_id not in seen:
                seen.add(event_id)
                unique_event_ids.append(event_id)
        return {
            "node": node,
            "nodes": child_nodes,
            "events": [self.get_event(event_id) for event_id in unique_event_ids],
        }

    def _collect_event_ids(self, node_id: int) -> list[int]:
        event_ids = self._source_event_ids(node_id)
        for child_id in self._source_node_ids(node_id):
            event_ids.extend(self._collect_event_ids(child_id))
        return event_ids

    def _source_event_ids(self, node_id: int) -> list[int]:
        rows = self._conn.execute(
            "SELECT event_id FROM summary_event_edges WHERE node_id = ? ORDER BY position ASC",
            (node_id,),
        ).fetchall()
        return [int(row["event_id"]) for row in rows]

    def _source_node_ids(self, node_id: int) -> list[int]:
        rows = self._conn.execute(
            "SELECT child_node_id FROM summary_node_edges WHERE parent_node_id = ? ORDER BY position ASC",
            (node_id,),
        ).fetchall()
        return [int(row["child_node_id"]) for row in rows]

    def _max_source_sensitivity(self, event_ids: list[int], node_ids: list[int]) -> str:
        rank = {"normal": 0, "private": 1, "restricted": 2, "secret": 3}
        values: list[str] = []
        if event_ids:
            placeholders = ",".join("?" for _ in event_ids)
            rows = self._conn.execute(
                f"SELECT sensitivity FROM events WHERE event_id IN ({placeholders})",
                event_ids,
            ).fetchall()
            values.extend(row["sensitivity"] for row in rows)
        if node_ids:
            placeholders = ",".join("?" for _ in node_ids)
            rows = self._conn.execute(
                f"SELECT sensitivity FROM summary_nodes WHERE node_id IN ({placeholders})",
                node_ids,
            ).fetchall()
            values.extend(row["sensitivity"] for row in rows)
        return max(values or ["normal"], key=lambda value: rank.get(value, 0))

    def _result_from_row(self, row: sqlite3.Row) -> dict[str, Any]:
        data = dict(row)
        data["snippet"] = self._make_snippet(data["content"])
        data.pop("content", None)
        return data

    @staticmethod
    def _fts_query(query: str) -> str:
        tokens = re.findall(r"[\w]+", query)
        return " ".join(tokens) or query

    @staticmethod
    def _make_snippet(content: str, max_chars: int = 240) -> str:
        snippet = content[:max_chars]
        if len(content) > max_chars:
            snippet += "…"
        return _SECRET_ASSIGNMENT_RE.sub(lambda m: f"{m.group(1)}=[REDACTED]", snippet)
