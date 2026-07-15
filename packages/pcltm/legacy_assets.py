"""Quarantined import of legacy SoulLink history assets.

The importer copies complete legacy records into an isolated searchable archive.
It never writes to the local production PCLTM database and never exposes archive
rows through the active prompt-memory adapter. Promotion candidates are a
separate, review-only projection.
"""
from __future__ import annotations

import hashlib
import json
import re
import sqlite3
from pathlib import Path
from typing import Any

from .secret_policy import evaluate_memory_write

_ARCHIVE_TABLES = (
    "memory_records",
    "short_term_events",
    "dac_raw_messages",
    "dac_summary_nodes",
    "dac_context_snapshots",
)
_PRIMARY_KEYS = {
    "memory_records": "record_id",
    "short_term_events": "short_event_id",
    "dac_raw_messages": "raw_id",
    "dac_summary_nodes": "node_id",
    "dac_context_snapshots": "snapshot_id",
}
_BODY_COLUMNS = {
    "memory_records": "content",
    "short_term_events": "content",
    "dac_raw_messages": "content",
    "dac_summary_nodes": "summary",
    "dac_context_snapshots": None,
}


def _normalize(text: Any) -> str:
    return re.sub(r"\s+", " ", str(text or "")).strip().casefold()


def _body_hash(text: Any) -> str:
    return hashlib.sha256(_normalize(text).encode("utf-8")).hexdigest()


class LegacyAssetImporter:
    """Import legacy assets into a quarantine DB with review-only candidates."""

    def __init__(
        self,
        source_db: str | Path,
        local_db: str | Path,
        shadow_db: str | Path,
        *,
        source_name: str,
    ) -> None:
        self.source_db = Path(source_db)
        self.local_db = Path(local_db)
        self.shadow_db = Path(shadow_db)
        self.source_name = source_name

    def run(self) -> dict[str, Any]:
        if not self.source_db.is_file():
            raise FileNotFoundError(self.source_db)
        if not self.local_db.is_file():
            raise FileNotFoundError(self.local_db)
        self.shadow_db.parent.mkdir(parents=True, exist_ok=True)
        source = sqlite3.connect(self.source_db.resolve().as_uri() + "?mode=ro", uri=True)
        local = sqlite3.connect(self.local_db.resolve().as_uri() + "?mode=ro", uri=True)
        shadow = sqlite3.connect(self.shadow_db)
        source.row_factory = local.row_factory = shadow.row_factory = sqlite3.Row
        try:
            self._bootstrap(shadow)
            local_hashes = self._local_memory_hashes(local)
            report: dict[str, Any] = {
                "archived": 0,
                "inserted": 0,
                "existing": 0,
                "candidates": 0,
                "candidate_exclusions": {
                    "duplicate_local": 0,
                    "not_approved": 0,
                    "sensitive": 0,
                    "secret_policy": 0,
                },
            }
            seen_candidate_hashes: set[str] = set()
            shadow.execute("BEGIN IMMEDIATE")
            for table in _ARCHIVE_TABLES:
                if not self._table_exists(source, table):
                    continue
                for row in source.execute(f'SELECT * FROM "{table}" ORDER BY "{_PRIMARY_KEYS[table]}"'):
                    payload = dict(row)
                    body_column = _BODY_COLUMNS[table]
                    body = str(payload.get(body_column) or "") if body_column else ""
                    external_id = f"{self.source_name}:{table}:{payload[_PRIMARY_KEYS[table]]}"
                    asset_id, inserted = self._archive(
                        shadow,
                        external_id=external_id,
                        source_table=table,
                        source_row_id=str(payload[_PRIMARY_KEYS[table]]),
                        body=body,
                        payload=payload,
                    )
                    report["archived"] += 1
                    report["inserted" if inserted else "existing"] += 1
                    if table == "memory_records" and inserted:
                        exclusion = self._candidate_exclusion(payload, body, local_hashes, seen_candidate_hashes)
                        if exclusion:
                            report["candidate_exclusions"][exclusion] += 1
                        else:
                            self._add_candidate(shadow, asset_id, payload, body)
                            seen_candidate_hashes.add(_body_hash(body))
                            report["candidates"] += 1
            shadow.commit()
            return report
        except BaseException:
            shadow.rollback()
            raise
        finally:
            shadow.close()
            local.close()
            source.close()

    @staticmethod
    def _bootstrap(conn: sqlite3.Connection) -> None:
        conn.executescript(
            """
            PRAGMA journal_mode=WAL;
            CREATE TABLE IF NOT EXISTS legacy_assets (
                asset_id INTEGER PRIMARY KEY AUTOINCREMENT,
                external_id TEXT NOT NULL UNIQUE,
                source_table TEXT NOT NULL,
                source_row_id TEXT NOT NULL,
                body TEXT NOT NULL,
                body_sha256 TEXT NOT NULL,
                payload_json TEXT NOT NULL,
                archived_at TEXT NOT NULL DEFAULT (strftime('%Y-%m-%dT%H:%M:%fZ','now'))
            );
            CREATE VIRTUAL TABLE IF NOT EXISTS legacy_asset_fts
            USING fts5(body, content='legacy_assets', content_rowid='asset_id');
            CREATE TABLE IF NOT EXISTS legacy_promotion_candidates (
                candidate_id INTEGER PRIMARY KEY AUTOINCREMENT,
                asset_id INTEGER NOT NULL UNIQUE,
                kind TEXT NOT NULL,
                target_file TEXT NOT NULL,
                body TEXT NOT NULL,
                body_sha256 TEXT NOT NULL,
                confidence REAL NOT NULL,
                status TEXT NOT NULL DEFAULT 'pending_review',
                decision_reason TEXT,
                created_at TEXT NOT NULL DEFAULT (strftime('%Y-%m-%dT%H:%M:%fZ','now')),
                FOREIGN KEY(asset_id) REFERENCES legacy_assets(asset_id)
            );
            """
        )
        conn.commit()

    @staticmethod
    def _table_exists(conn: sqlite3.Connection, table: str) -> bool:
        return conn.execute(
            "SELECT 1 FROM sqlite_master WHERE type='table' AND name=?", (table,)
        ).fetchone() is not None

    @staticmethod
    def _local_memory_hashes(conn: sqlite3.Connection) -> set[str]:
        if not LegacyAssetImporter._table_exists(conn, "memory_records"):
            return set()
        return {_body_hash(row[0]) for row in conn.execute("SELECT content FROM memory_records") if _normalize(row[0])}

    @staticmethod
    def _archive(
        conn: sqlite3.Connection,
        *,
        external_id: str,
        source_table: str,
        source_row_id: str,
        body: str,
        payload: dict[str, Any],
    ) -> tuple[int, bool]:
        existing = conn.execute(
            "SELECT asset_id FROM legacy_assets WHERE external_id=?", (external_id,)
        ).fetchone()
        if existing:
            return int(existing[0]), False
        cur = conn.execute(
            """
            INSERT INTO legacy_assets
            (external_id, source_table, source_row_id, body, body_sha256, payload_json)
            VALUES (?, ?, ?, ?, ?, ?)
            """,
            (
                external_id,
                source_table,
                source_row_id,
                body,
                hashlib.sha256(body.encode("utf-8")).hexdigest(),
                json.dumps(payload, ensure_ascii=False, sort_keys=True),
            ),
        )
        asset_id = int(cur.lastrowid)
        conn.execute("INSERT INTO legacy_asset_fts(rowid, body) VALUES (?, ?)", (asset_id, body))
        return asset_id, True

    @staticmethod
    def _candidate_exclusion(
        payload: dict[str, Any],
        body: str,
        local_hashes: set[str],
        seen_hashes: set[str],
    ) -> str | None:
        if str(payload.get("status") or "") != "approved":
            return "not_approved"
        if str(payload.get("sensitivity") or "normal") != "normal":
            return "sensitive"
        decision = evaluate_memory_write(body, target_file=str(payload.get("target_file") or "MEMORY.md"))
        if not decision.allowed or decision.action != "allow":
            return "secret_policy"
        digest = _body_hash(body)
        if digest in local_hashes or digest in seen_hashes:
            return "duplicate_local"
        return None

    @staticmethod
    def _add_candidate(
        conn: sqlite3.Connection,
        asset_id: int,
        payload: dict[str, Any],
        body: str,
    ) -> None:
        conn.execute(
            """
            INSERT OR IGNORE INTO legacy_promotion_candidates
            (asset_id, kind, target_file, body, body_sha256, confidence, status)
            VALUES (?, ?, ?, ?, ?, ?, 'pending_review')
            """,
            (
                asset_id,
                str(payload.get("kind") or "memory_note"),
                str(payload.get("target_file") or "MEMORY.md"),
                body,
                _body_hash(body),
                float(payload.get("confidence") or 0.0),
            ),
        )


__all__ = ["LegacyAssetImporter"]
