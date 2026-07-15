"""SQLite-backed semantic memory store for PCLTM."""

from __future__ import annotations

import json
import sqlite3
from pathlib import Path
from typing import Iterable

from .temporal_fact import TemporalFact, datetime_to_text


class SemanticStore:
    """Durable CRUD/search store for governed temporal facts."""

    def __init__(self, path: str | Path) -> None:
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._init_db()

    def _connect(self) -> sqlite3.Connection:
        conn = sqlite3.connect(self.path)
        conn.row_factory = sqlite3.Row
        return conn

    def _init_db(self) -> None:
        with self._connect() as conn:
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS semantic_memory (
                    memory_id TEXT PRIMARY KEY,
                    subject TEXT NOT NULL,
                    predicate TEXT NOT NULL,
                    object TEXT NOT NULL,
                    confidence REAL NOT NULL,
                    valid_from TEXT NOT NULL,
                    valid_until TEXT,
                    source_refs TEXT NOT NULL,
                    stability TEXT NOT NULL,
                    namespace TEXT NOT NULL,
                    conflict_group TEXT,
                    supersedes TEXT NOT NULL,
                    superseded_by TEXT,
                    write_reason TEXT NOT NULL,
                    last_verified_at TEXT
                )
                """
            )
            conn.execute(
                """
                CREATE INDEX IF NOT EXISTS idx_semantic_fact_key
                ON semantic_memory(namespace, subject, predicate)
                """
            )
            conn.execute(
                """
                CREATE INDEX IF NOT EXISTS idx_semantic_conflict_group
                ON semantic_memory(conflict_group)
                """
            )
            conn.execute(
                """
                CREATE INDEX IF NOT EXISTS idx_semantic_active
                ON semantic_memory(namespace, valid_until, superseded_by)
                """
            )

    def add(self, fact: TemporalFact) -> TemporalFact:
        """Insert or replace one semantic fact."""
        with self._connect() as conn:
            conn.execute(
                """
                INSERT OR REPLACE INTO semantic_memory (
                    memory_id, subject, predicate, object, confidence,
                    valid_from, valid_until, source_refs, stability, namespace,
                    conflict_group, supersedes, superseded_by, write_reason,
                    last_verified_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                self._to_row(fact),
            )
        return fact

    def update(self, fact: TemporalFact) -> TemporalFact:
        if self.get(fact.memory_id) is None:
            raise KeyError(f"semantic memory not found: {fact.memory_id}")
        return self.add(fact)

    def delete(self, memory_id: str) -> bool:
        with self._connect() as conn:
            cur = conn.execute("DELETE FROM semantic_memory WHERE memory_id = ?", (memory_id,))
            return cur.rowcount > 0

    def get(self, memory_id: str) -> TemporalFact | None:
        with self._connect() as conn:
            row = conn.execute(
                "SELECT * FROM semantic_memory WHERE memory_id = ?", (memory_id,)
            ).fetchone()
        return self._from_row(row) if row is not None else None

    def search(
        self,
        *,
        subject: str | None = None,
        predicate: str | None = None,
        object_contains: str | None = None,
        namespace: str | None = None,
        active_only: bool = True,
        conflict_group: str | None = None,
        limit: int = 50,
    ) -> list[TemporalFact]:
        clauses: list[str] = []
        args: list[object] = []
        if subject is not None:
            clauses.append("subject = ?")
            args.append(subject)
        if predicate is not None:
            clauses.append("predicate = ?")
            args.append(predicate)
        if object_contains is not None:
            clauses.append("LOWER(object) LIKE ?")
            args.append(f"%{object_contains.lower()}%")
        if namespace is not None:
            clauses.append("namespace = ?")
            args.append(namespace)
        if conflict_group is not None:
            clauses.append("conflict_group = ?")
            args.append(conflict_group)
        if active_only:
            clauses.append("valid_until IS NULL")
            clauses.append("superseded_by IS NULL")
        where = " WHERE " + " AND ".join(clauses) if clauses else ""
        args.append(limit)
        with self._connect() as conn:
            rows = conn.execute(
                f"""
                SELECT * FROM semantic_memory
                {where}
                ORDER BY confidence DESC, valid_from DESC
                LIMIT ?
                """,
                tuple(args),
            ).fetchall()
        return [self._from_row(row) for row in rows]

    def list_conflicts(self, conflict_group: str) -> list[TemporalFact]:
        return self.search(active_only=False, conflict_group=conflict_group, limit=200)

    def mark_superseded(
        self,
        *,
        old_memory_ids: Iterable[str],
        superseded_by: str,
        valid_until,
    ) -> list[TemporalFact]:
        updated: list[TemporalFact] = []
        for memory_id in old_memory_ids:
            current = self.get(memory_id)
            if current is None:
                continue
            changed = current.with_supersession(
                superseded_by=superseded_by,
                valid_until=valid_until,
            )
            self.update(changed)
            updated.append(changed)
        return updated

    @staticmethod
    def _to_row(fact: TemporalFact) -> tuple[object, ...]:
        return (
            fact.memory_id,
            fact.subject,
            fact.predicate,
            fact.object,
            fact.confidence,
            datetime_to_text(fact.valid_from),
            datetime_to_text(fact.valid_until),
            json.dumps(list(fact.source_refs), ensure_ascii=False),
            fact.stability,
            fact.namespace,
            fact.conflict_group,
            json.dumps(list(fact.supersedes), ensure_ascii=False),
            fact.superseded_by,
            fact.write_reason,
            datetime_to_text(fact.last_verified_at),
        )

    @staticmethod
    def _from_row(row: sqlite3.Row) -> TemporalFact:
        return TemporalFact.from_dict(
            {
                "memory_id": row["memory_id"],
                "subject": row["subject"],
                "predicate": row["predicate"],
                "object": row["object"],
                "confidence": row["confidence"],
                "valid_from": row["valid_from"],
                "valid_until": row["valid_until"],
                "source_refs": json.loads(row["source_refs"] or "[]"),
                "stability": row["stability"],
                "namespace": row["namespace"],
                "conflict_group": row["conflict_group"],
                "supersedes": json.loads(row["supersedes"] or "[]"),
                "superseded_by": row["superseded_by"],
                "write_reason": row["write_reason"],
                "last_verified_at": row["last_verified_at"],
            }
        )
