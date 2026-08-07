"""L1 integrity-checked and policy-governed exact transcript retrieval.

This module detects accidental corruption and inconsistent local projections. It
is not a tamper-evident security boundary against an attacker who can rewrite
all SQLite authority tables.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

from .evidence_chain import sha256_text
from .memory_contracts import (
    AccessSurface,
    AuthoritySnapshot,
    LifecycleState,
    MemoryAccessRequest,
    PersonaMode,
    Sensitivity,
)
from .memory_policy import admit_access
from .store import EventStore


@dataclass(frozen=True)
class RecallEvidence:
    evidence_level: Literal["E0", "E1", "E2", "E3", "E4"]
    event_id: int
    chunk_id: int
    quote: str
    start_char: int
    end_char: int
    source_created_at: str | None
    payload_sha256: str
    verified: bool
    source_type: str = "raw_event"
    integrity_scope: str = "l1_local_consistency"


def search_exact_evidence(
    store: EventStore,
    query: str,
    *,
    limit: int = 10,
    persona_mode: PersonaMode = PersonaMode.DEFAULT,
    sensitivity_ceiling: Sensitivity = Sensitivity.RESTRICTED,
) -> list[RecallEvidence]:
    """Return E0 quotes whose authority and access policy are both valid."""
    if type(persona_mode) is not PersonaMode or type(sensitivity_ceiling) is not Sensitivity:
        raise TypeError("exact recall policy inputs are invalid")
    if not query:
        return []
    conn = store._conn
    started_transaction = not conn.in_transaction
    if started_transaction:
        conn.execute("BEGIN")
    try:
        chain_report = store.verify_event_chain()
        if not chain_report["ok"]:
            return []
        rows = conn.execute(
            """
            SELECT event_id, content, payload_sha256, sensitivity, persona_mode,
                   COALESCE(source_created_at, created_at) AS source_created_at,
                   COALESCE((
                       SELECT g.new_state FROM event_governance g
                       WHERE g.event_id = events.event_id
                       ORDER BY g.governance_id DESC LIMIT 1
                   ), evidence_state) AS lifecycle_state,
                   COALESCE((
                       SELECT g.governance_id FROM event_governance g
                       WHERE g.event_id = events.event_id
                       ORDER BY g.governance_id DESC LIMIT 1
                   ), event_id) AS governance_id
            FROM events
            WHERE instr(content, ?) > 0
            ORDER BY event_id DESC
            LIMIT ?
            """,
            (query, max(1, min(int(limit) * 4, 400))),
        ).fetchall()
        results: list[RecallEvidence] = []
        for row in rows:
            content = str(row["content"])
            payload_sha256 = str(row["payload_sha256"])
            if sha256_text(content) != payload_sha256:
                continue
            try:
                lifecycle = LifecycleState(str(row["lifecycle_state"]))
                sensitivity = Sensitivity(str(row["sensitivity"]))
                event_mode = PersonaMode(str(row["persona_mode"] or "default"))
                snapshot = AuthoritySnapshot(
                    authority_kind="event",
                    object_id=str(int(row["event_id"])),
                    object_version=1,
                    payload_sha256=payload_sha256,
                    governance_id=int(row["governance_id"]),
                    governance_state=lifecycle,
                    sensitivity=sensitivity,
                    lifecycle_state=lifecycle,
                    source_refs=(),
                    projection_generation=None,
                    mode_scope=(event_mode,),
                    injection_policy="retrieve_only",
                )
            except (TypeError, ValueError):
                continue
            decision = admit_access(
                snapshot,
                MemoryAccessRequest(AccessSurface.EXACT, persona_mode, sensitivity_ceiling),
            )
            if not decision.allowed:
                continue
            start = content.find(query)
            if start < 0:
                continue
            end = start + len(query)
            overlapping_chunks = conn.execute(
                """
                SELECT chunk_id, chunk_text, chunk_sha256, start_char, end_char
                FROM event_chunks
                WHERE event_id = ? AND start_char < ? AND end_char > ?
                ORDER BY start_char ASC, end_char DESC, chunk_ordinal ASC
                """,
                (int(row["event_id"]), end, start),
            ).fetchall()
            covered_until = start
            first_chunk_id: int | None = None
            chunks_valid = True
            for chunk in overlapping_chunks:
                chunk_text = str(chunk["chunk_text"])
                chunk_start = int(chunk["start_char"])
                chunk_end = int(chunk["end_char"])
                if sha256_text(chunk_text) != chunk["chunk_sha256"]:
                    chunks_valid = False
                    break
                if content[chunk_start:chunk_end] != chunk_text:
                    chunks_valid = False
                    break
                if chunk_start > covered_until:
                    chunks_valid = False
                    break
                if first_chunk_id is None:
                    first_chunk_id = int(chunk["chunk_id"])
                covered_until = max(covered_until, chunk_end)
                if covered_until >= end:
                    break
            if not chunks_valid or first_chunk_id is None or covered_until < end:
                continue
            results.append(RecallEvidence(
                evidence_level="E0",
                event_id=int(row["event_id"]),
                chunk_id=first_chunk_id,
                quote=query,
                start_char=start,
                end_char=end,
                source_created_at=row["source_created_at"],
                payload_sha256=payload_sha256,
                verified=True,
            ))
            if len(results) >= limit:
                break
        return results
    finally:
        if started_transaction and conn.in_transaction:
            conn.rollback()
