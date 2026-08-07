"""Authority-bound lexical retrieval for governed memory claims.

FTS is only a bounded candidate generator. Every returned item is reopened from
``memory_current`` and re-admitted by the deterministic memory policy.
"""

from __future__ import annotations

import json
import re
import sqlite3
from contextlib import contextmanager
from dataclasses import dataclass
from enum import Enum

from .evidence_chain import sha256_text
from .memory_contracts import (
    AccessSurface,
    AuthorityRef,
    AuthoritySnapshot,
    LifecycleState,
    MemoryAccessRequest,
    PersonaMode,
    Sensitivity,
)
from .memory_policy import admit_access
from .store import EventStore

MAX_SEARCH_LIMIT = 100


class MemoryRetrievalStatus(str, Enum):
    OK = "ok"
    ABSTAINED = "abstained"
    UNAVAILABLE = "unavailable"


@dataclass(frozen=True, slots=True)
class GovernedMemorySearchRequest:
    query: str
    persona_mode: PersonaMode
    limit: int = 8
    sensitivity_ceiling: Sensitivity = Sensitivity.RESTRICTED

    def __post_init__(self) -> None:
        if type(self.query) is not str:
            raise TypeError("query must be str")
        if type(self.persona_mode) is not PersonaMode:
            raise TypeError("persona_mode must be PersonaMode")
        if (
            type(self.limit) is not int
            or isinstance(self.limit, bool)
            or self.limit <= 0
            or self.limit > MAX_SEARCH_LIMIT
        ):
            raise ValueError(f"limit must be an int from 1 to {MAX_SEARCH_LIMIT}")
        if type(self.sensitivity_ceiling) is not Sensitivity:
            raise TypeError("sensitivity_ceiling must be Sensitivity")


@dataclass(frozen=True, slots=True)
class GovernedMemoryOpenRequest:
    claim_id: int
    persona_mode: PersonaMode
    sensitivity_ceiling: Sensitivity = Sensitivity.RESTRICTED

    def __post_init__(self) -> None:
        if type(self.claim_id) is not int or isinstance(self.claim_id, bool) or self.claim_id <= 0:
            raise ValueError("claim_id must be a positive int")
        if type(self.persona_mode) is not PersonaMode:
            raise TypeError("persona_mode must be PersonaMode")
        if type(self.sensitivity_ceiling) is not Sensitivity:
            raise TypeError("sensitivity_ceiling must be Sensitivity")


@dataclass(frozen=True, slots=True)
class GovernedMemoryItem:
    claim_id: int
    claim_version: int
    governance_id: int
    content: str
    content_sha256: str
    canonical_key: str
    target: str
    memory_type: str
    sensitivity: Sensitivity
    mode_scope: tuple[PersonaMode, ...]
    injection_policy: str
    source_refs: tuple[AuthorityRef, ...]
    policy_reason: str
    policy_version: str
    authority_verified: bool = True
    rank: int | None = None
    rank_score: float | None = None
    rank_score_is_authority: bool = False


@dataclass(frozen=True, slots=True)
class GovernedMemoryRetrievalResult:
    status: MemoryRetrievalStatus
    items: tuple[GovernedMemoryItem, ...] = ()
    reason: str | None = None

    def __post_init__(self) -> None:
        if self.status is MemoryRetrievalStatus.OK:
            if not self.items or self.reason is not None:
                raise ValueError("ok requires items and no reason")
        elif self.items or type(self.reason) is not str or not self.reason:
            raise ValueError("non-ok requires no items and a reason")

    @classmethod
    def ok(cls, items: list[GovernedMemoryItem]) -> GovernedMemoryRetrievalResult:
        return cls(MemoryRetrievalStatus.OK, tuple(items))

    @classmethod
    def abstained(cls, reason: str) -> GovernedMemoryRetrievalResult:
        return cls(MemoryRetrievalStatus.ABSTAINED, (), reason)

    @classmethod
    def unavailable(cls, reason: str) -> GovernedMemoryRetrievalResult:
        return cls(MemoryRetrievalStatus.UNAVAILABLE, (), reason)


def _authority_row(store: EventStore, claim_id: int):
    return store._conn.execute(
        """
        SELECT c.claim_id, c.canonical_key, c.target, c.memory_type,
               v.claim_version_id, v.version, v.content, v.content_sha256,
               v.sensitivity, v.injection_policy, v.mode_scope,
               mc.lifecycle_state, g.memory_governance_id, g.action,
               g.previous_state, g.new_state,
               g.policy_version
        FROM memory_current mc
        JOIN memory_claims c ON c.claim_id = mc.claim_id
        JOIN memory_claim_versions v
          ON v.claim_version_id = mc.claim_version_id AND v.claim_id = mc.claim_id
        JOIN memory_governance_events g
          ON g.memory_governance_id = mc.memory_governance_id
         AND g.claim_id = mc.claim_id
         AND g.claim_version_id = mc.claim_version_id
         AND g.new_state = mc.lifecycle_state
        WHERE mc.claim_id = ?
        """,
        (claim_id,),
    ).fetchone()


@contextmanager
def _read_snapshot(store: EventStore):
    conn = store._conn
    owns_transaction = not conn.in_transaction
    if owns_transaction:
        conn.execute("BEGIN")
    try:
        yield
        if owns_transaction:
            conn.commit()
    except BaseException:
        if owns_transaction:
            conn.rollback()
        raise


def _source_refs(store: EventStore, version_id: int) -> tuple[AuthorityRef, ...] | None:
    rows = store._conn.execute(
        """
        SELECT source_kind, event_id, event_revision, event_payload_sha256,
               legacy_record_id, legacy_content_sha256
        FROM memory_claim_sources WHERE claim_version_id = ?
        ORDER BY claim_source_id
        """,
        (version_id,),
    ).fetchall()
    refs: list[AuthorityRef] = []
    for row in rows:
        if row["source_kind"] == "event":
            event = store._conn.execute(
                "SELECT source_revision, payload_sha256 FROM events WHERE event_id = ?",
                (row["event_id"],),
            ).fetchone()
            try:
                event_id = int(row["event_id"])
                event_revision = int(row["event_revision"])
                if event is None or int(event["source_revision"]) != event_revision:
                    return None
                payload_sha256 = str(row["event_payload_sha256"])
                if str(event["payload_sha256"]) != payload_sha256:
                    return None
                refs.append(AuthorityRef("event", str(event_id), event_revision, payload_sha256))
            except (TypeError, ValueError):
                return None
        elif row["source_kind"] == "legacy_record":
            legacy = store._conn.execute(
                "SELECT content, status FROM memory_records WHERE record_id = ?",
                (row["legacy_record_id"],),
            ).fetchone()
            try:
                record_id = int(row["legacy_record_id"])
                payload_sha256 = str(row["legacy_content_sha256"])
                if (
                    legacy is None
                    or str(legacy["status"]) != "approved"
                    or sha256_text(str(legacy["content"])) != payload_sha256
                ):
                    return None
                refs.append(AuthorityRef(
                    "legacy_record", str(record_id), 1, payload_sha256,
                ))
            except (TypeError, ValueError):
                return None
        else:
            return None
    return tuple(refs) if refs else None


def _projection_matches_authority(projection, authority, sources: tuple[AuthorityRef, ...]) -> bool:
    content_hash = sha256_text(str(authority["content"]))
    try:
        claim_version = int(projection["claim_version"])
        governance_id = int(projection["governance_id"])
        projection_generation = int(projection["projection_generation"])
    except (TypeError, ValueError):
        return False
    if (
        content_hash != str(authority["content_sha256"])
        or claim_version != int(authority["version"])
        or governance_id != int(authority["memory_governance_id"])
        or projection_generation <= 0
        or str(projection["policy_version"]) != str(authority["policy_version"])
        or str(projection["lifecycle_state"]) != str(authority["lifecycle_state"])
        or str(projection["payload_sha256"]) != content_hash
        or sha256_text(str(projection["content"])) != content_hash
    ):
        return False
    try:
        projected_sources = json.loads(str(projection["source_refs"]))
    except (TypeError, json.JSONDecodeError):
        return False
    authoritative_sources = [
        {
            "authority_kind": ref.authority_kind,
            "object_id": ref.object_id,
            "object_version": ref.object_version,
            "payload_sha256": ref.payload_sha256,
        }
        for ref in sources
    ]
    return projected_sources == authoritative_sources


def _reopen(
    store: EventStore,
    claim_id: int,
    *,
    surface: AccessSurface,
    persona_mode: PersonaMode,
    sensitivity_ceiling: Sensitivity,
    projection=None,
    rank: int | None = None,
    rank_score: float | None = None,
) -> GovernedMemoryItem | None:
    row = _authority_row(store, claim_id)
    if row is None:
        return None
    transition = (
        str(row["action"]),
        None if row["previous_state"] is None else str(row["previous_state"]),
        str(row["new_state"]),
    )
    if transition not in {
        ("submit", None, "pending_review"),
        ("activate", "pending_review", "active"),
        ("supersede", "active", "superseded"),
        ("retire", "active", "retired"),
        ("expire", "active", "expired"),
        ("reject", "pending_review", "rejected"),
        ("quarantine", "pending_review", "quarantined"),
    }:
        return None
    content = str(row["content"])
    content_hash = sha256_text(content)
    if content_hash != str(row["content_sha256"]):
        return None
    try:
        lifecycle = LifecycleState(str(row["lifecycle_state"]))
        governance_state = LifecycleState(str(row["new_state"]))
        sensitivity = Sensitivity(str(row["sensitivity"]))
        raw_modes = json.loads(str(row["mode_scope"]))
        mode_scope = tuple(PersonaMode(value) for value in raw_modes)
    except (ValueError, TypeError, json.JSONDecodeError):
        return None
    sources = _source_refs(store, int(row["claim_version_id"]))
    if sources is None:
        return None

    if projection is not None and not _projection_matches_authority(projection, row, sources):
        return None

    try:
        snapshot = AuthoritySnapshot(
            authority_kind="memory_claim",
            object_id=str(claim_id),
            object_version=int(row["version"]),
            payload_sha256=content_hash,
            governance_id=int(row["memory_governance_id"]),
            governance_state=governance_state,
            sensitivity=sensitivity,
            lifecycle_state=lifecycle,
            source_refs=sources,
            projection_generation=(int(projection["projection_generation"]) if projection is not None else None),
            mode_scope=mode_scope,
            injection_policy=str(row["injection_policy"]),
        )
    except (TypeError, ValueError):
        return None
    decision = admit_access(
        snapshot,
        MemoryAccessRequest(surface, persona_mode, sensitivity_ceiling),
    )
    if not decision.allowed:
        return None
    return GovernedMemoryItem(
        claim_id=claim_id,
        claim_version=int(row["version"]),
        governance_id=int(row["memory_governance_id"]),
        content=content,
        content_sha256=content_hash,
        canonical_key=str(row["canonical_key"]),
        target=str(row["target"]),
        memory_type=str(row["memory_type"]),
        sensitivity=sensitivity,
        mode_scope=mode_scope,
        injection_policy=str(row["injection_policy"]),
        source_refs=sources,
        policy_reason=decision.reason_code,
        policy_version=str(row["policy_version"]),
        rank=rank,
        rank_score=rank_score,
    )


def _search_in_snapshot(
    store: EventStore,
    request: GovernedMemorySearchRequest,
) -> GovernedMemoryRetrievalResult:
    if type(request) is not GovernedMemorySearchRequest:
        raise TypeError("request must be GovernedMemorySearchRequest")
    query = request.query.strip()
    if not query:
        return GovernedMemoryRetrievalResult.abstained("no_answer")
    exists = store._conn.execute(
        "SELECT 1 FROM sqlite_master WHERE type = 'table' AND name = 'memory_fts'"
    ).fetchone()
    if exists is None:
        current = store._conn.execute("SELECT 1 FROM memory_current LIMIT 1").fetchone()
        if current is None:
            return GovernedMemoryRetrievalResult.abstained("no_answer")
        return GovernedMemoryRetrievalResult.unavailable("projection_unavailable")
    # FTS5 trigram tokenizer cannot match 1-2 char terms, and a whole-string
    # phrase match requires contiguous occurrence — natural-language queries
    # ("状态机 语义 影子运行") would miss memories that contain the terms
    # separately. Split into terms, keep the FTS-mappable ones (>=3 chars),
    # and AND them together; fall back to the whole-string phrase when no
    # term survives (the len(query) < 3 LIKE branch above covers shortest
    # queries).
    query_terms = [t for t in re.split(r"[\s,，。;；:：、]+", query) if t]
    fts_terms = [t for t in query_terms if len(t) >= 3]
    short_terms = [t for t in query_terms if len(t) < 3]
    if fts_terms:
        literal_query = " AND ".join(
            '"' + t.replace('"', '""') + '"' for t in fts_terms
        )
    else:
        literal_query = '"' + query.replace('"', '""') + '"'
    try:
        if len(query) < 3:
            # trigram tokenizer cannot match 1-2 char queries (e.g. "心血",
            # "老婆"). Fall back to an escaped substring scan on the tiny
            # projection table; results still pass the same authority checks.
            like_pattern = "%" + (
                query.replace("\\", "\\\\").replace("%", "\\%").replace("_", "\\_")
            ) + "%"
            rows = store._conn.execute(
                """
                SELECT rowid AS claim_id, content, lifecycle_state, claim_version,
                       governance_id, payload_sha256, projection_generation, policy_version,
                       source_refs, 0.0 AS rank_score
                FROM memory_fts
                WHERE content LIKE ? ESCAPE '\\'
                ORDER BY rowid
                LIMIT ?
                """,
                (like_pattern, request.limit),
            ).fetchall()
        elif short_terms:
            literal_conditions = []
            params: list[object] = []
            for term in short_terms:
                literal_conditions.append("instr(lower(content), lower(?)) > 0")
                params.append(term)
            params.extend((literal_query, request.limit))
            rows = store._conn.execute(
                f"""
                SELECT rowid AS claim_id, content, lifecycle_state, claim_version,
                       governance_id, payload_sha256, projection_generation, policy_version,
                       source_refs,
                       bm25(memory_fts) AS rank_score
                FROM memory_fts
                WHERE {' AND '.join(literal_conditions)}
                  AND memory_fts MATCH ?
                ORDER BY rank_score, rowid
                LIMIT ?
                """,
                tuple(params),
            ).fetchall()
        else:
            rows = store._conn.execute(
                """
                SELECT rowid AS claim_id, content, lifecycle_state, claim_version,
                       governance_id, payload_sha256, projection_generation, policy_version,
                       source_refs, bm25(memory_fts) AS rank_score
                FROM memory_fts
                WHERE memory_fts MATCH ?
                ORDER BY rank_score, rowid
                LIMIT ?
                """,
                (literal_query, request.limit),
            ).fetchall()
    except sqlite3.Error:
        return GovernedMemoryRetrievalResult.unavailable("projection_query_failed")
    if not rows:
        # Optional local neural retrieval is a candidate generator only. IDs
        # still come back through memory_fts commitments and the same authority
        # reopen/policy path below; model scores never become memory authority.
        try:
            from .semantic_claim_retrieval import semantic_claim_candidates

            semantic = semantic_claim_candidates(
                store, query, limit=max(request.limit * 4, request.limit),
            )
        except sqlite3.Error:
            return GovernedMemoryRetrievalResult.unavailable("authority_store_unavailable")
        except (ImportError, OSError):
            semantic = []
        if semantic:
            score_by_id = dict(semantic)
            placeholders = ",".join("?" for _ in semantic)
            try:
                projected = store._conn.execute(
                    f"""
                    SELECT rowid AS claim_id, content, lifecycle_state, claim_version,
                           governance_id, payload_sha256, projection_generation,
                           policy_version, source_refs
                    FROM memory_fts WHERE rowid IN ({placeholders})
                    """,
                    tuple(claim_id for claim_id, _score in semantic),
                ).fetchall()
            except sqlite3.Error:
                return GovernedMemoryRetrievalResult.unavailable("authority_store_unavailable")
            rows = sorted(
                (
                    {**dict(row), "rank_score": score_by_id[int(row["claim_id"])]}
                    for row in projected
                ),
                key=lambda row: (-float(row["rank_score"]), int(row["claim_id"])),
            )[:request.limit]
        if not rows:
            return GovernedMemoryRetrievalResult.abstained("no_answer")
    items: list[GovernedMemoryItem] = []
    reopen_failed = False
    for rank, row in enumerate(rows, start=1):
        item = _reopen(
            store,
            int(row["claim_id"]),
            surface=AccessSurface.SEARCH,
            persona_mode=request.persona_mode,
            sensitivity_ceiling=request.sensitivity_ceiling,
            projection=row,
            rank=rank,
            rank_score=float(row["rank_score"]),
        )
        if item is None:
            authority = _authority_row(store, int(row["claim_id"]))
            sources = (
                None if authority is None
                else _source_refs(store, int(authority["claim_version_id"]))
            )
            if (
                authority is None
                or sources is None
                or not _projection_matches_authority(row, authority, sources)
            ):
                reopen_failed = True
            continue
        items.append(item)
    if items:
        return GovernedMemoryRetrievalResult.unavailable("authority_reopen_failed") if reopen_failed else GovernedMemoryRetrievalResult.ok(items)
    return GovernedMemoryRetrievalResult.abstained(
        "authority_reopen_failed" if reopen_failed else "policy_filtered"
    )


def search_governed_memories(
    store: EventStore,
    request: GovernedMemorySearchRequest,
) -> GovernedMemoryRetrievalResult:
    try:
        with _read_snapshot(store):
            return _search_in_snapshot(store, request)
    except sqlite3.Error:
        return GovernedMemoryRetrievalResult.unavailable("authority_store_unavailable")


def _open_in_snapshot(
    store: EventStore,
    request: GovernedMemoryOpenRequest,
) -> GovernedMemoryRetrievalResult:
    if type(request) is not GovernedMemoryOpenRequest:
        raise TypeError("request must be GovernedMemoryOpenRequest")
    if _authority_row(store, request.claim_id) is None:
        return GovernedMemoryRetrievalResult.abstained("authority_not_found")
    item = _reopen(
        store,
        request.claim_id,
        surface=AccessSurface.OPEN,
        persona_mode=request.persona_mode,
        sensitivity_ceiling=request.sensitivity_ceiling,
    )
    if item is None:
        return GovernedMemoryRetrievalResult.abstained("policy_filtered")
    return GovernedMemoryRetrievalResult.ok([item])


def open_governed_memory(
    store: EventStore,
    request: GovernedMemoryOpenRequest,
) -> GovernedMemoryRetrievalResult:
    try:
        with _read_snapshot(store):
            return _open_in_snapshot(store, request)
    except sqlite3.Error:
        return GovernedMemoryRetrievalResult.unavailable("authority_store_unavailable")
