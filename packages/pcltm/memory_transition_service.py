"""Versioned, CAS-bound lifecycle transitions for governed memory claims."""

from __future__ import annotations

import json
from dataclasses import dataclass

from . import memory_policy as policy
from .classifier import parse_memory_command
from .evidence_chain import sha256_text
from .memory_contracts import (
    AuthorityRef,
    AuthoritySnapshot,
    LifecycleState,
    LineageKind,
    MemoryTransitionCommand,
    MemoryWriteCommand,
    MemoryWriteReceipt,
    PersonaMode,
    Sensitivity,
)
from .projection_outbox import enqueue_memory_projections
from .secret_policy import evaluate_memory_write
from .store import EventStore


@dataclass(frozen=True, slots=True)
class MemoryReplaceRequest:
    idempotency_key: str
    claim_id: int
    expected_current_version: int
    content: str
    sensitivity: Sensitivity
    mode_scope: tuple[PersonaMode, ...]
    injection_policy: str
    session_id: str = "memory-transition-service"
    conversation_id: str = "memory-replace"
    platform: str = "internal"
    lineage_kind: LineageKind = LineageKind.EXPLICIT_USER_ASSERTION
    source_refs: tuple[AuthorityRef, ...] = ()

    def __post_init__(self) -> None:
        for name in ("idempotency_key", "content", "injection_policy", "session_id", "conversation_id", "platform"):
            value = getattr(self, name)
            if type(value) is not str or not value.strip():
                raise ValueError(f"{name} is required")
        if type(self.claim_id) is not int or self.claim_id <= 0:
            raise TypeError("claim_id must be a positive int")
        if type(self.expected_current_version) is not int or self.expected_current_version <= 0:
            raise TypeError("expected_current_version must be a positive int")
        if type(self.sensitivity) is not Sensitivity:
            raise TypeError("sensitivity must be Sensitivity")
        if type(self.mode_scope) is not tuple or not self.mode_scope or not all(
            type(mode) is PersonaMode for mode in self.mode_scope
        ):
            raise TypeError("mode_scope must be a non-empty tuple of PersonaMode")
        if type(self.lineage_kind) is not LineageKind:
            raise TypeError("lineage_kind must be LineageKind")
        if type(self.source_refs) is not tuple or not all(type(ref) is AuthorityRef for ref in self.source_refs):
            raise TypeError("source_refs must be a tuple of AuthorityRef")


@dataclass(frozen=True, slots=True)
class MemoryLifecycleRequest:
    idempotency_key: str
    claim_id: int
    expected_current_version: int
    reason_code: str

    def __post_init__(self) -> None:
        for name in ("idempotency_key", "reason_code"):
            value = getattr(self, name)
            if type(value) is not str or not value.strip():
                raise ValueError(f"{name} is required")
        if type(self.claim_id) is not int or self.claim_id <= 0:
            raise TypeError("claim_id must be a positive int")
        if type(self.expected_current_version) is not int or self.expected_current_version <= 0:
            raise TypeError("expected_current_version must be a positive int")


class MemoryTransitionService:
    def __init__(self, store: EventStore) -> None:
        self._store = store

    @staticmethod
    def _failure(status: str, reason: str, *, claim_id: int | None = None) -> MemoryWriteReceipt:
        return MemoryWriteReceipt(False, status, claim_id, None, None, False, "none", False, reason)

    @staticmethod
    def _source_hash(request: MemoryReplaceRequest) -> str:
        payload = {
            "action": "replace",
            "claim_id": request.claim_id,
            "content": request.content,
            "expected_current_version": request.expected_current_version,
            "injection_policy": request.injection_policy,
            "mode_scope": [mode.value for mode in request.mode_scope],
            "sensitivity": request.sensitivity.value,
            "session_id": request.session_id,
            "conversation_id": request.conversation_id,
            "platform": request.platform,
            "lineage_kind": request.lineage_kind.value,
            "source_refs": [
                (ref.authority_kind, ref.object_id, ref.object_version, ref.payload_sha256)
                for ref in request.source_refs
            ],
        }
        return sha256_text(json.dumps(payload, ensure_ascii=False, sort_keys=True, separators=(",", ":")))

    def _lifecycle(self, request: MemoryLifecycleRequest, *, action: str, target: LifecycleState) -> MemoryWriteReceipt:
        if type(request) is not MemoryLifecycleRequest:
            raise TypeError("request must be MemoryLifecycleRequest")
        conn = self._store._conn
        source_hash = sha256_text(json.dumps({
            "action": action,
            "claim_id": request.claim_id,
            "expected_current_version": request.expected_current_version,
            "reason_code": request.reason_code,
        }, sort_keys=True, separators=(",", ":")))
        try:
            conn.execute("BEGIN IMMEDIATE")
            existing = conn.execute(
                """
                SELECT request_sha256, action, claim_id, claim_version,
                       memory_governance_id
                FROM memory_transition_receipts WHERE idempotency_key = ?
                """,
                (request.idempotency_key,),
            ).fetchone()
            if existing is not None:
                if (
                    str(existing["request_sha256"]) != source_hash
                    or str(existing["action"]) != action
                    or int(existing["claim_id"]) != request.claim_id
                ):
                    conn.rollback()
                    return self._failure("conflict", "idempotency_conflict", claim_id=request.claim_id)
                row = conn.execute(
                    """
                    SELECT g.new_state FROM memory_governance_events g
                    WHERE g.memory_governance_id = ? AND g.claim_id = ?
                      AND g.action = ?
                    """,
                    (int(existing["memory_governance_id"]), request.claim_id, action),
                ).fetchone()
                conn.commit()
                if row is None:
                    return self._failure("conflict", "idempotency_conflict", claim_id=request.claim_id)
                return MemoryWriteReceipt(
                    True, str(row["new_state"]), request.claim_id,
                    int(existing["claim_version"]), int(existing["memory_governance_id"]),
                    True, "pending", False, policy.REASON_TRANSITION_ALLOWED,
                )

            current = conn.execute(
                """
                SELECT mc.claim_version_id, v.version, v.content_sha256, mc.lifecycle_state,
                       s.source_kind, s.event_id, s.legacy_record_id
                FROM memory_current mc
                JOIN memory_claim_versions v ON v.claim_version_id = mc.claim_version_id
                JOIN memory_claim_sources s ON s.claim_version_id = mc.claim_version_id
                WHERE mc.claim_id = ?
                """,
                (request.claim_id,),
            ).fetchall()
            if not current:
                conn.rollback()
                return self._failure("not_found", "claim_not_found", claim_id=request.claim_id)
            authority = current[0]
            if int(authority["version"]) != request.expected_current_version:
                conn.rollback()
                return self._failure("conflict", "stale_expected_version", claim_id=request.claim_id)
            try:
                current_state = LifecycleState(str(authority["lifecycle_state"]))
            except ValueError:
                conn.rollback()
                return self._failure("conflict", policy.REASON_INVALID_TRANSITION, claim_id=request.claim_id)
            decision = policy.resolve_transition(current_state, MemoryTransitionCommand(action, target))
            if not decision.allowed:
                conn.rollback()
                return self._failure("conflict", decision.reason_code, claim_id=request.claim_id)

            governance = conn.execute(
                """
                INSERT INTO memory_governance_events(
                    claim_id, claim_version_id, action, previous_state, new_state,
                    actor, reason_code, policy_version
                ) VALUES (?, ?, ?, 'active', ?, 'memory_transition_service', ?, ?)
                """,
                (
                    request.claim_id, int(authority["claim_version_id"]), action,
                    target.value, request.reason_code, decision.policy_version,
                ),
            )
            governance_id = int(governance.lastrowid)
            switched = conn.execute(
                """
                UPDATE memory_current
                SET memory_governance_id = ?, lifecycle_state = ?,
                    updated_at = strftime('%Y-%m-%dT%H:%M:%fZ', 'now')
                WHERE claim_id = ? AND claim_version_id = ? AND lifecycle_state = 'active'
                """,
                (
                    governance_id, target.value, request.claim_id,
                    int(authority["claim_version_id"]),
                ),
            )
            if switched.rowcount != 1:
                conn.rollback()
                return self._failure("conflict", "stale_expected_version", claim_id=request.claim_id)
            conn.execute(
                """
                INSERT INTO memory_transition_receipts(
                    idempotency_key, request_sha256, claim_id, claim_version,
                    action, memory_governance_id
                ) VALUES (?, ?, ?, ?, ?, ?)
                """,
                (
                    request.idempotency_key, source_hash, request.claim_id,
                    request.expected_current_version, action, governance_id,
                ),
            )
            source_kind = str(authority["source_kind"])
            if source_kind == "event":
                event_id = int(authority["event_id"])
                authority_kind, authority_id = "event", str(event_id)
            elif source_kind == "legacy_record":
                event_id = None
                authority_kind, authority_id = "legacy_record", str(authority["legacy_record_id"])
            else:
                conn.rollback()
                return self._failure("rejected", "source_snapshot_mismatch", claim_id=request.claim_id)
            enqueue_memory_projections(
                conn,
                event_id=event_id,
                authority_kind=authority_kind,
                authority_id=authority_id,
                aggregate_id=f"memory:{request.claim_id}",
                aggregate_version=request.expected_current_version,
                payload_sha256=str(authority["content_sha256"]),
            )
            # Re-open the already-applied version's projection jobs as explicit
            # invalidation work; the outbox uniqueness key remains unchanged.
            conn.execute(
                """
                UPDATE projection_outbox
                SET status = 'pending', lease_owner = NULL, lease_until = NULL,
                    applied_at = NULL, next_retry_at = NULL, last_error = NULL
                WHERE aggregate_id = ? AND aggregate_version = ?
                  AND projection_kind IN ('memory_fts', 'memory_memfs')
                """,
                (f"memory:{request.claim_id}", request.expected_current_version),
            )
            conn.commit()
            return MemoryWriteReceipt(
                True, target.value, request.claim_id, request.expected_current_version,
                governance_id, True, "pending", False, decision.reason_code,
            )
        except BaseException:
            conn.rollback()
            raise

    def retire(self, request: MemoryLifecycleRequest) -> MemoryWriteReceipt:
        return self._lifecycle(request, action="retire", target=LifecycleState.RETIRED)

    def expire(self, request: MemoryLifecycleRequest) -> MemoryWriteReceipt:
        return self._lifecycle(request, action="expire", target=LifecycleState.EXPIRED)

    def replace(self, request: MemoryReplaceRequest) -> MemoryWriteReceipt:
        if type(request) is not MemoryReplaceRequest:
            raise TypeError("request must be MemoryReplaceRequest")
        if not evaluate_memory_write(request.content, target_file="memory").allowed:
            return self._failure("rejected", policy.REASON_SECRET_WRITE, claim_id=request.claim_id)

        conn = self._store._conn
        external_id = f"memory-replace:{request.idempotency_key}"
        source_hash = self._source_hash(request)
        try:
            conn.execute("BEGIN IMMEDIATE")
            existing = conn.execute(
                "SELECT source_hash, kind, event_id FROM ingest_events WHERE external_id = ?",
                (external_id,),
            ).fetchone()
            if existing is not None:
                if str(existing["source_hash"]) != source_hash or str(existing["kind"]) != "memory_replace":
                    conn.rollback()
                    return self._failure("conflict", "idempotency_conflict", claim_id=request.claim_id)
                row = conn.execute(
                    """
                    SELECT v.version, g.memory_governance_id, g.new_state
                    FROM memory_claim_sources s
                    JOIN memory_claim_versions v ON v.claim_version_id = s.claim_version_id
                    JOIN memory_governance_events g
                      ON g.claim_version_id = v.claim_version_id AND g.action = 'activate'
                    WHERE s.source_kind = 'event' AND s.event_id = ?
                    ORDER BY g.memory_governance_id DESC LIMIT 1
                    """,
                    (int(existing["event_id"]),),
                ).fetchone()
                conn.commit()
                if row is None:
                    return self._failure("conflict", "idempotency_conflict", claim_id=request.claim_id)
                return MemoryWriteReceipt(
                    True, str(row["new_state"]), request.claim_id, int(row["version"]),
                    int(row["memory_governance_id"]), True, "pending", False,
                    policy.REASON_TRANSITION_ALLOWED,
                )

            current = conn.execute(
                """
                SELECT c.target, c.memory_type, v.claim_version_id, v.version,
                       v.sensitivity, mc.lifecycle_state
                FROM memory_current mc
                JOIN memory_claims c ON c.claim_id = mc.claim_id
                JOIN memory_claim_versions v ON v.claim_version_id = mc.claim_version_id
                WHERE mc.claim_id = ?
                """,
                (request.claim_id,),
            ).fetchone()
            if current is None:
                conn.rollback()
                return self._failure("not_found", "claim_not_found", claim_id=request.claim_id)
            if int(current["version"]) != request.expected_current_version:
                conn.rollback()
                return self._failure("conflict", "stale_expected_version", claim_id=request.claim_id)
            if str(current["lifecycle_state"]) != LifecycleState.ACTIVE.value:
                conn.rollback()
                return self._failure("conflict", policy.REASON_INVALID_TRANSITION, claim_id=request.claim_id)

            supersede = policy.resolve_transition(
                LifecycleState.ACTIVE,
                MemoryTransitionCommand("supersede", LifecycleState.SUPERSEDED),
            )
            if not supersede.allowed:
                conn.rollback()
                return self._failure("rejected", supersede.reason_code, claim_id=request.claim_id)

            if request.lineage_kind is LineageKind.EVENT_DERIVED:
                if len(request.source_refs) != 1:
                    conn.rollback()
                    return self._failure("rejected", policy.REASON_SOURCE_SNAPSHOT_MISSING, claim_id=request.claim_id)
                ref = request.source_refs[0]
                if ref.authority_kind != "event":
                    conn.rollback()
                    return self._failure("rejected", policy.REASON_SOURCE_SNAPSHOT_MISSING, claim_id=request.claim_id)
                if (
                    type(ref.object_id) is not str
                    or not ref.object_id.isascii()
                    or not ref.object_id.isdecimal()
                    or ref.object_id.startswith("0")
                ):
                    conn.rollback()
                    return self._failure("rejected", policy.REASON_SOURCE_SNAPSHOT_MISMATCH, claim_id=request.claim_id)
                try:
                    source_event_id = int(ref.object_id)
                except (TypeError, ValueError):
                    conn.rollback()
                    return self._failure("rejected", policy.REASON_SOURCE_SNAPSHOT_MISMATCH, claim_id=request.claim_id)
                event_row = conn.execute(
                    """SELECT event_id, source_revision, payload_sha256, sensitivity,
                              evidence_state, inject_policy, role, source, content
                       FROM events WHERE event_id=?""", (source_event_id,),
                ).fetchone()
                command = None if event_row is None else parse_memory_command(str(event_row["content"]))
                if (
                    event_row is None or str(event_row["evidence_state"]) != "active"
                    or str(event_row["inject_policy"]) != "candidate_only"
                    or str(event_row["role"]) != "user"
                    or str(event_row["source"]) not in {"chat", "hermes_state_db"}
                    or command is None or command[0] != "replace"
                    or command[2] != request.content
                    or int(event_row["source_revision"]) != ref.object_version
                    or str(event_row["payload_sha256"]) != ref.payload_sha256
                ):
                    conn.rollback()
                    return self._failure("rejected", policy.REASON_SOURCE_SNAPSHOT_MISMATCH, claim_id=request.claim_id)
                event_id = int(event_row["event_id"])
                event_governance = conn.execute(
                    """SELECT governance_id, new_state FROM event_governance
                       WHERE event_id=? ORDER BY governance_id DESC LIMIT 1""",
                    (event_id,),
                ).fetchone()
                if event_governance is None:
                    inserted_governance = conn.execute(
                        """INSERT INTO event_governance(
                               event_id, action, previous_state, new_state, actor, reason
                           ) VALUES (?, 'activate', NULL, 'active', 'memory_transition_service',
                                     'derived replacement source admitted')""",
                        (event_id,),
                    )
                    event_governance_id = int(inserted_governance.lastrowid)
                elif str(event_governance["new_state"]) == LifecycleState.ACTIVE.value:
                    event_governance_id = int(event_governance["governance_id"])
                else:
                    conn.rollback()
                    return self._failure(
                        "rejected", policy.REASON_SOURCE_SNAPSHOT_MISMATCH,
                        claim_id=request.claim_id,
                    )
                conn.execute(
                    """INSERT INTO ingest_events(
                           external_id, source_hash, event_id, kind, payload_metadata
                       ) VALUES (?, ?, ?, 'memory_replace', ?)""",
                    (external_id, source_hash, event_id, json.dumps({
                        "claim_id": request.claim_id,
                        "expected_current_version": request.expected_current_version,
                    }, separators=(",", ":"))),
                )
            else:
                if request.source_refs:
                    conn.rollback()
                    return self._failure("rejected", policy.REASON_SOURCE_SNAPSHOT_MISSING, claim_id=request.claim_id)
                event_id = self._store.ingest_external_event_in_transaction(
                    external_id=external_id, source_hash=source_hash, kind="memory_replace",
                    payload_metadata={"claim_id": request.claim_id, "expected_current_version": request.expected_current_version},
                    session_id=request.session_id, conversation_id=request.conversation_id,
                    platform=request.platform, role="user", source="memory_replace",
                    content=request.content, persona_mode=request.mode_scope[0].value,
                    sensitivity=request.sensitivity.value, category="memory_assertion",
                    subcategory="replacement", inject_policy=request.injection_policy,
                )
                event_row = conn.execute(
                    "SELECT source_revision, payload_sha256, sensitivity FROM events WHERE event_id = ?",
                    (event_id,),
                ).fetchone()
                event_governance = conn.execute(
                    """INSERT INTO event_governance(event_id, action, previous_state, new_state, actor, reason)
                       VALUES (?, 'activate', NULL, 'active', 'memory_transition_service', 'explicit memory replacement')""",
                    (event_id,),
                )
                event_governance_id = int(event_governance.lastrowid)
                ref = AuthorityRef("event", str(event_id), int(event_row["source_revision"]), str(event_row["payload_sha256"]))
            snapshot = AuthoritySnapshot(
                "event", str(event_id), int(event_row["source_revision"]), str(event_row["payload_sha256"]),
                event_governance_id, LifecycleState.ACTIVE,
                Sensitivity(str(event_row["sensitivity"])), LifecycleState.ACTIVE, (), None,
            )
            write_decision = policy.admit_write(
                MemoryWriteCommand(
                    request.lineage_kind,
                    request.sensitivity,
                    request.mode_scope,
                    (ref,),
                    LifecycleState.ACTIVE,
                ),
                (snapshot,),
            )
            if not write_decision.allowed:
                conn.rollback()
                return self._failure("rejected", write_decision.reason_code, claim_id=request.claim_id)

            next_version = request.expected_current_version + 1
            content_hash = sha256_text(request.content)
            version = conn.execute(
                """
                INSERT INTO memory_claim_versions(
                    claim_id, version, content, content_sha256, confidence, sensitivity,
                    injection_policy, mode_scope, lineage_kind, schema_version
                ) VALUES (?, ?, ?, ?, 1.0, ?, ?, ?, ?, 1)
                """,
                (
                    request.claim_id, next_version, request.content, content_hash,
                    (write_decision.effective_sensitivity or request.sensitivity).value,
                    request.injection_policy,
                    json.dumps([mode.value for mode in request.mode_scope], separators=(",", ":")),
                    request.lineage_kind.value,
                ),
            )
            new_version_id = int(version.lastrowid)
            conn.execute(
                """
                INSERT INTO memory_claim_sources(
                    claim_version_id, source_kind, event_id, event_revision, event_payload_sha256
                ) VALUES (?, 'event', ?, ?, ?)
                """,
                (new_version_id, event_id, int(event_row["source_revision"]), str(event_row["payload_sha256"])),
            )
            conn.execute(
                """
                INSERT INTO memory_governance_events(
                    claim_id, claim_version_id, action, previous_state, new_state,
                    actor, reason_code, policy_version
                ) VALUES (?, ?, 'supersede', 'active', 'superseded',
                          'memory_transition_service', ?, ?)
                """,
                (request.claim_id, int(current["claim_version_id"]), supersede.reason_code, supersede.policy_version),
            )
            conn.execute(
                """
                INSERT INTO memory_governance_events(
                    claim_id, claim_version_id, action, previous_state, new_state,
                    actor, reason_code, policy_version
                ) VALUES (?, ?, 'submit', NULL, 'pending_review',
                          'memory_transition_service', 'replacement_submitted', ?)
                """,
                (request.claim_id, new_version_id, policy.POLICY_VERSION),
            )
            activation = conn.execute(
                """
                INSERT INTO memory_governance_events(
                    claim_id, claim_version_id, action, previous_state, new_state,
                    actor, reason_code, policy_version
                ) VALUES (?, ?, 'activate', 'pending_review', 'active',
                          'memory_transition_service', ?, ?)
                """,
                (request.claim_id, new_version_id, write_decision.reason_code, write_decision.policy_version),
            )
            governance_id = int(activation.lastrowid)
            switched = conn.execute(
                """
                UPDATE memory_current
                SET claim_version_id = ?, memory_governance_id = ?, lifecycle_state = 'active',
                    updated_at = strftime('%Y-%m-%dT%H:%M:%fZ', 'now')
                WHERE claim_id = ? AND claim_version_id = ? AND lifecycle_state = 'active'
                """,
                (new_version_id, governance_id, request.claim_id, int(current["claim_version_id"])),
            )
            if switched.rowcount != 1:
                conn.rollback()
                return self._failure("conflict", "stale_expected_version", claim_id=request.claim_id)
            enqueue_memory_projections(
                conn,
                event_id=event_id,
                authority_kind="event",
                authority_id=str(event_id),
                aggregate_id=f"memory:{request.claim_id}",
                aggregate_version=next_version,
                payload_sha256=content_hash,
            )
            conn.commit()
            return MemoryWriteReceipt(
                True, "active", request.claim_id, next_version, governance_id,
                True, "pending", False, policy.REASON_TRANSITION_ALLOWED,
            )
        except BaseException:
            conn.rollback()
            raise
