from __future__ import annotations

import json
from collections.abc import Callable
from dataclasses import dataclass

from . import memory_policy as policy
from .classifier import parse_memory_command, parse_stable_memory_assertion
from .evidence_chain import sha256_text
from .memory_contracts import (
    AuthorityRef,
    AuthoritySnapshot,
    LifecycleState,
    LineageKind,
    MemoryWriteCommand,
    MemoryWriteReceipt,
    PersonaMode,
    Sensitivity,
)
from .projection_outbox import enqueue_memory_projections
from .secret_policy import evaluate_memory_write
from .store import EventStore


@dataclass(frozen=True, slots=True)
class MemoryWriteRequest:
    """Strict, immutable input for one explicit authority assertion."""

    idempotency_key: str
    content: str
    canonical_key: str
    target: str
    memory_type: str
    sensitivity: Sensitivity
    mode_scope: tuple[PersonaMode, ...]
    injection_policy: str
    lineage_kind: LineageKind = LineageKind.EXPLICIT_USER_ASSERTION
    session_id: str = "memory-write-service"
    conversation_id: str = "memory-assertion"
    platform: str = "internal"
    source_sensitivity: Sensitivity | None = None
    source_refs: tuple[AuthorityRef, ...] = ()

    def __post_init__(self) -> None:
        for name in ("idempotency_key", "content", "canonical_key", "target", "memory_type", "injection_policy"):
            value = getattr(self, name)
            if type(value) is not str or not value.strip():
                raise ValueError(f"{name} is required")
        for name in ("session_id", "conversation_id", "platform"):
            value = getattr(self, name)
            if type(value) is not str or not value.strip():
                raise ValueError(f"{name} is required")
        if type(self.sensitivity) is not Sensitivity:
            raise TypeError("sensitivity must be Sensitivity")
        if self.source_sensitivity is not None and type(self.source_sensitivity) is not Sensitivity:
            raise TypeError("source_sensitivity must be Sensitivity or None")
        if type(self.mode_scope) is not tuple or not self.mode_scope or not all(
            type(mode) is PersonaMode for mode in self.mode_scope
        ):
            raise TypeError("mode_scope must be a non-empty tuple of PersonaMode")
        if type(self.lineage_kind) is not LineageKind:
            raise TypeError("lineage_kind must be LineageKind")
        if type(self.source_refs) is not tuple or not all(type(ref) is AuthorityRef for ref in self.source_refs):
            raise TypeError("source_refs must be a tuple of AuthorityRef")


class MemoryWriteService:
    """Own one BEGIN IMMEDIATE/COMMIT boundary for authority writes."""

    def __init__(self, store: EventStore, *, fault_hook: Callable[[str], None] | None = None) -> None:
        self._store = store
        self._fault_hook = fault_hook

    def _checkpoint(self, name: str) -> None:
        if self._fault_hook is not None:
            self._fault_hook(name)

    @staticmethod
    def _command_source_hash(request: MemoryWriteRequest) -> str:
        payload = {
            "canonical_key": request.canonical_key,
            "content": request.content,
            "injection_policy": request.injection_policy,
            "lineage_kind": request.lineage_kind.value,
            "memory_type": request.memory_type,
            "mode_scope": [mode.value for mode in request.mode_scope],
            "session_id": request.session_id,
            "conversation_id": request.conversation_id,
            "platform": request.platform,
            "sensitivity": request.sensitivity.value,
            "source_sensitivity": request.source_sensitivity.value if request.source_sensitivity else None,
            "source_refs": [
                {
                    "authority_kind": ref.authority_kind,
                    "object_id": ref.object_id,
                    "object_version": ref.object_version,
                    "payload_sha256": ref.payload_sha256,
                }
                for ref in request.source_refs
            ],
            "target": request.target,
        }
        return sha256_text(json.dumps(payload, ensure_ascii=False, sort_keys=True, separators=(",", ":")))

    @staticmethod
    def _reject(reason_code: str) -> MemoryWriteReceipt:
        return MemoryWriteReceipt(False, "rejected", None, None, None, False, "none", False, reason_code)

    def _receipt_from_authority(self, external_id: str, *, kind: str) -> MemoryWriteReceipt:
        conn = self._store._conn
        row = conn.execute(
            """
            SELECT c.claim_id, v.version, g.memory_governance_id, g.new_state
            FROM ingest_events i
            JOIN memory_claim_sources s ON s.event_id = i.event_id
              AND s.event_revision = 1 AND s.source_kind = 'event'
            JOIN memory_claim_versions v ON v.claim_version_id = s.claim_version_id
            JOIN memory_claims c ON c.claim_id = v.claim_id
            JOIN memory_governance_events g ON g.claim_id = c.claim_id AND g.claim_version_id = v.claim_version_id
            WHERE i.external_id = ? AND i.kind = ?
            ORDER BY v.version DESC, g.memory_governance_id DESC LIMIT 1
            """,
            (external_id, kind),
        ).fetchone()
        if row is None:
            return self._reject("idempotency_conflict")
        return MemoryWriteReceipt(
            True, str(row["new_state"]), int(row["claim_id"]), int(row["version"]),
            int(row["memory_governance_id"]), True, "pending",
            False, "write_allowed",
        )

    def _receipt_by_canonical_key(self, canonical_key: str, *, reason_code: str = "canonical_key_exists") -> MemoryWriteReceipt:
        row = self._store._conn.execute(
            """SELECT c.claim_id, v.version, g.memory_governance_id, g.new_state
               FROM memory_claims c JOIN memory_claim_versions v ON v.claim_id = c.claim_id
               JOIN memory_governance_events g ON g.claim_id = c.claim_id AND g.claim_version_id = v.claim_version_id
               WHERE c.canonical_key = ? ORDER BY v.version DESC, g.memory_governance_id DESC LIMIT 1""",
            (canonical_key,),
        ).fetchone()
        if row is None:
            return self._reject("canonical_key_conflict")
        return MemoryWriteReceipt(
            True, str(row["new_state"]), int(row["claim_id"]), int(row["version"]),
            int(row["memory_governance_id"]), True, "pending", False, reason_code,
        )

    def write(self, request: MemoryWriteRequest) -> MemoryWriteReceipt:
        if type(request) is not MemoryWriteRequest:
            raise TypeError("request must be MemoryWriteRequest")
        secret_decision = evaluate_memory_write(request.content, target_file=request.target)
        if not secret_decision.allowed:
            return self._reject(policy.REASON_SECRET_WRITE)
        derived = request.lineage_kind is LineageKind.EVENT_DERIVED
        if derived:
            if len(request.source_refs) != 1 or request.source_refs[0].authority_kind != "event":
                return self._reject(policy.REASON_SOURCE_SNAPSHOT_MISSING)
            raw_event_id = request.source_refs[0].object_id
            if (
                type(raw_event_id) is not str
                or not raw_event_id.isascii()
                or not raw_event_id.isdecimal()
                or raw_event_id.startswith("0")
            ):
                return self._reject(policy.REASON_SOURCE_SNAPSHOT_MISMATCH)
            try:
                derived_event_id = int(raw_event_id)
            except (TypeError, ValueError):
                return self._reject(policy.REASON_SOURCE_SNAPSHOT_MISMATCH)
        else:
            if request.source_refs:
                return self._reject(policy.REASON_SOURCE_SNAPSHOT_MISSING)
            derived_event_id = None
        external_id = f"memory-derived:{request.idempotency_key}" if derived else f"memory-assertion:{request.idempotency_key}"
        ingest_kind = "memory_derived" if derived else "memory_assertion"
        source_hash = self._command_source_hash(request)
        conn = self._store._conn
        existing = self._store.find_ingest_event(external_id)
        if existing is not None:
            if existing["source_hash"] != source_hash or existing["kind"] != ingest_kind:
                return self._reject("idempotency_conflict")
            return self._receipt_from_authority(external_id, kind=ingest_kind)
        try:
            conn.execute("BEGIN IMMEDIATE")
            existing = conn.execute(
                "SELECT source_hash, kind FROM ingest_events WHERE external_id = ?", (external_id,)
            ).fetchone()
            if existing is not None:
                if existing["source_hash"] != source_hash or existing["kind"] != ingest_kind:
                    conn.rollback()
                    return self._reject("idempotency_conflict")
                receipt = self._receipt_from_authority(external_id, kind=ingest_kind)
                conn.commit()
                return receipt
            source_row = None
            source_ref = request.source_refs[0] if derived else None
            if derived:
                source_row = conn.execute(
                    """SELECT event_id, source_revision, payload_sha256, sensitivity,
                              evidence_state, inject_policy, role, source, content
                       FROM events WHERE event_id = ?""",
                    (derived_event_id,),
                ).fetchone()
                parsed = None if source_row is None else parse_memory_command(str(source_row["content"]))
                stable_assertion = (
                    None if source_row is None
                    else parse_stable_memory_assertion(str(source_row["content"]))
                )
                source_body = (
                    parsed[2] if parsed is not None
                    else stable_assertion.content if stable_assertion is not None
                    else None
                )
                source_action_allowed = (
                    parsed is not None and parsed[0] in {"memory", "replace"}
                ) or stable_assertion is not None
                if (
                    source_row is None
                    or str(source_row["evidence_state"]) != "active"
                    or str(source_row["inject_policy"]) != "candidate_only"
                    or str(source_row["role"]) != "user"
                    or str(source_row["source"]) not in {"chat", "hermes_state_db"}
                    or not source_action_allowed
                    or source_body != request.content
                    or int(source_row["source_revision"]) != source_ref.object_version
                    or str(source_row["payload_sha256"]) != source_ref.payload_sha256
                ):
                    conn.rollback()
                    return self._reject(policy.REASON_SOURCE_SNAPSHOT_MISMATCH)
            canonical_claim = conn.execute(
                "SELECT claim_id FROM memory_claims WHERE canonical_key = ?",
                (request.canonical_key,),
            ).fetchone()
            if canonical_claim is not None:
                conn.rollback()
                if not derived:
                    return self._reject("canonical_key_conflict")
                exact_replay = None
                if source_ref is not None:
                    exact_replay = conn.execute(
                        """SELECT 1 FROM memory_claims c
                           JOIN memory_claim_versions v ON v.claim_id=c.claim_id
                           JOIN memory_claim_sources s ON s.claim_version_id=v.claim_version_id
                           WHERE c.canonical_key=? AND v.content_sha256=?
                             AND s.source_kind='event' AND s.event_id=?
                             AND s.event_revision=? AND s.event_payload_sha256=? LIMIT 1""",
                        (request.canonical_key, sha256_text(request.content), derived_event_id,
                         source_ref.object_version, source_ref.payload_sha256),
                    ).fetchone()
                return self._receipt_by_canonical_key(
                    request.canonical_key,
                    reason_code="write_allowed" if exact_replay is not None else "canonical_key_exists",
                )
            if derived:
                event_id = int(source_row["event_id"])
                event_row = source_row
                ref = source_ref
                event_governance = conn.execute(
                    """SELECT governance_id, new_state FROM event_governance
                       WHERE event_id=? ORDER BY governance_id DESC LIMIT 1""",
                    (event_id,),
                ).fetchone()
                if event_governance is None:
                    inserted_governance = conn.execute(
                        """INSERT INTO event_governance(
                               event_id, action, previous_state, new_state, actor, reason
                           ) VALUES (?, 'activate', NULL, 'active', 'memory_write_service',
                                     'derived memory source admitted')""",
                        (event_id,),
                    )
                    event_governance_id = int(inserted_governance.lastrowid)
                elif str(event_governance["new_state"]) == LifecycleState.ACTIVE.value:
                    event_governance_id = int(event_governance["governance_id"])
                else:
                    conn.rollback()
                    return self._reject(policy.REASON_SOURCE_SNAPSHOT_MISMATCH)
                snapshot = AuthoritySnapshot(
                    "event", str(event_id), int(source_row["source_revision"]), str(source_row["payload_sha256"]),
                    event_governance_id, LifecycleState.ACTIVE,
                    Sensitivity(str(source_row["sensitivity"])), LifecycleState.ACTIVE, (), None,
                )
            else:
                event_id = self._store.ingest_external_event_in_transaction(
                external_id=external_id, source_hash=source_hash, kind=ingest_kind,
                payload_metadata={"canonical_key": request.canonical_key, "memory_type": request.memory_type},
                session_id=request.session_id, conversation_id=request.conversation_id, platform=request.platform,
                role="user", source="memory_assertion", content=request.content,
                persona_mode=request.mode_scope[0].value,
                sensitivity=(request.source_sensitivity or request.sensitivity).value,
                category="memory_assertion", subcategory="explicit", inject_policy=request.injection_policy,
                )
                self._checkpoint("assertion_after")
                event_row = conn.execute(
                "SELECT source_revision, payload_sha256, sensitivity FROM events WHERE event_id = ?", (event_id,)
                ).fetchone()
                event_governance = conn.execute(
                """
                INSERT INTO event_governance(
                    event_id, action, previous_state, new_state, actor, reason
                ) VALUES (?, 'activate', NULL, 'active', 'memory_write_service',
                          'explicit memory assertion')
                """,
                    (event_id,),
                )
                event_governance_id = int(event_governance.lastrowid)
                ref = AuthorityRef("event", str(event_id), int(event_row["source_revision"]), str(event_row["payload_sha256"]))
                snapshot = AuthoritySnapshot(
                "event", str(event_id), int(event_row["source_revision"]), str(event_row["payload_sha256"]),
                event_governance_id, LifecycleState.ACTIVE, Sensitivity(str(event_row["sensitivity"])),
                LifecycleState.ACTIVE, (), None,
                )
            command = MemoryWriteCommand(
                request.lineage_kind, request.sensitivity, request.mode_scope,
                (ref,), LifecycleState.ACTIVE,
            )
            decision = policy.admit_write(
                command, (snapshot,)
            )
            if not decision.allowed:
                conn.rollback()
                return self._reject(decision.reason_code)
            effective = decision.effective_sensitivity or request.sensitivity
            claim = conn.execute(
                "INSERT INTO memory_claims(canonical_key, target, memory_type) VALUES (?, ?, ?)",
                (request.canonical_key, request.target, request.memory_type),
            )
            claim_id = int(claim.lastrowid)
            content_sha256 = sha256_text(request.content)
            mode_scope = json.dumps([mode.value for mode in request.mode_scope], separators=(",", ":"))
            version = conn.execute(
                """
                INSERT INTO memory_claim_versions(
                    claim_id, version, content, content_sha256, confidence, sensitivity,
                    injection_policy, mode_scope, lineage_kind, schema_version
                ) VALUES (?, 1, ?, ?, 1.0, ?, ?, ?, ?, 1)
                """,
                (claim_id, request.content, content_sha256, effective.value, request.injection_policy,
                 mode_scope, request.lineage_kind.value),
            )
            version_id = int(version.lastrowid)
            self._checkpoint("claim_version_after")
            conn.execute(
                """
                INSERT INTO memory_claim_sources(
                    claim_version_id, source_kind, event_id, event_revision, event_payload_sha256
                ) VALUES (?, 'event', ?, ?, ?)
                """,
                (version_id, event_id, int(event_row["source_revision"]), str(event_row["payload_sha256"])),
            )
            governance = conn.execute(
                """
                INSERT INTO memory_governance_events(
                    claim_id, claim_version_id, action, previous_state, new_state,
                    actor, reason_code, policy_version
                ) VALUES (?, ?, 'activate', 'pending_review', 'active', ?, ?, ?)
                """,
                (claim_id, version_id, "memory_write_service", decision.reason_code, decision.policy_version),
            )
            governance_id = int(governance.lastrowid)
            conn.execute(
                "INSERT INTO memory_current(claim_id, claim_version_id, memory_governance_id, lifecycle_state) VALUES (?, ?, ?, 'active')",
                (claim_id, version_id, governance_id),
            )
            enqueue_memory_projections(
                conn, event_id=event_id, authority_kind="event",
                authority_id=str(event_id), aggregate_id=f"memory:{claim_id}",
                aggregate_version=1, payload_sha256=content_sha256,
            )
            self._checkpoint("outbox_before_commit")
            conn.commit()
            return MemoryWriteReceipt(True, "active", claim_id, 1, governance_id, True, "pending", False, decision.reason_code)
        except BaseException:
            conn.rollback()
            raise

    write_memory = write
