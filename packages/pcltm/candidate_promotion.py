"""Automatic candidate-to-claim promotion with confidence guardrails.

PCLTM owns the persona memory pipeline: the state machine decides modes, the
ingestor classifies events, the extractor surfaces candidates, and this module
applies the promotion guardrails:

* confidence >= 0.85  -> promote to an active claim (auto-activate)
* confidence 0.6-0.85 -> enqueue as pending for human review
* confidence < 0.6    -> drop

Deduplication follows the existing supersede path: if a claim already exists
for the same canonical key, the incoming content supersedes it (new version)
instead of creating a conflicting duplicate.
"""

from __future__ import annotations

import json
import math
from dataclasses import dataclass, field
from typing import Any

from .evidence_chain import sha256_text
from .memory_contracts import (
    AuthorityRef,
    LineageKind,
    MemoryWriteReceipt,
    PersonaMode,
    Sensitivity,
)
from .memory_transition_service import (
    MemoryLifecycleRequest,
    MemoryReplaceRequest,
    MemoryTransitionService,
)
from .memory_write_service import MemoryWriteRequest, MemoryWriteService
from .store import EventStore

# confidence thresholds (guardrail policy)
AUTO_ACTIVATE_CONFIDENCE = 0.85
PENDING_REVIEW_CONFIDENCE = 0.6

# map state-machine modes to the narrower PersonaMode enum used by claims
_PERSONA_MODE_BY_STATE = {
    "work": PersonaMode.WORK,
    "system_maintenance": PersonaMode.WORK,
    "daily": PersonaMode.DAILY,
    "intimacy": PersonaMode.DAILY,
    "conflict": PersonaMode.DAILY,
    "repair": PersonaMode.DAILY,
    "sex_candidate": PersonaMode.SEX,
    "sex": PersonaMode.SEX,
}


@dataclass(frozen=True, slots=True)
class PromotionOutcome:
    """Per-candidate outcome of one promotion pass."""

    candidate_id: str
    decision: str  # activated | duplicate | pending | conflict | retracted | dropped | superseded | rejected | error
    reason: str
    claim_id: int | None = None
    claim_version: int | None = None
    target_file: str = ""


@dataclass(frozen=True, slots=True)
class PromotionReport:
    """Aggregate result of one promotion pass."""

    scanned: int = 0
    activated: int = 0
    pending: int = 0
    dropped: int = 0
    superseded: int = 0
    rejected: int = 0
    failed: int = 0
    outcomes: tuple[PromotionOutcome, ...] = field(default_factory=tuple)


class CandidatePromotionService:
    """Apply guardrail policy to extracted candidates, promoting to claims."""

    def __init__(self, store: EventStore) -> None:
        self._store = store

    # ------------------------------------------------------------------ #
    # public API
    # ------------------------------------------------------------------ #
    def promote(self, candidates: list[dict[str, Any]]) -> PromotionReport:
        if not candidates:
            return PromotionReport()
        outcomes: list[PromotionOutcome] = []
        for candidate in candidates:
            if type(candidate) is not dict:
                outcomes.append(PromotionOutcome("", "error", "malformed_candidate"))
                continue
            invalid = self._validate_candidate(candidate)
            if invalid is not None:
                outcomes.append(invalid)
                continue
            outcomes.append(self._promote_durable(candidate))
        report = PromotionReport(
            scanned=len(outcomes),
            activated=sum(o.decision == "activated" for o in outcomes),
            pending=sum(o.decision in {"pending", "conflict"} for o in outcomes),
            dropped=sum(o.decision == "dropped" for o in outcomes),
            superseded=sum(o.decision == "superseded" for o in outcomes),
            rejected=sum(o.decision == "rejected" for o in outcomes),
            failed=sum(o.decision == "error" for o in outcomes),
            outcomes=tuple(outcomes),
        )
        return report

    @staticmethod
    def _validate_candidate(candidate: dict[str, Any]) -> PromotionOutcome | None:
        raw_candidate_id = candidate.get("candidate_id")
        candidate_id = raw_candidate_id if type(raw_candidate_id) is str else ""
        target_file = str(candidate.get("target_file") or "USER.md")
        confidence = candidate.get("confidence")
        if isinstance(confidence, bool) or type(confidence) not in {int, float}:
            return PromotionOutcome(candidate_id, "error", "invalid_confidence", target_file=target_file)
        if not math.isfinite(float(confidence)) or not 0.0 <= float(confidence) <= 1.0:
            return PromotionOutcome(candidate_id, "error", "invalid_confidence", target_file=target_file)
        content = candidate.get("content")
        if (
            not candidate_id
            or type(content) is not str
            or (not content.strip() and candidate.get("identity_action") != "forget")
        ):
            return PromotionOutcome(candidate_id, "error", "malformed_candidate", target_file=target_file)
        mode = candidate.get("mode")
        if type(mode) is not str or mode not in _PERSONA_MODE_BY_STATE:
            return PromotionOutcome(candidate_id, "error", "invalid_mode", target_file=target_file)
        sensitivity = candidate.get("sensitivity")
        if type(sensitivity) is not str or sensitivity not in {item.value for item in Sensitivity}:
            return PromotionOutcome(candidate_id, "error", "invalid_sensitivity", target_file=target_file)
        source_refs = candidate.get("source_refs", ())
        if type(source_refs) is not tuple or not all(type(ref) is AuthorityRef for ref in source_refs):
            return PromotionOutcome(candidate_id, "error", "malformed_candidate", target_file=target_file)
        if not source_refs:
            return PromotionOutcome(candidate_id, "rejected", "source_snapshot_missing", target_file=target_file)
        if (
            type(candidate.get("source_event_ids")) is not list
            or type(candidate.get("source_node_ids")) is not list
        ):
            return PromotionOutcome(candidate_id, "error", "malformed_candidate", target_file=target_file)
        return None

    @staticmethod
    def _request_sha256(candidate: dict[str, Any]) -> str:
        source_refs = candidate.get("source_refs") or ()
        payload = {
            **{key: value for key, value in candidate.items() if key != "source_refs"},
            "source_refs": [
                (ref.authority_kind, ref.object_id, ref.object_version, ref.payload_sha256)
                for ref in source_refs
            ],
        }
        return sha256_text(json.dumps(payload, ensure_ascii=False, sort_keys=True, default=str))

    def _finalize_receipt(
        self,
        candidate_id: str,
        request_sha256: str,
        outcome: PromotionOutcome,
    ) -> None:
        updated = self._store._conn.execute(
            """UPDATE candidate_promotion_receipts
               SET decision=?, reason=?, claim_id=?, claim_version=?, target_file=?
               WHERE candidate_id=? AND request_sha256=? AND decision='processing'""",
            (
                outcome.decision, outcome.reason, outcome.claim_id,
                outcome.claim_version, outcome.target_file, candidate_id,
                request_sha256,
            ),
        )
        if updated.rowcount != 1:
            persisted = self._store._conn.execute(
                "SELECT * FROM candidate_promotion_receipts WHERE candidate_id=?",
                (candidate_id,),
            ).fetchone()
            if (
                persisted is None
                or str(persisted["request_sha256"]) != request_sha256
                or str(persisted["decision"]) == "processing"
                or (
                    str(persisted["decision"]), str(persisted["reason"]),
                    persisted["claim_id"], persisted["claim_version"],
                    str(persisted["target_file"]),
                ) != (
                    outcome.decision, outcome.reason, outcome.claim_id,
                    outcome.claim_version, outcome.target_file,
                )
            ):
                raise RuntimeError("candidate receipt finalization conflict")
        self._store._conn.commit()

    def _promote_durable(self, candidate: dict[str, Any]) -> PromotionOutcome:
        candidate_id = str(candidate.get("candidate_id") or "")
        if not candidate_id:
            return PromotionOutcome(
                candidate_id, "error", "malformed_candidate",
                target_file=str(candidate.get("target_file") or "USER.md"),
            )
        request_sha256 = self._request_sha256(candidate)
        existing = self._store._conn.execute(
            "SELECT * FROM candidate_promotion_receipts WHERE candidate_id=?",
            (candidate_id,),
        ).fetchone()
        recovering = existing is not None and str(existing["decision"]) == "processing"
        if existing is not None:
            if str(existing["request_sha256"]) != request_sha256:
                return PromotionOutcome(candidate_id, "error", "idempotency_conflict")
            if str(existing["decision"]) != "processing":
                return PromotionOutcome(
                    candidate_id, str(existing["decision"]), str(existing["reason"]),
                    existing["claim_id"], existing["claim_version"], str(existing["target_file"]),
                )
        else:
            inserted = self._store._conn.execute(
                """INSERT OR IGNORE INTO candidate_promotion_receipts(
                       candidate_id, request_sha256, decision, reason, target_file
                   ) VALUES (?, ?, 'processing', 'in_progress', ?)""",
                (candidate_id, request_sha256, str(candidate.get("target_file") or "")),
            )
            self._store._conn.commit()
            recovering = inserted.rowcount == 0
            existing = self._store._conn.execute(
                "SELECT * FROM candidate_promotion_receipts WHERE candidate_id=?",
                (candidate_id,),
            ).fetchone()
            if existing is None:
                raise RuntimeError("candidate receipt creation failed")
            if str(existing["request_sha256"]) != request_sha256:
                return PromotionOutcome(candidate_id, "error", "idempotency_conflict")
            if str(existing["decision"]) != "processing":
                return PromotionOutcome(
                    candidate_id, str(existing["decision"]), str(existing["reason"]),
                    existing["claim_id"], existing["claim_version"], str(existing["target_file"]),
                )
        outcome = self._promote_one(candidate)
        if recovering:
            outcome = self._recover_processing_outcome(candidate, outcome)
        self._finalize_receipt(candidate_id, request_sha256, outcome)
        return outcome

    @staticmethod
    def _recover_processing_outcome(
        candidate: dict[str, Any], outcome: PromotionOutcome,
    ) -> PromotionOutcome:
        """Normalize lower-layer replay wording to the original command outcome."""
        if candidate.get("identity_action") == "memory" and outcome.decision == "duplicate":
            return PromotionOutcome(
                outcome.candidate_id, "activated", "write_allowed", outcome.claim_id,
                outcome.claim_version, outcome.target_file,
            )
        return outcome

    def approve_pending(
        self,
        candidate: dict[str, Any],
        *,
        reviewer: str,
        decision_reason: str,
    ) -> PromotionOutcome:
        """Promote a reviewed queue item without making the queue authoritative."""
        candidate_id = str(candidate.get("candidate_id") or "")
        request_sha256 = self._request_sha256(candidate)
        receipt = self._store._conn.execute(
            "SELECT * FROM candidate_promotion_receipts WHERE candidate_id=?",
            (candidate_id,),
        ).fetchone()
        if receipt is None or str(receipt["request_sha256"]) != request_sha256:
            return PromotionOutcome(candidate_id, "error", "pending_receipt_conflict")
        if str(receipt["decision"]) != "pending":
            if (
                str(receipt["reviewer"] or "") == reviewer
                and str(receipt["decision_reason"] or "") == decision_reason
                and str(receipt["decision"]) in {"activated", "duplicate", "superseded"}
            ):
                return PromotionOutcome(
                    candidate_id, str(receipt["decision"]), str(receipt["reason"]),
                    receipt["claim_id"], receipt["claim_version"], str(receipt["target_file"]),
                )
            return PromotionOutcome(candidate_id, "error", "pending_receipt_conflict")
        approved = {
            **candidate,
            "confidence": AUTO_ACTIVATE_CONFIDENCE,
            "requires_human_confirmation": False,
        }
        outcome = self._promote_one(approved, reviewed=True)
        if outcome.decision in {"activated", "duplicate", "superseded"}:
            self._store._conn.execute(
                """UPDATE candidate_promotion_receipts
                   SET decision=?, reason=?, claim_id=?, claim_version=?,
                       target_file=?, reviewer=?, decision_reason=?,
                       reviewed_at=strftime('%Y-%m-%dT%H:%M:%fZ', 'now')
                   WHERE candidate_id=? AND request_sha256=? AND decision='pending'""",
                (
                    outcome.decision, outcome.reason, outcome.claim_id,
                    outcome.claim_version, outcome.target_file, reviewer,
                    decision_reason, candidate_id, request_sha256,
                ),
            )
            self._store._conn.commit()
        return outcome

    # ------------------------------------------------------------------ #
    # per-candidate guardrail decision
    # ------------------------------------------------------------------ #
    def _promote_one(
        self, candidate: dict[str, Any], *, reviewed: bool = False,
    ) -> PromotionOutcome:
        candidate_id = str(candidate.get("candidate_id") or "")
        content = str(candidate.get("content") or "")
        candidate = dict(candidate)
        target_file = str(candidate.get("target_file") or "USER.md")
        raw_confidence = candidate.get("confidence")
        if isinstance(raw_confidence, bool) or type(raw_confidence) not in {int, float}:
            return PromotionOutcome(candidate_id, "error", "invalid_confidence", target_file=target_file)
        confidence = float(raw_confidence)
        if not math.isfinite(confidence) or not 0.0 <= confidence <= 1.0:
            return PromotionOutcome(candidate_id, "error", "invalid_confidence", target_file=target_file)
        if not candidate_id or (not content.strip() and candidate.get("identity_action") != "forget"):
            return PromotionOutcome(candidate_id, "error", "malformed_candidate", target_file=target_file)
        mode = candidate.get("mode")
        if type(mode) is not str or mode not in _PERSONA_MODE_BY_STATE:
            return PromotionOutcome(candidate_id, "error", "invalid_mode", target_file=target_file)
        sensitivity = candidate.get("sensitivity")
        if type(sensitivity) is not str or sensitivity not in {item.value for item in Sensitivity}:
            return PromotionOutcome(candidate_id, "error", "invalid_sensitivity", target_file=target_file)

        if confidence < PENDING_REVIEW_CONFIDENCE:
            return PromotionOutcome(candidate_id, "dropped", "confidence_below_pending_threshold", target_file=target_file)
        if candidate.get("requires_human_confirmation") is True:
            record_id = self._store.enqueue_candidate(candidate)
            return PromotionOutcome(
                candidate_id, "pending", f"human_confirmation_required:record={record_id}",
                target_file=target_file,
            )
        if confidence < AUTO_ACTIVATE_CONFIDENCE:
            record_id = self._store.enqueue_candidate(candidate)
            return PromotionOutcome(
                candidate_id, "pending", f"queued_for_review:record={record_id}",
                target_file=target_file,
            )

        source_refs = tuple(candidate.get("source_refs") or ())
        if not source_refs:
            return PromotionOutcome(
                candidate_id, "rejected", "source_snapshot_missing", target_file=target_file,
            )
        if not self._matches_event_authority(candidate, reviewed=reviewed):
            return PromotionOutcome(
                candidate_id, "error", "candidate_authority_mismatch", target_file=target_file,
            )

        if candidate.get("identity_action") == "forget":
            return self._retract(candidate)

        # auto-activate path: write, deduplicate, detect conflict, or supersede
        try:
            return self._activate(candidate)
        except Exception as exc:  # noqa: BLE001 - batch isolation is the API contract
            return PromotionOutcome(candidate_id, "error", f"promotion_exception:{type(exc).__name__}", target_file=target_file)

    def _activate(self, candidate: dict[str, Any]) -> PromotionOutcome:
        candidate_id = str(candidate["candidate_id"])
        content = str(candidate["content"])
        target_file = str(candidate["target_file"])
        mode = str(candidate["mode"])
        persona_mode = _PERSONA_MODE_BY_STATE[mode]
        kind = str(candidate.get("kind") or "user_preference")
        sensitivity = Sensitivity(str(candidate["sensitivity"]))
        canonical_key = self._canonical_key(candidate)
        identity_existed = self._store._conn.execute(
            "SELECT 1 FROM memory_claims WHERE canonical_key=?", (canonical_key,),
        ).fetchone() is not None

        request = MemoryWriteRequest(
            idempotency_key=f"candidate:{candidate_id}",
            content=content,
            canonical_key=canonical_key,
            target=target_file,
            memory_type=kind,
            sensitivity=sensitivity,
            mode_scope=(persona_mode,),
            injection_policy="allow",
            lineage_kind=LineageKind.EVENT_DERIVED,
            source_refs=tuple(candidate.get("source_refs") or ()),
            session_id="candidate-promotion",
            conversation_id="candidate-promotion",
            platform="pcltm",
        )
        receipt = MemoryWriteService(self._store).write(request)
        if identity_existed and receipt.success and candidate.get("identity_action") == "replace":
            return self._supersede(candidate, request)
        if identity_existed and receipt.success and receipt.reason_code == "write_allowed" and candidate.get("identity_action") == "memory":
            current = self._store._conn.execute(
                """SELECT v.content_sha256 FROM memory_current mc
                   JOIN memory_claim_versions v ON v.claim_version_id=mc.claim_version_id
                   WHERE mc.claim_id=?""", (receipt.claim_id,),
            ).fetchone()
            if current is not None and str(current["content_sha256"]) == sha256_text(request.content):
                return PromotionOutcome(
                    candidate_id, "duplicate", "already_active_identical",
                    receipt.claim_id, receipt.claim_version, target_file,
                )
        if receipt.success and receipt.reason_code == "canonical_key_exists":
            return self._resolve_existing_identity(candidate, request, receipt)
        if receipt.success:
            return PromotionOutcome(
                candidate_id, "activated", receipt.reason_code, receipt.claim_id,
                receipt.claim_version, target_file,
            )
        if receipt.reason_code == "canonical_key_conflict" and candidate.get("identity_action") == "replace":
            return self._supersede(candidate, request)
        return PromotionOutcome(candidate_id, "rejected", receipt.reason_code, target_file=target_file)

    def _resolve_existing_identity(
        self, candidate: dict[str, Any], request: MemoryWriteRequest, receipt: MemoryWriteReceipt,
    ) -> PromotionOutcome:
        candidate_id = str(candidate["candidate_id"])
        if receipt.claim_id is None or receipt.claim_version is None:
            return PromotionOutcome(candidate_id, "rejected", "canonical_identity_missing", target_file=request.target)
        current = self._store._conn.execute(
            """SELECT v.content_sha256 FROM memory_current mc
               JOIN memory_claim_versions v ON v.claim_version_id=mc.claim_version_id
               WHERE mc.claim_id=? AND mc.lifecycle_state='active'""",
            (receipt.claim_id,),
        ).fetchone()
        if current is None:
            return PromotionOutcome(candidate_id, "rejected", "canonical_identity_not_active", receipt.claim_id, target_file=request.target)
        if str(current["content_sha256"]) == sha256_text(request.content):
            return PromotionOutcome(
                candidate_id, "duplicate", "already_active_identical",
                receipt.claim_id, receipt.claim_version, request.target,
            )
        if candidate.get("identity_action") == "replace":
            return self._supersede(candidate, request)
        record_id = self._store.enqueue_candidate({**candidate, "conflict_with_claim_id": receipt.claim_id})
        return PromotionOutcome(
            candidate_id, "conflict", f"semantic_identity_conflict:record={record_id}",
            receipt.claim_id, receipt.claim_version, request.target,
        )

    def _retract(self, candidate: dict[str, Any]) -> PromotionOutcome:
        candidate_id = str(candidate["candidate_id"])
        target_file = str(candidate.get("target_file") or "USER.md")
        canonical_key = self._canonical_key(candidate)
        current = self._store._conn.execute(
            """SELECT c.claim_id, v.version FROM memory_claims c
               JOIN memory_current mc ON mc.claim_id=c.claim_id
               JOIN memory_claim_versions v ON v.claim_version_id=mc.claim_version_id
               WHERE c.canonical_key=? AND mc.lifecycle_state='active'""",
            (canonical_key,),
        ).fetchone()
        if current is None:
            replay = self._store._conn.execute(
                """SELECT r.claim_id, r.claim_version, r.memory_governance_id, g.new_state
                   FROM memory_transition_receipts r
                   JOIN memory_governance_events g
                     ON g.memory_governance_id=r.memory_governance_id
                   WHERE r.idempotency_key=? AND r.action='retire'""",
                (f"candidate:{candidate_id}:forget",),
            ).fetchone()
            if replay is not None:
                return PromotionOutcome(
                    candidate_id, "retracted", "transition_allowed",
                    int(replay["claim_id"]), int(replay["claim_version"]), target_file,
                )
            return PromotionOutcome(candidate_id, "rejected", "canonical_identity_not_active", target_file=target_file)
        receipt = MemoryTransitionService(self._store).retire(MemoryLifecycleRequest(
            idempotency_key=f"candidate:{candidate_id}:forget",
            claim_id=int(current["claim_id"]),
            expected_current_version=int(current["version"]),
            reason_code="explicit_memory_retraction",
        ))
        if receipt.success:
            return PromotionOutcome(
                candidate_id, "retracted", receipt.reason_code, receipt.claim_id,
                receipt.claim_version, target_file,
            )
        return PromotionOutcome(candidate_id, "rejected", receipt.reason_code, receipt.claim_id, target_file=target_file)

    def _supersede(self, candidate: dict[str, Any], request: MemoryWriteRequest) -> PromotionOutcome:
        """Existing claim for the canonical key: supersede it with the new content."""
        candidate_id = str(candidate["candidate_id"])
        conn = self._store._conn
        claim = conn.execute(
            "SELECT claim_id FROM memory_claims WHERE canonical_key = ? ORDER BY claim_id LIMIT 1",
            (request.canonical_key,),
        ).fetchone()
        if claim is None:
            return PromotionOutcome(candidate_id, "rejected", "canonical_key_conflict_but_no_claim", target_file=request.target)
        current = conn.execute(
            """
            SELECT v.version, v.content_sha256 FROM memory_current mc
            JOIN memory_claim_versions v ON v.claim_version_id = mc.claim_version_id
            WHERE mc.claim_id = ?
            """,
            (int(claim["claim_id"]),),
        ).fetchone()
        if current is None:
            return PromotionOutcome(candidate_id, "rejected", "claim_no_current_version", claim_id=int(claim["claim_id"]), target_file=request.target)
        if str(current["content_sha256"]) == sha256_text(request.content):
            replay = conn.execute(
                """SELECT 1 FROM ingest_events
                   WHERE external_id=? AND kind='memory_replace'""",
                (f"memory-replace:candidate:{candidate_id}",),
            ).fetchone()
            if replay is not None and candidate.get("identity_action") == "replace":
                return PromotionOutcome(
                    candidate_id, "superseded", "transition_allowed",
                    int(claim["claim_id"]), int(current["version"]), request.target,
                )
            # identical content already active: no-op
            return PromotionOutcome(
                candidate_id, "activated", "already_active_identical",
                int(claim["claim_id"]), int(current["version"]), request.target,
            )
        replace_request = MemoryReplaceRequest(
            idempotency_key=f"candidate:{candidate_id}",
            claim_id=int(claim["claim_id"]),
            expected_current_version=int(current["version"]),
            content=request.content,
            sensitivity=request.sensitivity,
            mode_scope=request.mode_scope,
            injection_policy=request.injection_policy,
            session_id="candidate-promotion",
            conversation_id="candidate-promotion",
            platform="pcltm",
            lineage_kind=request.lineage_kind,
            source_refs=request.source_refs,
        )
        receipt = MemoryTransitionService(self._store).replace(replace_request)
        if receipt.success:
            return PromotionOutcome(
                candidate_id, "superseded", receipt.reason_code, receipt.claim_id,
                receipt.claim_version, request.target,
            )
        return PromotionOutcome(candidate_id, "rejected", receipt.reason_code, claim_id=receipt.claim_id, target_file=request.target)

    # ------------------------------------------------------------------ #
    # helpers
    # ------------------------------------------------------------------ #
    def _matches_event_authority(
        self, candidate: dict[str, Any], *, reviewed: bool = False,
    ) -> bool:
        """Bind auto-activation semantics to a freshly reopened canonical event."""
        from .candidates import PersonaCandidateExtractor

        refs = tuple(candidate.get("source_refs") or ())
        if len(refs) != 1:
            return False
        ref = refs[0]
        if ref.authority_kind != "event":
            return False
        raw_id = ref.object_id
        if (
            type(raw_id) is not str
            or not raw_id.isascii()
            or not raw_id.isdecimal()
            or raw_id != str(int(raw_id))
            or int(raw_id) <= 0
        ):
            return False
        event = self._store.get_event(int(raw_id))
        if event is None:
            return False
        expected = PersonaCandidateExtractor(self._store)._candidate_from_event(event)
        if expected is None:
            return False
        if refs != tuple(expected.get("source_refs") or ()):
            return False
        bound_fields = (
            "kind", "target_file", "content", "mode", "sensitivity",
            "identity_action", "semantic_key", "canonical_key",
        )
        if not all(candidate.get(field) == expected.get(field) for field in bound_fields):
            return False
        return reviewed or candidate.get("confidence") == expected.get("confidence")

    @staticmethod
    def _canonical_key(candidate: dict[str, Any]) -> str:
        """Stable canonical key for the candidate.

        A candidate may carry an explicit ``canonical_key`` (semantic identity,
        e.g. from a dedup layer); otherwise we fall back to content addressing
        (target + kind + content hash). Identical content under the same key is
        idempotent; changed content under the same key supersedes the old claim.
        """
        explicit = candidate.get("canonical_key")
        if isinstance(explicit, str) and explicit.strip():
            return explicit.strip()
        content = str(candidate.get("content") or "")
        kind = str(candidate.get("kind") or "user_preference")
        target_file = str(candidate.get("target_file") or "USER.md")
        return f"persona:{target_file}:{kind}:{sha256_text(content)[:24]}"
