"""Authority-bound admission of governed memory into the existing ContextPacket."""

from __future__ import annotations

import json
import sqlite3
from dataclasses import dataclass
from enum import Enum

from ..evidence_chain import sha256_text
from ..memory_contracts import (
    AccessSurface,
    AuthoritySnapshot,
    LifecycleState,
    MemoryAccessRequest,
    PersonaMode,
    Sensitivity,
)
from ..memory_policy import admit_injection
from ..memory_retrieval import (
    GovernedMemoryItem,
    GovernedMemoryRetrievalResult,
    MemoryRetrievalStatus,
    _authority_row,
    _read_snapshot,
    _source_refs,
)
from ..store import EventStore
from .arbitrator import InjectionArbitrator
from .candidate import CandidateType, InjectionCandidate
from .conflict_filter import _seal_governed_memory_candidate
from .context_packet import ContextPacket


class GovernedInjectionStatus(str, Enum):
    OK = "ok"
    ABSTAINED = "abstained"
    UNAVAILABLE = "unavailable"


@dataclass(frozen=True, slots=True)
class GovernedInjectionResult:
    status: GovernedInjectionStatus
    packet: ContextPacket | None = None
    reason: str | None = None

    def __post_init__(self) -> None:
        if self.status is GovernedInjectionStatus.OK:
            if self.packet is None or self.reason is not None:
                raise ValueError("ok requires packet and no reason")
        elif self.packet is not None or type(self.reason) is not str or not self.reason:
            raise ValueError("non-ok requires no packet and a reason")

    @classmethod
    def ok(cls, packet: ContextPacket) -> "GovernedInjectionResult":
        return cls(GovernedInjectionStatus.OK, packet, None)

    @classmethod
    def abstained(cls, reason: str) -> "GovernedInjectionResult":
        return cls(GovernedInjectionStatus.ABSTAINED, None, reason)

    @classmethod
    def unavailable(cls, reason: str) -> "GovernedInjectionResult":
        return cls(GovernedInjectionStatus.UNAVAILABLE, None, reason)


_MEMORY_TYPE_TO_CANDIDATE = {
    "preference": CandidateType.SEMANTIC_MEMORY,
    "user_preference": CandidateType.SEMANTIC_MEMORY,
    "memory_note": CandidateType.SEMANTIC_MEMORY,
}


def _reopen_for_injection(
    store: EventStore,
    item: GovernedMemoryItem,
    persona_mode: PersonaMode,
) -> tuple[InjectionCandidate | None, str]:
    row = _authority_row(store, item.claim_id)
    if row is None:
        return None, "authority_receipt_changed"
    content = str(row["content"])
    content_hash = sha256_text(content)
    try:
        claim_version_id = int(row["claim_version_id"])
        claim_version = int(row["version"])
        governance_id = int(row["memory_governance_id"])
        lifecycle = LifecycleState(str(row["lifecycle_state"]))
        governance_state = LifecycleState(str(row["new_state"]))
        sensitivity = Sensitivity(str(row["sensitivity"]))
        mode_scope = tuple(PersonaMode(value) for value in json.loads(str(row["mode_scope"])))
    except (KeyError, IndexError, TypeError, ValueError, json.JSONDecodeError):
        return None, "authority_receipt_changed"
    sources = _source_refs(store, claim_version_id)
    if sources is None:
        return None, "authority_receipt_changed"
    if (
        claim_version != item.claim_version
        or governance_id != item.governance_id
        or content_hash != item.content_sha256
        or content != item.content
        or str(row["policy_version"]) != item.policy_version
        or sources != item.source_refs
    ):
        return None, "authority_receipt_changed"
    snapshot = AuthoritySnapshot(
        authority_kind="memory_claim",
        object_id=str(item.claim_id),
        object_version=claim_version,
        payload_sha256=content_hash,
        governance_id=governance_id,
        governance_state=governance_state,
        sensitivity=sensitivity,
        lifecycle_state=lifecycle,
        source_refs=sources,
        projection_generation=None,
        mode_scope=mode_scope,
        injection_policy=str(row["injection_policy"]),
    )
    candidate_type = _MEMORY_TYPE_TO_CANDIDATE.get(str(row["memory_type"]))
    if candidate_type is None:
        return None, "memory_type_not_injectable"
    decision = admit_injection(
        snapshot,
        MemoryAccessRequest(AccessSurface.INJECT, persona_mode, Sensitivity.RESTRICTED),
    )
    if not decision.allowed:
        return None, "injection_policy_filtered"
    candidate = _seal_governed_memory_candidate(InjectionCandidate(
        key=f"memory:{item.claim_id}",
        content=content,
        type=candidate_type,
        source="pcltm.memory_current",
        confidence=1.0,
        relevance=0.0,
        freshness=0.0,
        metadata={
            "claim_id": item.claim_id,
            "claim_version": item.claim_version,
            "governance_id": item.governance_id,
            "content_sha256": item.content_sha256,
            "policy_version": item.policy_version,
            "source_refs": [
                {
                    "authority_kind": ref.authority_kind,
                    "object_id": ref.object_id,
                    "object_version": ref.object_version,
                    "payload_sha256": ref.payload_sha256,
                }
                for ref in sources
            ],
            "authority_verified": True,
            "rank_score_is_authority": False,
        },
    ))
    return candidate, "injection_allowed"


def build_governed_memory_context(
    store: EventStore,
    retrieval: GovernedMemoryRetrievalResult,
    *,
    persona_mode: PersonaMode,
    total_budget: int,
) -> GovernedInjectionResult:
    if type(retrieval) is not GovernedMemoryRetrievalResult:
        raise TypeError("retrieval must be GovernedMemoryRetrievalResult")
    if type(persona_mode) is not PersonaMode:
        raise TypeError("persona_mode must be PersonaMode")
    if type(total_budget) is not int or isinstance(total_budget, bool) or total_budget <= 0:
        raise ValueError("total_budget must be a positive int")
    if retrieval.status is MemoryRetrievalStatus.ABSTAINED:
        return GovernedInjectionResult.abstained("retrieval_no_answer")
    if retrieval.status is MemoryRetrievalStatus.UNAVAILABLE:
        return GovernedInjectionResult.unavailable("retrieval_unavailable")

    try:
        with _read_snapshot(store):
            candidates: list[InjectionCandidate] = []
            rejection_reason: str | None = None
            for item in retrieval.items:
                candidate, reason = _reopen_for_injection(store, item, persona_mode)
                if candidate is None:
                    if reason in {"authority_receipt_changed", "memory_type_not_injectable"}:
                        return GovernedInjectionResult.abstained(reason)
                    rejection_reason = reason
                    continue
                candidates.append(candidate)
            if not candidates:
                return GovernedInjectionResult.abstained(
                    rejection_reason or "injection_policy_filtered"
                )
            packet = InjectionArbitrator(total_budget=total_budget).arbitrate(
                candidates,
                metadata={
                    "authority": "pcltm.memory_current",
                    "retrieval_status": retrieval.status.value,
                    "persona_mode": persona_mode.value,
                },
            )
    except sqlite3.Error:
        return GovernedInjectionResult.unavailable("authority_store_unavailable")
    if not packet.sections:
        return GovernedInjectionResult.abstained("injection_budget_filtered")
    return GovernedInjectionResult.ok(packet)
