"""Conflict filtering for PCLTM injection arbitration."""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Iterable

from .candidate import CandidateType, InjectionCandidate, RiskFlag


_GOVERNED_MEMORY_AUTHORITY = object()


def _seal_governed_memory_candidate(
    candidate: InjectionCandidate,
) -> InjectionCandidate:
    """Mark a candidate produced by the authority-reopening injection gate."""

    metadata = dict(candidate.metadata)
    metadata["_governed_memory_authority"] = _GOVERNED_MEMORY_AUTHORITY
    return InjectionCandidate(
        key=candidate.key,
        content=candidate.content,
        type=candidate.type,
        source=candidate.source,
        confidence=candidate.confidence,
        freshness=candidate.freshness,
        relevance=candidate.relevance,
        priority=candidate.priority,
        risk_flags=candidate.risk_flags,
        token_cost=candidate.token_cost,
        metadata=metadata,
    )


class ConflictAction(str, Enum):
    KEEP = "keep"
    REJECT = "reject"
    DOWNRANK = "downrank"


@dataclass(frozen=True)
class ConflictDecision:
    candidate: InjectionCandidate
    action: ConflictAction
    reason: str
    flags: tuple[RiskFlag | str, ...] = ()

    def to_dict(self) -> dict[str, object]:
        return {
            "key": self.candidate.key,
            "action": self.action.value,
            "reason": self.reason,
            "flags": [str(flag) for flag in self.flags],
        }


class ConflictFilter:
    """Applies hard layer-separation rules before budget arbitration."""

    def __init__(self, *, allow_conflict_resolution: bool = False) -> None:
        self.allow_conflict_resolution = allow_conflict_resolution

    def filter(
        self,
        candidates: Iterable[InjectionCandidate],
    ) -> tuple[list[InjectionCandidate], list[ConflictDecision]]:
        accepted: list[InjectionCandidate] = []
        decisions: list[ConflictDecision] = []
        for candidate in candidates:
            decision = self.evaluate(candidate)
            decisions.append(decision)
            if decision.action is ConflictAction.KEEP:
                accepted.append(decision.candidate)
            elif decision.action is ConflictAction.DOWNRANK:
                accepted.append(decision.candidate)
        return accepted, decisions

    def evaluate(self, candidate: InjectionCandidate) -> ConflictDecision:
        flags = set(candidate.risk_flags)

        if RiskFlag.CORE_CONFLICT in flags:
            return ConflictDecision(candidate, ConflictAction.REJECT, "conflicts-with-core-soul", tuple(flags))

        if RiskFlag.EMOTION_FACT_CONTAMINATION in flags:
            return ConflictDecision(candidate, ConflictAction.REJECT, "emotion-layer-cannot-write-facts", tuple(flags))

        if candidate.type is CandidateType.SEMANTIC_MEMORY:
            metadata = dict(candidate.metadata)
            marker = metadata.pop("_governed_memory_authority", None)
            required_receipt = {
                "authority_verified",
                "claim_id",
                "claim_version",
                "governance_id",
                "content_sha256",
                "policy_version",
                "source_refs",
            }
            if (
                marker is not _GOVERNED_MEMORY_AUTHORITY
                or metadata.get("authority_verified") is not True
                or not required_receipt.issubset(metadata)
            ):
                flagged = candidate.with_risk(RiskFlag.MISSING_SOURCE)
                return ConflictDecision(
                    flagged,
                    ConflictAction.REJECT,
                    "semantic-memory-needs-governed-authority",
                    flagged.risk_flags,
                )
            candidate = InjectionCandidate(
                key=candidate.key,
                content=candidate.content,
                type=candidate.type,
                source=candidate.source,
                confidence=candidate.confidence,
                freshness=candidate.freshness,
                relevance=candidate.relevance,
                priority=candidate.priority,
                risk_flags=candidate.risk_flags,
                token_cost=candidate.token_cost,
                metadata=metadata,
            )

        if candidate.type is CandidateType.EPISODIC_MEMORY:
            event_marker = candidate.metadata.get("event") or candidate.metadata.get("is_event")
            if event_marker is not True:
                flagged = candidate.with_risk(RiskFlag.PERMANENT_EPISODIC_CLAIM)
                return ConflictDecision(flagged, ConflictAction.REJECT, "episodic-memory-must-be-event", flagged.risk_flags)

        if candidate.type is CandidateType.PROCEDURAL_SKILL and candidate.metadata.get("needed") is not True:
            flagged = candidate.with_risk(RiskFlag.PROCEDURAL_NOT_NEEDED)
            return ConflictDecision(flagged, ConflictAction.REJECT, "procedural-memory-only-on-demand", flagged.risk_flags)

        conflict_flags = {
            RiskFlag.CURRENT_TASK_CONFLICT,
            RiskFlag.NEW_FACT_CONFLICT,
            RiskFlag.CONFLICTING_MEMORY,
        }
        if flags.intersection(conflict_flags) and not self.allow_conflict_resolution:
            return ConflictDecision(candidate, ConflictAction.REJECT, "conflict-not-requested", tuple(flags))

        if flags.intersection(conflict_flags):
            downranked = candidate.with_risk(*flags)
            object.__setattr__(downranked, "priority", max(candidate.priority or 0, 50))
            return ConflictDecision(downranked, ConflictAction.DOWNRANK, "conflict-included-for-resolution", downranked.risk_flags)

        return ConflictDecision(candidate, ConflictAction.KEEP, "accepted", candidate.risk_flags)
