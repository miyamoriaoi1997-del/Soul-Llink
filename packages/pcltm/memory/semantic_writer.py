"""Governed writer for PCLTM semantic memory."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from enum import StrEnum
import re
from typing import Iterable

from .conflict_resolver import ConflictResolution, ConflictResolver
from .semantic_store import SemanticStore
from .temporal_fact import (
    SemanticNamespace,
    Stability,
    TemporalFact,
    default_conflict_group,
    new_memory_id,
    utc_now,
)


class WriteDecision(StrEnum):
    ACCEPTED = "accepted"
    REJECTED = "rejected"


@dataclass(frozen=True)
class SemanticWriteRequest:
    """A candidate fact before governance is applied."""

    subject: str
    predicate: str
    object: str
    namespace: str
    source_refs: tuple[str, ...]
    write_reason: str
    confidence: float = 0.75
    stability: str = Stability.MEDIUM.value
    valid_from: datetime | None = None
    last_verified_at: datetime | None = None
    conflict_group: str | None = None
    explicit: bool = False
    repeated_observation: bool = False
    verified: bool = False


@dataclass(frozen=True)
class SemanticWriteResult:
    decision: WriteDecision
    fact: TemporalFact | None = None
    reason: str = ""
    resolution: ConflictResolution | None = None


_TEMPORARY_PATTERNS = [
    re.compile(pattern, re.IGNORECASE)
    for pattern in (
        r"\btoday\b",
        r"\btomorrow\b",
        r"\bthis week\b",
        r"\bfor now\b",
        r"\btemporar(?:y|ily)\b",
        r"\bcurrent task\b",
        r"\bissue\s*#?\d+\b",
        r"\bpr\s*#?\d+\b",
        r"\bcommit\b",
        r"\bbranch\b",
        r"\bjust now\b",
        r"\b本轮\b",
        r"\b临时\b",
        r"\b今天\b",
        r"\b明天\b",
        r"\b这周\b",
        r"\b当前任务\b",
        r"\b提交\b",
    )
]


class SemanticWriter:
    """Applies write governance before committing semantic memory."""

    def __init__(self, store: SemanticStore) -> None:
        self.store = store
        self.resolver = ConflictResolver(store)

    def add(self, request: SemanticWriteRequest) -> SemanticWriteResult:
        rejection = self._rejection_reason(request)
        if rejection:
            return SemanticWriteResult(decision=WriteDecision.REJECTED, reason=rejection)

        now = utc_now()
        namespace = str(request.namespace)
        conflict_group = request.conflict_group or default_conflict_group(
            namespace, request.subject, request.predicate
        )
        fact = TemporalFact(
            memory_id=new_memory_id(),
            subject=request.subject.strip(),
            predicate=request.predicate.strip(),
            object=request.object.strip(),
            confidence=request.confidence,
            valid_from=request.valid_from or now,
            valid_until=None,
            source_refs=tuple(request.source_refs),
            stability=request.stability,
            namespace=namespace,
            conflict_group=conflict_group,
            supersedes=(),
            superseded_by=None,
            write_reason=request.write_reason.strip(),
            last_verified_at=request.last_verified_at or (now if request.verified else None),
        )
        resolution = self.resolver.resolve(fact, now=now)
        return SemanticWriteResult(
            decision=WriteDecision.ACCEPTED,
            fact=resolution.fact,
            reason=resolution.reason,
            resolution=resolution,
        )

    def search(self, **kwargs) -> list[TemporalFact]:
        return self.store.search(**kwargs)

    def update(self, fact: TemporalFact) -> TemporalFact:
        return self.store.update(fact)

    def delete(self, memory_id: str) -> bool:
        return self.store.delete(memory_id)

    def _rejection_reason(self, request: SemanticWriteRequest) -> str | None:
        if not request.source_refs:
            return "semantic_memory_requires_source_refs"
        if not request.write_reason.strip():
            return "semantic_memory_requires_write_reason"
        if not request.subject.strip() or not request.predicate.strip() or not request.object.strip():
            return "semantic_memory_requires_subject_predicate_object"
        try:
            SemanticNamespace(request.namespace)
        except ValueError:
            return "semantic_memory_namespace_not_allowed"
        try:
            Stability(request.stability)
        except ValueError:
            return "semantic_memory_stability_not_allowed"
        if request.namespace == SemanticNamespace.ENVIRONMENT_FACT.value and not request.verified:
            return "environment_fact_requires_verification"
        if self._looks_temporary(request):
            return "semantic_memory_rejects_temporary_or_task_state"
        if not (request.explicit or request.repeated_observation or request.verified):
            return "semantic_memory_requires_explicit_repeated_or_verified_basis"
        if request.confidence < 0.5:
            return "semantic_memory_confidence_too_low"
        return None

    @staticmethod
    def _looks_temporary(request: SemanticWriteRequest) -> bool:
        haystack = " ".join(
            [
                request.subject,
                request.predicate,
                request.object,
                request.write_reason,
                " ".join(request.source_refs),
            ]
        )
        return any(pattern.search(haystack) for pattern in _TEMPORARY_PATTERNS)


def make_request(
    *,
    subject: str,
    predicate: str,
    object: str,
    namespace: str | SemanticNamespace,
    source_refs: Iterable[str],
    write_reason: str,
    confidence: float = 0.75,
    stability: str | Stability = Stability.MEDIUM,
    valid_from: datetime | None = None,
    last_verified_at: datetime | None = None,
    conflict_group: str | None = None,
    explicit: bool = False,
    repeated_observation: bool = False,
    verified: bool = False,
) -> SemanticWriteRequest:
    return SemanticWriteRequest(
        subject=subject,
        predicate=predicate,
        object=object,
        namespace=str(namespace),
        source_refs=tuple(source_refs),
        write_reason=write_reason,
        confidence=confidence,
        stability=str(stability),
        valid_from=valid_from,
        last_verified_at=last_verified_at,
        conflict_group=conflict_group,
        explicit=explicit,
        repeated_observation=repeated_observation,
        verified=verified,
    )
