"""Conflict handling for governed semantic memory facts."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime

from .semantic_store import SemanticStore
from .temporal_fact import Stability, TemporalFact, utc_now


_STABILITY_RANK = {
    Stability.LOW.value: 0,
    Stability.MEDIUM.value: 1,
    Stability.HIGH.value: 2,
    Stability.VERIFIED.value: 3,
}


@dataclass(frozen=True)
class ConflictResolution:
    """Decision made for a candidate semantic fact."""

    fact: TemporalFact
    superseded: tuple[str, ...] = ()
    reason: str = "inserted"
    conflict_group: str | None = None


class ConflictResolver:
    """Resolve semantic fact conflicts without destructive overwrite."""

    def __init__(self, store: SemanticStore) -> None:
        self.store = store

    def resolve(self, candidate: TemporalFact, *, now: datetime | None = None) -> ConflictResolution:
        """Insert candidate and supersede lower-ranked active conflicts when justified.

        A conflict is scoped to ``conflict_group``.  The new fact is always written
        into that group first.  Older facts are only closed when the candidate is
        materially different and has at least the same authority score.
        """
        now = now or utc_now()
        group = candidate.conflict_group
        active_conflicts = self._active_conflicts(candidate)
        differing = [fact for fact in active_conflicts if fact.object != candidate.object]
        same = [fact for fact in active_conflicts if fact.object == candidate.object]

        if same and not differing:
            self.store.add(candidate)
            return ConflictResolution(
                fact=candidate,
                superseded=(),
                reason="same_fact_added_with_additional_provenance",
                conflict_group=group,
            )

        superseded: list[str] = []
        for old in differing:
            if self._can_supersede(candidate, old):
                superseded.append(old.memory_id)

        if superseded:
            candidate = candidate.with_supersession(supersedes=tuple(superseded))

        self.store.add(candidate)
        if superseded:
            self.store.mark_superseded(
                old_memory_ids=superseded,
                superseded_by=candidate.memory_id,
                valid_until=now,
            )
            reason = "candidate_superseded_conflicting_fact"
        elif differing:
            reason = "candidate_kept_in_conflict_group_without_supersession"
        else:
            reason = "inserted"

        return ConflictResolution(
            fact=candidate,
            superseded=tuple(superseded),
            reason=reason,
            conflict_group=group,
        )

    def _active_conflicts(self, candidate: TemporalFact) -> list[TemporalFact]:
        if candidate.conflict_group:
            facts = self.store.search(
                conflict_group=candidate.conflict_group,
                active_only=True,
                limit=200,
            )
        else:
            facts = self.store.search(
                namespace=candidate.namespace,
                subject=candidate.subject,
                predicate=candidate.predicate,
                active_only=True,
                limit=200,
            )
        return [fact for fact in facts if fact.memory_id != candidate.memory_id]

    @staticmethod
    def _can_supersede(candidate: TemporalFact, old: TemporalFact) -> bool:
        candidate_rank = _STABILITY_RANK[candidate.stability]
        old_rank = _STABILITY_RANK[old.stability]
        candidate_score = candidate.confidence + candidate_rank * 0.15
        old_score = old.confidence + old_rank * 0.15
        explicitly_sourced = bool(candidate.source_refs) and bool(candidate.write_reason.strip())
        newer_or_equal = candidate.valid_from >= old.valid_from
        return explicitly_sourced and newer_or_equal and candidate_score >= old_score
