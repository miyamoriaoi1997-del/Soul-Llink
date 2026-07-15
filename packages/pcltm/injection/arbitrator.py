"""PCLTM injection arbitration core.

The arbitrator is the central policy layer that decides what context is injected
for a turn, how much of each layer is allowed, and why every selected or rejected
candidate was handled that way.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from hashlib import sha256
from typing import Iterable, Mapping, Sequence

from .budget_manager import BudgetManager, DEFAULT_BUDGET_FRACTIONS
from .candidate import CandidateType, InjectionCandidate, normalize_candidate
from .conflict_filter import ConflictFilter
from .context_packet import ContextPacket, build_context_packet


@dataclass(frozen=True)
class ArbitrationInput:
    candidates: Sequence[InjectionCandidate | Mapping[str, object]]
    total_budget: int
    allow_conflict_resolution: bool = False
    metadata: Mapping[str, object] = field(default_factory=dict)


@dataclass
class InjectionArbitrator:
    """Deterministic injection policy engine for one model turn."""

    total_budget: int = 800
    budget_fractions: Mapping[CandidateType, float] = field(default_factory=lambda: dict(DEFAULT_BUDGET_FRACTIONS))
    allow_conflict_resolution: bool = False

    def arbitrate(
        self,
        candidates: Iterable[InjectionCandidate | Mapping[str, object]],
        *,
        total_budget: int | None = None,
        allow_conflict_resolution: bool | None = None,
        metadata: Mapping[str, object] | None = None,
    ) -> ContextPacket:
        normalized = self.normalize_candidates(candidates)
        conflict_filter = ConflictFilter(
            allow_conflict_resolution=(
                self.allow_conflict_resolution
                if allow_conflict_resolution is None
                else allow_conflict_resolution
            )
        )
        accepted, conflict_decisions = conflict_filter.filter(normalized)
        deduped, duplicate_rejected = _deduplicate_candidates(accepted)
        manager = BudgetManager(
            total_budget=total_budget or self.total_budget,
            fractions=self.budget_fractions,
        )
        selected, allocation, budget_rejected = manager.select_within_budgets(deduped)
        conflict_rejected = [
            (decision.candidate, decision.reason)
            for decision in conflict_decisions
            if decision.action.value == "reject"
        ]
        return build_context_packet(
            selected,
            rejected=[*conflict_rejected, *duplicate_rejected, *budget_rejected],
            conflict_decisions=conflict_decisions,
            budget=allocation,
            metadata=metadata,
        )

    def replay(self, arbitration_input: ArbitrationInput) -> ContextPacket:
        return self.arbitrate(
            arbitration_input.candidates,
            total_budget=arbitration_input.total_budget,
            allow_conflict_resolution=arbitration_input.allow_conflict_resolution,
            metadata=arbitration_input.metadata,
        )

    @staticmethod
    def normalize_candidates(
        candidates: Iterable[InjectionCandidate | Mapping[str, object]],
    ) -> list[InjectionCandidate]:
        return [normalize_candidate(candidate) for candidate in candidates]


def _deduplicate_candidates(
    candidates: Sequence[InjectionCandidate],
) -> tuple[list[InjectionCandidate], list[tuple[InjectionCandidate, str]]]:
    deduped: list[InjectionCandidate] = []
    rejected: list[tuple[InjectionCandidate, str]] = []
    seen: set[str] = set()
    for candidate in candidates:
        fingerprint = _candidate_injection_fingerprint(candidate)
        if fingerprint in seen:
            rejected.append((candidate, "duplicate-injection-fingerprint"))
            continue
        seen.add(fingerprint)
        deduped.append(candidate)
    return deduped, rejected


def _candidate_injection_fingerprint(candidate: InjectionCandidate) -> str:
    normalized_content = " ".join(str(candidate.content).strip().split())
    payload = "\0".join((candidate.type.value, normalized_content))
    return sha256(payload.encode("utf-8")).hexdigest()


def arbitrate_injection(
    candidates: Iterable[InjectionCandidate | Mapping[str, object]],
    *,
    total_budget: int = 800,
    allow_conflict_resolution: bool = False,
    metadata: Mapping[str, object] | None = None,
) -> ContextPacket:
    return InjectionArbitrator(total_budget=total_budget).arbitrate(
        candidates,
        allow_conflict_resolution=allow_conflict_resolution,
        metadata=metadata,
    )
