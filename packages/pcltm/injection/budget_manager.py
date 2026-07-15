"""Fixed-bucket token budget management for PCLTM injection."""

from __future__ import annotations

from dataclasses import dataclass, field
from math import floor
from typing import Mapping

from .candidate import CandidateType, InjectionCandidate


DEFAULT_BUDGET_FRACTIONS: Mapping[CandidateType, float] = {
    CandidateType.ACTIVE_DIALOGUE: 0.20,
    CandidateType.CURRENT_TASK: 0.20,
    CandidateType.SESSION_SPINE: 0.15,
    CandidateType.PERSONA_EMOTION: 0.08,
    CandidateType.EPISODIC_MEMORY: 0.15,
    CandidateType.SEMANTIC_MEMORY: 0.15,
    CandidateType.PROCEDURAL_SKILL: 0.07,
}

FIXED_RESERVE_TYPES = (
    CandidateType.SYSTEM,
    CandidateType.CORE_IDENTITY,
    CandidateType.RUNTIME_MODE,
)


@dataclass(frozen=True)
class BudgetAllocation:
    total_budget: int
    fixed_reserve: int
    bucket_limits: Mapping[CandidateType, int]
    used_by_bucket: Mapping[CandidateType, int]
    remaining: int

    def to_dict(self) -> dict[str, object]:
        return {
            "total_budget": self.total_budget,
            "fixed_reserve": self.fixed_reserve,
            "bucket_limits": {key.value: value for key, value in self.bucket_limits.items()},
            "used_by_bucket": {key.value: value for key, value in self.used_by_bucket.items()},
            "remaining": self.remaining,
        }


@dataclass
class BudgetManager:
    """Allocates fixed per-layer budgets so retrieval volume cannot dominate."""

    total_budget: int
    fractions: Mapping[CandidateType, float] = field(default_factory=lambda: dict(DEFAULT_BUDGET_FRACTIONS))
    min_bucket_tokens: int = 1

    def __post_init__(self) -> None:
        if self.total_budget <= 0:
            raise ValueError("total budget must be positive")

    def fixed_reserve(self, candidates: list[InjectionCandidate]) -> int:
        return sum(candidate.token_cost or 0 for candidate in candidates if candidate.type in FIXED_RESERVE_TYPES)

    def bucket_limits(self, fixed_reserve: int = 0) -> dict[CandidateType, int]:
        available = max(0, self.total_budget - fixed_reserve)
        limits: dict[CandidateType, int] = {}
        for candidate_type, fraction in self.fractions.items():
            limit = floor(available * fraction)
            if fraction > 0 and available > 0:
                limit = max(self.min_bucket_tokens, limit)
            limits[CandidateType(candidate_type)] = limit
        return limits

    def select_within_budgets(
        self,
        candidates: list[InjectionCandidate],
    ) -> tuple[list[InjectionCandidate], BudgetAllocation, list[tuple[InjectionCandidate, str]]]:
        """Select candidates without allowing one layer to consume another layer."""

        selected: list[InjectionCandidate] = []
        rejected: list[tuple[InjectionCandidate, str]] = []
        fixed = [candidate for candidate in candidates if candidate.type in FIXED_RESERVE_TYPES]
        variable = [candidate for candidate in candidates if candidate.type not in FIXED_RESERVE_TYPES]
        fixed_reserve = self.fixed_reserve(fixed)
        limits = self.bucket_limits(fixed_reserve)
        used: dict[CandidateType, int] = {candidate_type: 0 for candidate_type in limits}

        running_total = 0
        for candidate in sorted(fixed, key=lambda item: (item.priority, item.key)):
            cost = candidate.token_cost or 0
            if running_total + cost > self.total_budget:
                rejected.append((candidate, "fixed-reserve-exceeds-total-budget"))
                continue
            selected.append(candidate)
            running_total += cost

        for candidate in sorted(variable, key=lambda item: (item.priority, -item.rank_score, item.key)):
            limit = limits.get(candidate.type, 0)
            cost = candidate.token_cost or 0
            if limit <= 0:
                rejected.append((candidate, "no-budget-for-layer"))
                continue
            if used.get(candidate.type, 0) + cost > limit:
                rejected.append((candidate, "layer-budget-exceeded"))
                continue
            if running_total + cost > self.total_budget:
                rejected.append((candidate, "total-budget-exceeded"))
                continue
            selected.append(candidate)
            used[candidate.type] = used.get(candidate.type, 0) + cost
            running_total += cost

        allocation = BudgetAllocation(
            total_budget=self.total_budget,
            fixed_reserve=fixed_reserve,
            bucket_limits=limits,
            used_by_bucket=used,
            remaining=max(0, self.total_budget - running_total),
        )
        return selected, allocation, rejected
