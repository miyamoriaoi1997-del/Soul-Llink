from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from math import floor
from typing import Iterable, Mapping, Sequence

from .memory_object import InjectionPolicy, MemoryObject
from .memory_selection import SelectionDecision


class ContextBudgetBucket(str, Enum):
    """Typed prompt-budget buckets for PCLTM active-frame observability."""

    ACTIVE_FRAME = "active_frame"
    PINNED_MEMORY = "pinned_memory"
    SELECTED_MEMORY = "selected_memory"
    TRANSCRIPT = "transcript"
    TOOL_EVIDENCE = "tool_evidence"
    REFERENCE = "reference"
    RESERVE = "reserve"


@dataclass(frozen=True)
class ContextBudgetLine:
    bucket: ContextBudgetBucket
    token_count: int
    token_limit: int
    truncated: bool = False

    @property
    def remaining(self) -> int:
        return max(self.token_limit - self.token_count, 0)

    @property
    def utilization(self) -> float:
        if self.token_limit <= 0:
            return 1.0 if self.token_count > 0 else 0.0
        return self.token_count / self.token_limit

    def to_dict(self) -> dict[str, object]:
        return {
            "bucket": self.bucket.value,
            "token_count": self.token_count,
            "token_limit": self.token_limit,
            "remaining": self.remaining,
            "utilization": self.utilization,
            "truncated": self.truncated,
        }


@dataclass(frozen=True)
class ContextBudgetReport:
    context_window: int
    lines: tuple[ContextBudgetLine, ...]
    estimated_total_tokens: int
    over_budget_tokens: int

    @property
    def within_budget(self) -> bool:
        return self.over_budget_tokens == 0

    def line_for(self, bucket: ContextBudgetBucket) -> ContextBudgetLine:
        for line in self.lines:
            if line.bucket is bucket:
                return line
        raise KeyError(bucket.value)

    def to_dict(self) -> dict[str, object]:
        return {
            "context_window": self.context_window,
            "estimated_total_tokens": self.estimated_total_tokens,
            "over_budget_tokens": self.over_budget_tokens,
            "within_budget": self.within_budget,
            "lines": [line.to_dict() for line in self.lines],
        }


DEFAULT_BUDGET_RATIOS: Mapping[ContextBudgetBucket, float] = {
    ContextBudgetBucket.ACTIVE_FRAME: 0.18,
    ContextBudgetBucket.PINNED_MEMORY: 0.10,
    ContextBudgetBucket.SELECTED_MEMORY: 0.18,
    ContextBudgetBucket.TRANSCRIPT: 0.36,
    ContextBudgetBucket.TOOL_EVIDENCE: 0.08,
    ContextBudgetBucket.REFERENCE: 0.04,
    ContextBudgetBucket.RESERVE: 0.06,
}


def estimate_context_budget(
    *,
    context_window: int,
    active_frame: str = "",
    pinned_memories: Sequence[MemoryObject] = (),
    selection_decisions: Sequence[SelectionDecision] = (),
    memory_objects: Sequence[MemoryObject] = (),
    transcript_messages: Sequence[Mapping[str, object]] = (),
    tool_evidence: Sequence[Mapping[str, object]] = (),
    reference_items: Sequence[object] = (),
    ratios: Mapping[ContextBudgetBucket, float] | None = None,
) -> ContextBudgetReport:
    """Estimate a typed PCLTM context-window budget without changing injection.

    This mirrors Letta's useful separation of context-window accounting from
    runtime mutation: every bucket is observable, but the report is read-only.
    """

    if context_window <= 0:
        raise ValueError("context_window must be positive")

    normalized_ratios = _normalize_ratios(ratios or DEFAULT_BUDGET_RATIOS)
    selected_keys = {
        decision.canonical_key for decision in selection_decisions if decision.selected
    }
    memory_by_key = {memory.canonical_key: memory for memory in memory_objects}
    selected_memories = [memory_by_key[key] for key in selected_keys if key in memory_by_key]

    counts = {
        ContextBudgetBucket.ACTIVE_FRAME: estimate_tokens(active_frame),
        ContextBudgetBucket.PINNED_MEMORY: sum(estimate_tokens(memory.content) for memory in pinned_memories),
        ContextBudgetBucket.SELECTED_MEMORY: sum(estimate_tokens(memory.content) for memory in selected_memories),
        ContextBudgetBucket.TRANSCRIPT: sum(_message_tokens(message) for message in transcript_messages),
        ContextBudgetBucket.TOOL_EVIDENCE: sum(_message_tokens(message) for message in tool_evidence),
        ContextBudgetBucket.REFERENCE: sum(estimate_tokens(item) for item in reference_items),
    }
    counted_total = sum(counts.values())
    counts[ContextBudgetBucket.RESERVE] = max(context_window - counted_total, 0)

    lines = tuple(
        ContextBudgetLine(
            bucket=bucket,
            token_count=counts.get(bucket, 0),
            token_limit=_bucket_limit(context_window, bucket, normalized_ratios),
            truncated=_bucket_truncated(
                bucket,
                counts.get(bucket, 0),
                _bucket_limit(context_window, bucket, normalized_ratios),
            ),
        )
        for bucket in ContextBudgetBucket
    )
    estimated_total = sum(line.token_count for line in lines if line.bucket is not ContextBudgetBucket.RESERVE)
    return ContextBudgetReport(
        context_window=context_window,
        lines=lines,
        estimated_total_tokens=estimated_total,
        over_budget_tokens=max(estimated_total - context_window, 0),
    )


def estimate_tokens(value: object) -> int:
    """Cheap deterministic token estimate for budget probes and tests."""

    if value is None:
        return 0
    text = str(value)
    if not text:
        return 0
    return max(1, (len(text) + 3) // 4)


def _message_tokens(message: Mapping[str, object]) -> int:
    role_tokens = estimate_tokens(message.get("role", ""))
    content = message.get("content", "")
    tool_tokens = estimate_tokens(message.get("tool_call_id", ""))
    return role_tokens + estimate_tokens(content) + tool_tokens


def _normalize_ratios(ratios: Mapping[ContextBudgetBucket, float]) -> Mapping[ContextBudgetBucket, float]:
    normalized: dict[ContextBudgetBucket, float] = {}
    for bucket in ContextBudgetBucket:
        raw_value = ratios.get(bucket, 0.0)
        normalized[bucket] = max(float(raw_value), 0.0)
    total = sum(normalized.values())
    if total <= 0:
        raise ValueError("at least one budget ratio must be positive")
    return {bucket: value / total for bucket, value in normalized.items()}


def _bucket_truncated(bucket: ContextBudgetBucket, token_count: int, token_limit: int) -> bool:
    if bucket is ContextBudgetBucket.RESERVE:
        return False
    return token_count > token_limit


def _bucket_limit(
    context_window: int,
    bucket: ContextBudgetBucket,
    ratios: Mapping[ContextBudgetBucket, float],
) -> int:
    if bucket is ContextBudgetBucket.RESERVE:
        return max(context_window - sum(
            floor(context_window * ratio)
            for ratio_bucket, ratio in ratios.items()
            if ratio_bucket is not ContextBudgetBucket.RESERVE
        ), 0)
    return floor(context_window * ratios[bucket])
