"""Candidate retrieval for PCLTM episodic memory.

Retrieval produces candidates for later injection arbitration. It does not inject
records directly and does not promote events into stable facts.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import UTC, datetime
import math
import re
from typing import Iterable

from .episodic_store import EpisodicMemory, EpisodicStore


_TOKEN_RE = re.compile(r"[\w\u4e00-\u9fff]+", re.UNICODE)


def _tokens(text: str) -> set[str]:
    return {token.lower() for token in _TOKEN_RE.findall(text or "") if token.strip()}


def _parse_timestamp(value: str) -> datetime | None:
    try:
        parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError:
        return None
    if parsed.tzinfo is None:
        return parsed.replace(tzinfo=UTC)
    return parsed.astimezone(UTC)


def _recency_score(timestamp: str, *, now: datetime) -> float:
    parsed = _parse_timestamp(timestamp)
    if parsed is None:
        return 0.0
    age_seconds = max(0.0, (now - parsed).total_seconds())
    age_days = age_seconds / 86400.0
    return math.exp(-age_days / 30.0)


def _jaccard(left: set[str], right: set[str]) -> float:
    if not left or not right:
        return 0.0
    return len(left & right) / len(left | right)


@dataclass(frozen=True)
class EpisodicRetrievalResult:
    memory: EpisodicMemory
    score: float
    relevance: float
    recency: float
    importance: float
    continuity_relevance: float
    current_task_relevance: float

    def to_candidate_dict(self) -> dict[str, object]:
        return {
            "event_id": self.memory.event_id,
            "score": self.score,
            "relevance": self.relevance,
            "recency": self.recency,
            "importance": self.importance,
            "continuity_relevance": self.continuity_relevance,
            "current_task_relevance": self.current_task_relevance,
            "source_session": self.memory.source_session,
            "raw_refs": list(self.memory.raw_refs),
            "event_summary": self.memory.event_summary,
            "event_type": self.memory.event_type,
            "injection_note": "candidate_only_for_injection_arbitration",
        }


@dataclass(frozen=True)
class EpisodicRetriever:
    store: EpisodicStore

    def retrieve(
        self,
        query: str,
        *,
        current_task: str = "",
        limit: int = 5,
        now: datetime | None = None,
        include_superseded: bool = False,
    ) -> tuple[EpisodicRetrievalResult, ...]:
        query_tokens = _tokens(query)
        task_tokens = _tokens(current_task)
        current_time = now or datetime.now(UTC)
        results: list[EpisodicRetrievalResult] = []

        for memory in self.store.list_events(include_superseded=include_superseded):
            memory_tokens = _tokens(
                " ".join(
                    (
                        memory.event_summary,
                        memory.event_type,
                        " ".join(memory.entities),
                        " ".join(memory.tags),
                    )
                )
            )
            relevance = _jaccard(query_tokens, memory_tokens)
            task_relevance = _jaccard(task_tokens, memory_tokens) if task_tokens else 0.0
            combined_relevance = max(relevance, task_relevance)
            recency = _recency_score(memory.timestamp, now=current_time)
            importance = memory.importance_score
            continuity = max(memory.continuity_relevance, task_relevance)
            score = (
                combined_relevance * 0.45
                + recency * 0.20
                + importance * 0.20
                + continuity * 0.15
            )
            if score <= 0.0:
                continue
            results.append(
                EpisodicRetrievalResult(
                    memory=memory,
                    score=round(score, 6),
                    relevance=round(combined_relevance, 6),
                    recency=round(recency, 6),
                    importance=round(importance, 6),
                    continuity_relevance=round(continuity, 6),
                    current_task_relevance=round(task_relevance, 6),
                )
            )

        results.sort(key=lambda result: (result.score, result.memory.timestamp), reverse=True)
        return tuple(results[: max(0, int(limit))])

    def answer_what_happened(
        self,
        query: str,
        *,
        current_task: str = "",
        limit: int = 3,
        now: datetime | None = None,
    ) -> str:
        results = self.retrieve(query, current_task=current_task, limit=limit, now=now)
        if not results:
            return "没有找到可追溯的 episodic memory 事件。"
        lines = ["之前这件事的可追溯事件："]
        for result in results:
            memory = result.memory
            refs = ", ".join(memory.raw_refs)
            lines.append(
                f"- [{memory.event_type}] {memory.event_summary} "
                f"(source_session={memory.source_session}; raw_refs={refs}; score={result.score:.3f})"
            )
        lines.append("注意：这些是事件候选，不是长期事实或人格结论。")
        return "\n".join(lines)
