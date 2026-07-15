"""Extract episodic memory records from raw-derived session segments."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Protocol

from .episodic_store import EpisodicMemory


class SegmentLike(Protocol):
    segment_id: int
    time_range: tuple[int, int]
    raw_message_refs: tuple[str, ...]
    local_summary: str
    decisions: tuple[str, ...]
    commitments: tuple[str, ...]
    unresolved_items: tuple[str, ...]
    emotional_delta: str
    memory_candidates: tuple[str, ...]
    verification_notes: tuple[str, ...]
    completed_items: tuple[str, ...]
    paused_items: tuple[str, ...]
    revoked_items: tuple[str, ...]


_LOW_CONFIDENCE_MARKERS = (
    "可能",
    "也许",
    "大概",
    "不确定",
    "猜测",
    "推测",
    "maybe",
    "perhaps",
    "uncertain",
    "guess",
)
_EMOTIONAL_MARKERS = (
    "生气",
    "难过",
    "焦虑",
    "担心",
    "情绪",
    "受伤",
    "不安",
    "angry",
    "sad",
    "anxious",
    "worried",
    "hurt",
)
_PREFERENCE_MARKERS = (
    "喜欢",
    "不喜欢",
    "偏好",
    "讨厌",
    "希望",
    "prefer",
    "preference",
    "likes",
    "dislikes",
)
_IMPORTANT_MARKERS = (
    "提交",
    "完成",
    "验收",
    "决定",
    "风险",
    "阻塞",
    "commit",
    "passed",
    "decision",
    "blocked",
)


def _contains_any(text: str, markers: Iterable[str]) -> bool:
    lowered = text.lower()
    return any(marker.lower() in lowered for marker in markers)


def _score_importance(text: str, *, base: float) -> float:
    score = base
    if _contains_any(text, _IMPORTANT_MARKERS):
        score += 0.25
    if len(text) > 160:
        score += 0.1
    return max(0.0, min(1.0, score))


def _score_emotional_salience(text: str) -> float:
    if _contains_any(text, _EMOTIONAL_MARKERS):
        return 0.75
    return 0.0


def _score_confidence(text: str) -> float:
    if _contains_any(text, _LOW_CONFIDENCE_MARKERS):
        return 0.45
    return 0.9


def _fact_promotion_blockers(text: str, *, single_segment: bool = True) -> tuple[str, ...]:
    blockers = ["episodic_record_only"]
    if _contains_any(text, _LOW_CONFIDENCE_MARKERS):
        blockers.append("low_confidence_fragment")
    if _contains_any(text, _EMOTIONAL_MARKERS):
        blockers.append("emotional_fragment_not_stable_personality")
    if single_segment and _contains_any(text, _PREFERENCE_MARKERS):
        blockers.append("single_event_not_permanent_preference")
    return tuple(dict.fromkeys(blockers))


def _entities_from_text(text: str, participants: tuple[str, ...]) -> tuple[str, ...]:
    entities: list[str] = list(participants)
    for token in ("用户", "user", "assistant", "PCLTM", "Hermes", "MemFS", "session"):
        if token.lower() in text.lower() and token not in entities:
            entities.append(token)
    return tuple(entities)


@dataclass(frozen=True)
class EpisodeExtractor:
    """Deterministic extractor for phase-3 episodic memory.

    The extractor only emits evidence-backed events. It never turns a segment
    into a durable user preference or stable personality conclusion.
    """

    default_participants: tuple[str, ...] = ("user", "assistant")

    def extract_from_segment(
        self,
        segment: SegmentLike,
        *,
        source_session: str,
        timestamp: str | None = None,
    ) -> tuple[EpisodicMemory, ...]:
        raw_refs = tuple(segment.raw_message_refs)
        if not raw_refs:
            return ()

        events: list[EpisodicMemory] = []
        segment_tag = f"segment:{segment.segment_id}"

        if segment.local_summary:
            events.append(
                self._build_event(
                    source_session=source_session,
                    raw_refs=raw_refs,
                    summary=segment.local_summary,
                    event_type="segment_summary",
                    tags=(segment_tag, "raw_segment"),
                    timestamp=timestamp,
                    importance_base=0.35,
                    continuity_relevance=0.55,
                )
            )

        events.extend(
            self._build_event(
                source_session=source_session,
                raw_refs=raw_refs,
                summary=item,
                event_type="decision",
                tags=(segment_tag, "decision"),
                timestamp=timestamp,
                importance_base=0.7,
                continuity_relevance=0.75,
            )
            for item in segment.decisions
        )
        events.extend(
            self._build_event(
                source_session=source_session,
                raw_refs=raw_refs,
                summary=item,
                event_type="commitment",
                tags=(segment_tag, "commitment"),
                timestamp=timestamp,
                importance_base=0.65,
                continuity_relevance=0.8,
            )
            for item in segment.commitments
        )
        events.extend(
            self._build_event(
                source_session=source_session,
                raw_refs=raw_refs,
                summary=item,
                event_type="unresolved_item",
                tags=(segment_tag, "unresolved"),
                timestamp=timestamp,
                importance_base=0.6,
                continuity_relevance=0.9,
            )
            for item in segment.unresolved_items
        )
        if segment.emotional_delta:
            events.append(
                self._build_event(
                    source_session=source_session,
                    raw_refs=raw_refs,
                    summary=segment.emotional_delta,
                    event_type="emotional_observation",
                    tags=(segment_tag, "emotion", "anti_pollution"),
                    timestamp=timestamp,
                    importance_base=0.4,
                    continuity_relevance=0.45,
                )
            )
        events.extend(
            self._build_event(
                source_session=source_session,
                raw_refs=raw_refs,
                summary=item,
                event_type="memory_candidate_observation",
                tags=(segment_tag, "memory_candidate", "anti_pollution"),
                timestamp=timestamp,
                importance_base=0.45,
                continuity_relevance=0.5,
            )
            for item in segment.memory_candidates
        )
        return tuple(events)

    def _build_event(
        self,
        *,
        source_session: str,
        raw_refs: tuple[str, ...],
        summary: str,
        event_type: str,
        tags: tuple[str, ...],
        timestamp: str | None,
        importance_base: float,
        continuity_relevance: float,
    ) -> EpisodicMemory:
        confidence = _score_confidence(summary)
        blockers = _fact_promotion_blockers(summary)
        return EpisodicMemory.create(
            source_session=source_session,
            raw_refs=raw_refs,
            event_summary=summary,
            participants=self.default_participants,
            event_type=event_type,
            timestamp=timestamp,
            importance_score=_score_importance(summary, base=importance_base),
            emotional_salience=_score_emotional_salience(summary),
            continuity_relevance=continuity_relevance,
            entities=_entities_from_text(summary, self.default_participants),
            tags=tags,
            confidence_score=confidence,
            fact_promotion_allowed=False,
            fact_promotion_blockers=blockers,
        )
