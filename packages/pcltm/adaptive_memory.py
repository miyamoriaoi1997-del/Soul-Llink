from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Iterable

from .retrieval_provider import Candidate, ChannelScoreEvidence


@dataclass(frozen=True, slots=True)
class RankFusionConfig:
    """Strict configuration for SoulLink's deterministic rank fusion."""

    rank_base: float = 60.0
    channel_weights: tuple[tuple[str, float], ...] = ()

    def __post_init__(self) -> None:
        if type(self.rank_base) not in (int, float) or isinstance(self.rank_base, bool):
            raise TypeError("rank_base must be numeric")
        if not math.isfinite(float(self.rank_base)) or self.rank_base <= 0:
            raise ValueError("rank_base must be finite and positive")
        if type(self.channel_weights) is not tuple:
            raise TypeError("channel_weights must be a tuple")
        seen: set[str] = set()
        for channel, weight in self.channel_weights:
            if type(channel) is not str or not channel or channel in seen:
                raise ValueError("channel names must be unique non-empty strings")
            if type(weight) not in (int, float) or isinstance(weight, bool):
                raise TypeError("channel weights must be numeric")
            if not math.isfinite(float(weight)) or weight < 0:
                raise ValueError("channel weights must be finite and non-negative")
            seen.add(channel)

    def weight_for(self, channel: str) -> float:
        for configured_channel, weight in self.channel_weights:
            if configured_channel == channel:
                return float(weight)
        return 1.0


def _reference_key(candidate: Candidate) -> tuple[object, ...]:
    reference = candidate.reference
    return (
        reference.event_id,
        reference.chunk_id,
        reference.source_revision,
        reference.start_char,
        reference.end_char,
        reference.payload_sha256,
        reference.chain_hash,
        reference.chunk_sha256,
        reference.chunk_ordinal,
        reference.source_hash,
        reference.source_created_at,
    )


def _evidence_key(evidence: ChannelScoreEvidence) -> tuple[str, int, float]:
    return (evidence.channel, evidence.channel_rank, float(evidence.raw_score))


def fuse_candidates(
    candidates: Iterable[Candidate],
    config: RankFusionConfig,
) -> tuple[Candidate, ...]:
    """Fuse candidate channels by rank, preserving raw score evidence.

    Candidate identity is the complete authority commitment tuple.  Raw
    channel scores are never combined and never candidate-max normalized.
    """
    grouped: dict[tuple[object, ...], list[Candidate]] = {}
    for candidate in candidates:
        if type(candidate) is not Candidate:
            raise TypeError("candidates must contain Candidate values")
        grouped.setdefault(_reference_key(candidate), []).append(candidate)

    fused: list[Candidate] = []
    for reference_key, group in grouped.items():
        evidence: list[ChannelScoreEvidence] = []
        for candidate in group:
            evidence.extend(candidate.channel_evidence)
            if not candidate.channel_evidence:
                evidence.append(
                    ChannelScoreEvidence(candidate.provider, 1, float(candidate.score))
                )
        unique_evidence = tuple(sorted(set(evidence), key=_evidence_key))
        ranked_observations = {
            (item.channel, item.channel_rank) for item in unique_evidence
        }
        rank_score = sum(
            config.weight_for(channel) / (config.rank_base + channel_rank)
            for channel, channel_rank in ranked_observations
        )
        representative = min(group, key=lambda item: (item.provider, item.quote))
        fused.append(
            Candidate(
                reference=representative.reference,
                score=rank_score,
                provider=(
                    "soullink.rank_fusion"
                    if len({item.channel for item in unique_evidence}) > 1
                    else representative.provider
                ),
                quote=representative.quote,
                channel_evidence=unique_evidence,
            )
        )
    return tuple(
        sorted(
            fused,
            key=lambda item: (-float(item.score), _reference_key(item)),
        )
    )


__all__ = ["RankFusionConfig", "fuse_candidates"]
