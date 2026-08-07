from __future__ import annotations

import pytest

from pcltm.adaptive_memory import (
    RankFusionConfig,
    fuse_candidates,
)
from pcltm.retrieval_provider import AuthorityReference, Candidate, ChannelScoreEvidence


def _reference(event_id: int) -> AuthorityReference:
    return AuthorityReference(
        event_id=event_id,
        chunk_id=event_id + 10,
        source_revision=1,
        start_char=0,
        end_char=5,
        payload_sha256="a" * 64,
        chain_hash="b" * 64,
        chunk_sha256="c" * 64,
        chunk_ordinal=0,
    )


def _candidate(event_id: int, channel: str, rank: int, raw_score: float) -> Candidate:
    evidence = ChannelScoreEvidence(channel, rank, raw_score)
    return Candidate(
        reference=_reference(event_id),
        score=raw_score,
        provider=channel,
        quote=f"quote-{event_id}",
        channel_evidence=(evidence,),
    )


def test_rank_fusion_uses_rank_evidence_and_merges_all_channel_evidence() -> None:
    candidates = (
        _candidate(1, "lexical", 1, 0.01),
        _candidate(1, "provider", 4, 999.0),
        _candidate(2, "lexical", 2, 100.0),
    )

    result = fuse_candidates(candidates, RankFusionConfig())

    assert [item.reference.event_id for item in result] == [1, 2]
    assert result[0].channel_evidence == (
        ChannelScoreEvidence("lexical", 1, 0.01),
        ChannelScoreEvidence("provider", 4, 999.0),
    )
    assert result[0].score != pytest.approx(0.01 + 999.0)


def test_rank_fusion_has_stable_reference_tiebreak_and_empty_input() -> None:
    first = _candidate(1, "a", 1, 0.0)
    second = _candidate(2, "a", 1, 0.0)

    assert [item.reference.event_id for item in fuse_candidates((second, first), RankFusionConfig())] == [1, 2]
    assert fuse_candidates((), RankFusionConfig()) == ()


def test_same_channel_rank_is_scored_once_while_all_raw_evidence_is_retained() -> None:
    first = _candidate(1, "provider", 1, 0.1)
    second = _candidate(1, "provider", 1, 0.2)

    result = fuse_candidates((first, second), RankFusionConfig())

    assert result[0].score == pytest.approx(1 / 61)
    assert result[0].channel_evidence == (
        ChannelScoreEvidence("provider", 1, 0.1),
        ChannelScoreEvidence("provider", 1, 0.2),
    )


@pytest.mark.parametrize(
    "kwargs",
    [
        {"rank_base": 0},
        {"rank_base": -1.0},
        {"rank_base": float("nan")},
        {"rank_base": True},
        {"channel_weights": (("x", -0.1),)},
        {"channel_weights": (("x", float("inf")),)},
    ],
)
def test_rank_fusion_configuration_is_strictly_bounded(kwargs: dict[str, object]) -> None:
    with pytest.raises((TypeError, ValueError)):
        RankFusionConfig(**kwargs)  # type: ignore[arg-type]
