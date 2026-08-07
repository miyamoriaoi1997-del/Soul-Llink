from __future__ import annotations

from dataclasses import FrozenInstanceError
import math

import pytest

from pcltm.retrieval_provider import (
    AuthorityReference,
    Candidate,
    ChannelScoreEvidence,
    RetrievalRequest,
)


def _reference() -> AuthorityReference:
    return AuthorityReference(
        event_id=1,
        chunk_id=2,
        source_revision=1,
        start_char=0,
        end_char=5,
        payload_sha256="a" * 64,
        chain_hash="b" * 64,
        chunk_sha256="c" * 64,
        chunk_ordinal=0,
    )


def test_channel_score_evidence_is_immutable_and_preserves_raw_finite_score() -> None:
    evidence = ChannelScoreEvidence(channel="core.lexical", channel_rank=3, raw_score=0.25)

    assert evidence.channel == "core.lexical"
    assert evidence.channel_rank == 3
    assert evidence.raw_score == 0.25
    with pytest.raises(FrozenInstanceError):
        evidence.channel_rank = 4  # type: ignore[misc]


@pytest.mark.parametrize("value", [True, False, math.nan, math.inf, -math.inf])
def test_channel_score_evidence_rejects_bool_and_nonfinite_scores(value: object) -> None:
    with pytest.raises((TypeError, ValueError)):
        ChannelScoreEvidence(channel="core.lexical", channel_rank=1, raw_score=value)  # type: ignore[arg-type]


def test_candidate_exposes_its_channel_evidence_without_losing_authority_reference() -> None:
    evidence = ChannelScoreEvidence(channel="provider.alpha", channel_rank=2, raw_score=7.5)
    candidate = Candidate(
        reference=_reference(),
        score=0.75,
        provider="provider.alpha",
        quote="hello",
        channel_evidence=(evidence,),
    )

    assert candidate.channel_evidence == (evidence,)
    assert candidate.reference == _reference()


def test_candidate_rejects_malformed_channel_evidence() -> None:
    with pytest.raises((TypeError, ValueError)):
        Candidate(
            reference=_reference(),
            score=0.75,
            provider="provider.alpha",
            channel_evidence=(object(),),  # type: ignore[arg-type]
        )


@pytest.mark.parametrize("field,value", [
    ("source_hash", []),
    ("source_hash", True),
    ("source_created_at", {}),
    ("source_created_at", math.nan),
    ("source_hash", ""),
])
def test_authority_reference_rejects_malformed_optional_commitments(
    field: str, value: object
) -> None:
    values = {item: getattr(_reference(), item) for item in _reference().__dataclass_fields__}
    values[field] = value
    with pytest.raises((TypeError, ValueError)):
        AuthorityReference(**values)  # type: ignore[arg-type]


@pytest.mark.parametrize("field,value", [
    ("session_id", []),
    ("conversation_id", {}),
    ("platform", 1),
    ("persona_mode", True),
    ("source", b"chat"),
])
def test_retrieval_request_rejects_malformed_scope_fields(field: str, value: object) -> None:
    with pytest.raises(TypeError):
        RetrievalRequest("query", **{field: value})  # type: ignore[arg-type]
