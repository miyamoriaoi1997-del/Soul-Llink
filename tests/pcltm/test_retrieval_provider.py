from __future__ import annotations

from dataclasses import FrozenInstanceError
import math
import os
from pathlib import Path
import subprocess
import sys

import pytest

from pcltm.retrieval_provider import (
    AuthorityReference,
    Candidate,
    NotConfiguredProvider,
    RetrievalRequest,
    RetrievalResult,
    RetrievalStatus,
)


def test_typed_provider_boundary_is_immutable_and_has_typed_statuses() -> None:
    reference = AuthorityReference(
        event_id=1,
        chunk_id=2,
        source_revision=1,
        start_char=0,
        end_char=4,
        payload_sha256="a" * 64,
        chain_hash="b" * 64,
        chunk_sha256="c" * 64,
    )
    request = RetrievalRequest(query="opaque query", limit=3, session_id="s")
    candidate = Candidate(reference=reference, score=0.5, provider="fake")
    result = RetrievalResult.ok((candidate,))

    assert request.limit == 3
    assert result.status is RetrievalStatus.OK
    assert result.candidates == (candidate,)
    with pytest.raises(FrozenInstanceError):
        request.limit = 4  # type: ignore[misc]
    with pytest.raises(FrozenInstanceError):
        candidate.score = 1.0  # type: ignore[misc]


def test_not_configured_provider_is_typed_unavailable() -> None:
    result = NotConfiguredProvider().retrieve(RetrievalRequest(query="opaque query"))

    assert result == RetrievalResult.unavailable("not_configured")
    assert result.status is RetrievalStatus.UNAVAILABLE
    assert result.reason == "not_configured"
    assert result.candidates == ()


def test_abstained_result_has_no_candidates() -> None:
    result = RetrievalResult.abstained("no_answer")

    assert result.status is RetrievalStatus.ABSTAINED
    assert result.reason == "no_answer"
    assert result.candidates == ()


def test_malformed_typed_boundaries_fail_closed() -> None:
    reference = AuthorityReference(
        event_id=1, chunk_id=2, source_revision=1, start_char=0, end_char=4,
        payload_sha256="a" * 64, chain_hash="b" * 64, chunk_sha256="c" * 64,
    )
    candidate = Candidate(reference=reference, score=0.5, provider="fake")

    with pytest.raises((TypeError, ValueError)):
        RetrievalRequest(query="opaque", include_sensitive="false")  # type: ignore[arg-type]
    with pytest.raises((TypeError, ValueError)):
        RetrievalResult(RetrievalStatus.UNAVAILABLE, (candidate,), "not_configured")


def test_hardening_rejects_invalid_limit_score_and_empty_ok() -> None:
    reference = AuthorityReference(
        event_id=1, chunk_id=2, source_revision=1, start_char=0, end_char=4,
        payload_sha256="a" * 64, chain_hash="b" * 64, chunk_sha256="c" * 64,
    )
    with pytest.raises((TypeError, ValueError)):
        RetrievalRequest(query="opaque", limit=0)
    with pytest.raises((TypeError, ValueError)):
        Candidate(reference=reference, score=math.nan, provider="fake")
    with pytest.raises((TypeError, ValueError)):
        RetrievalResult.ok(())


def _valid_reference(**overrides: object) -> AuthorityReference:
    values: dict[str, object] = {
        "event_id": 1,
        "chunk_id": 2,
        "source_revision": 1,
        "start_char": 0,
        "end_char": 4,
        "payload_sha256": "a" * 64,
        "chain_hash": "b" * 64,
        "chunk_sha256": "c" * 64,
        "chunk_ordinal": 0,
    }
    values.update(overrides)
    return AuthorityReference(**values)  # type: ignore[arg-type]


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("event_id", True), ("event_id", 0),
        ("chunk_id", True), ("chunk_id", 0),
        ("source_revision", True), ("source_revision", 0),
        ("chunk_ordinal", False), ("chunk_ordinal", -1),
        ("start_char", False), ("start_char", -1),
        ("end_char", True), ("end_char", 0),
    ],
)
def test_authority_reference_rejects_non_exact_or_out_of_range_integers(
    field: str, value: object,
) -> None:
    with pytest.raises((TypeError, ValueError)):
        _valid_reference(**{field: value})


@pytest.mark.parametrize("end_char", [0, 3])
def test_authority_reference_requires_non_empty_ordered_span(end_char: int) -> None:
    with pytest.raises(ValueError):
        _valid_reference(start_char=3, end_char=end_char)


@pytest.mark.parametrize("field", ["payload_sha256", "chain_hash", "chunk_sha256"])
@pytest.mark.parametrize("value", ["", "a" * 63, "g" * 64, b"a" * 64])
def test_authority_reference_rejects_malformed_commitment_hashes(
    field: str, value: object,
) -> None:
    with pytest.raises((TypeError, ValueError)):
        _valid_reference(**{field: value})


def test_candidate_requires_exact_authority_reference() -> None:
    with pytest.raises(TypeError):
        Candidate(reference=object(), score=0.5, provider="fake")  # type: ignore[arg-type]


def test_provider_modules_import_without_neural_packages() -> None:
    packages = Path(__file__).resolve().parents[2] / "packages"
    script = """
import builtins
real_import = builtins.__import__
def blocked(name, *args, **kwargs):
    if name.split('.', 1)[0] in {'torch', 'transformers', 'sentence_transformers'}:
        raise AssertionError(f'neural import attempted: {name}')
    return real_import(name, *args, **kwargs)
builtins.__import__ = blocked
import pcltm.retrieval_provider as provider
import pcltm.tiered_retrieval as core
assert provider.NotConfiguredProvider().retrieve(provider.RetrievalRequest('opaque')).reason == 'not_configured'
print(provider.__file__)
print(core.__file__)
"""
    env = os.environ.copy()
    env["PYTHONPATH"] = str(packages)
    completed = subprocess.run(
        [sys.executable, "-c", script],
        cwd=Path(__file__).resolve().parents[2],
        env=env,
        capture_output=True,
        text=True,
    )

    assert completed.returncode == 0, completed.stderr
    assert str(packages.resolve()).lower() in completed.stdout.lower()