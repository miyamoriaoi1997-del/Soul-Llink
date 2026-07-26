from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Protocol

from .retrieval_provider import (
    AuthorityReference,
    Candidate,
    NotConfiguredProvider,
    RetrievalProvider,
    RetrievalRequest,
    RetrievalResult,
    RetrievalStatus,
)
from .store import EventStore
from .transcript_search import RecallEvidence, search_exact_evidence


@dataclass(frozen=True, slots=True)
class VerifiedResult:
    candidate: Candidate
    evidence: RecallEvidence


def verified_background_records(
    store: EventStore,
    request: RetrievalRequest,
    *,
    provider: RetrievalProvider | None = None,
) -> tuple[dict[str, object], ...]:
    """Return only authority-reopened evidence suitable for prompt background memory.

    The returned shape is deliberately observational: it carries the exact
    quote/offset and the commitments used by the reopen, but never turns a
    candidate into a fact before the reopen succeeds.
    """
    result = retrieve_with_authority(store, request, provider)
    if not isinstance(result, tuple):
        return ()
    records: list[dict[str, object]] = []
    for verified in result:
        reference = verified.candidate.reference
        evidence = verified.evidence
        records.append({
            "event_id": evidence.event_id,
            "chunk_id": evidence.chunk_id,
            "quote": evidence.quote,
            "start_char": evidence.start_char,
            "end_char": evidence.end_char,
            "payload_sha256": reference.payload_sha256,
            "chain_hash": reference.chain_hash,
            "chunk_sha256": reference.chunk_sha256,
            "source_created_at": evidence.source_created_at,
            "provider": verified.candidate.provider,
            "verified": evidence.verified,
        })
    return tuple(records)


def _event_matches_request_scope(event: dict[str, object], request: RetrievalRequest) -> bool:
    for field in ("session_id", "conversation_id", "platform", "persona_mode", "source"):
        expected = getattr(request, field)
        if expected is not None and event.get(field) != expected:
            return False
    if not request.include_sensitive and event.get("sensitivity") in {
        "private", "secret", "restricted",
    }:
        return False
    return True


def reopen_candidate(
    store: EventStore,
    candidate: Candidate,
    request: RetrievalRequest,
) -> RecallEvidence | None:
    """Reopen a candidate through the existing exact authority seam and request scope."""
    reference = candidate.reference
    if not candidate.quote:
        return None
    try:
        event = store.get_event(reference.event_id)
    except KeyError:
        return None
    if not _event_matches_request_scope(event, request):
        return None
    authority = store._conn.execute(
        """SELECT e.source_revision, e.payload_sha256, e.chain_hash,
                  c.chunk_id, c.chunk_ordinal, c.chunk_sha256, c.start_char, c.end_char
           FROM events e JOIN event_chunks c ON c.event_id = e.event_id
           WHERE e.event_id = ? AND c.chunk_id = ?""",
        (reference.event_id, reference.chunk_id),
    ).fetchone()
    if authority is None or any(
        (
            int(authority["source_revision"]) != reference.source_revision,
            str(authority["payload_sha256"]) != reference.payload_sha256,
            str(authority["chain_hash"]) != reference.chain_hash,
            int(authority["chunk_ordinal"]) != reference.chunk_ordinal,
            str(authority["chunk_sha256"]) != reference.chunk_sha256,
            reference.start_char < int(authority["start_char"]),
            reference.start_char >= int(authority["end_char"]),
        )
    ):
        return None
    hits = search_exact_evidence(store, candidate.quote, limit=100)
    for evidence in hits:
        if (
            evidence.event_id == reference.event_id
            and evidence.chunk_id == reference.chunk_id
            and evidence.start_char == reference.start_char
            and evidence.end_char == reference.end_char
            and evidence.payload_sha256 == reference.payload_sha256
        ):
            return evidence
    return None


def _lexical_evidence_span(content: str, query: str) -> tuple[int, int] | None:
    folded_content = content.casefold()
    positions: list[tuple[int, int]] = []
    cursor = 0
    for term in query.split():
        start = folded_content.find(term.casefold(), cursor)
        if start < 0:
            return None
        end = start + len(term)
        positions.append((start, end))
        cursor = end
    if not positions:
        return None
    return positions[0][0], positions[-1][1]


def _candidate_from_event(store: EventStore, event: dict[str, object], query: str) -> Candidate | None:
    event_id = int(event["event_id"])
    content = str(event["content"])
    span = _lexical_evidence_span(content, query)
    if span is None:
        return None
    start, end = span
    chunk = store._conn.execute(
        """SELECT chunk_id, chunk_ordinal, chunk_sha256
           FROM event_chunks
           WHERE event_id = ? AND start_char < ? AND end_char > ?
           ORDER BY start_char ASC, end_char DESC, chunk_ordinal ASC LIMIT 1""",
        (event_id, end, start),
    ).fetchone()
    if chunk is None:
        return None
    chunk_id = int(chunk["chunk_id"])
    chunk_ordinal = int(chunk["chunk_ordinal"])
    chunk_sha256 = str(chunk["chunk_sha256"])
    return Candidate(
        reference=AuthorityReference(
            event_id=event_id,
            chunk_id=chunk_id,
            chunk_ordinal=chunk_ordinal,
            source_revision=int(event.get("source_revision") or 1),
            start_char=start,
            end_char=end,
            payload_sha256=str(event.get("payload_sha256") or ""),
            chain_hash=str(event.get("chain_hash") or ""),
            chunk_sha256=chunk_sha256,
        ),
        score=1.0,
        provider="core.lexical",
        quote=content[start:end],
    )


def retrieve_lexical(store: EventStore, request: RetrievalRequest) -> RetrievalResult:
    if not request.query:
        return RetrievalResult.abstained("no_answer")
    rows = store.search_events(
        request.query,
        session_id=request.session_id,
        conversation_id=request.conversation_id,
        platform=request.platform,
        persona_mode=request.persona_mode,
        source=request.source,
        limit=request.limit,
        include_sensitive=request.include_sensitive,
    )
    candidates_list: list[Candidate] = []
    for row in rows:
        event = store.get_event(int(row["event_id"]))
        candidate = _candidate_from_event(store, event, request.query)
        if candidate is not None:
            candidates_list.append(candidate)
    candidates = tuple(candidates_list)
    return RetrievalResult.ok(candidates) if candidates else RetrievalResult.abstained("no_answer")


def retrieve_with_authority(
    store: EventStore,
    request: RetrievalRequest,
    provider: RetrievalProvider | None = None,
) -> tuple[VerifiedResult, ...] | RetrievalResult:
    core_result = retrieve_lexical(store, request)
    provider_result = provider.retrieve(request) if provider is not None else None
    candidates = core_result.candidates
    if provider_result is not None and provider_result.status is RetrievalStatus.OK:
        candidates += provider_result.candidates
    if not candidates:
        if provider_result is not None and provider_result.status is RetrievalStatus.UNAVAILABLE:
            return provider_result
        return core_result
    verified = tuple(
        VerifiedResult(candidate, evidence)
        for candidate in candidates
        if (evidence := reopen_candidate(store, candidate, request)) is not None
    )
    return verified


__all__ = [
    "AuthorityReference", "Candidate", "NotConfiguredProvider", "RetrievalProvider",
    "RetrievalRequest", "RetrievalResult", "RetrievalStatus", "VerifiedResult",
    "reopen_candidate", "retrieve_lexical", "retrieve_with_authority",
    "verified_background_records",
]