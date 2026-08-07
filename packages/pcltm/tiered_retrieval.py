from __future__ import annotations

from dataclasses import dataclass

from .adaptive_memory import RankFusionConfig, fuse_candidates
from .retrieval_provider import (
    AuthorityReference,
    Candidate,
    ChannelScoreEvidence,
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
            "source_hash": reference.source_hash,
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


def _reopen_candidate(
    store: EventStore,
    candidate: Candidate,
    request: RetrievalRequest,
) -> tuple[RecallEvidence | None, str | None]:
    reference = candidate.reference
    if not candidate.quote:
        return None, "authority_reopen_failed"
    try:
        event = store.get_event(reference.event_id)
    except KeyError:
        return None, "authority_reopen_failed"
    if not _event_matches_request_scope(event, request):
        return None, "scope_filtered"
    authority = store._conn.execute(
        """SELECT e.source_revision, e.payload_sha256, e.chain_hash,
                  e.source_created_at, r.source_hash,
                  c.chunk_id, c.chunk_ordinal, c.chunk_sha256, c.start_char, c.end_char
           FROM events e
           JOIN event_chunks c ON c.event_id = e.event_id
           LEFT JOIN event_revisions r
             ON r.event_id = e.event_id AND r.source_revision = e.source_revision
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
            authority["source_hash"] != reference.source_hash,
            authority["source_created_at"] != reference.source_created_at,
            reference.start_char < int(authority["start_char"]),
            reference.start_char >= int(authority["end_char"]),
        )
    ):
        return None, "authority_reopen_failed"
    hits = search_exact_evidence(store, candidate.quote, limit=100)
    for evidence in hits:
        if (
            evidence.event_id == reference.event_id
            and evidence.chunk_id == reference.chunk_id
            and evidence.start_char == reference.start_char
            and evidence.end_char == reference.end_char
            and evidence.payload_sha256 == reference.payload_sha256
        ):
            return evidence, None
    return None, "authority_reopen_failed"


def reopen_candidate(
    store: EventStore,
    candidate: Candidate,
    request: RetrievalRequest,
) -> RecallEvidence | None:
    """Reopen a candidate through the exact authority seam and request scope."""
    evidence, _reason = _reopen_candidate(store, candidate, request)
    return evidence


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
    revision = store._conn.execute(
        """SELECT source_hash FROM event_revisions
           WHERE event_id = ? AND source_revision = ?""",
        (event_id, int(event.get("source_revision") or 1)),
    ).fetchone()
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
            source_hash=(str(revision["source_hash"]) if revision is not None else None),
            source_created_at=(
                str(event["source_created_at"])
                if event.get("source_created_at") is not None
                else None
            ),
        ),
        score=1.0,
        provider="core.lexical",
        quote=content[start:end],
    )


def retrieve_lexical(store: EventStore, request: RetrievalRequest) -> RetrievalResult:
    if not request.query.strip():
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
    *,
    fusion_config: RankFusionConfig | None = None,
) -> tuple[VerifiedResult, ...] | RetrievalResult:
    """Retrieve candidates, fuse channels, and reopen every fused reference."""
    if not request.query.strip():
        return RetrievalResult.abstained("no_answer")
    core_result = retrieve_lexical(store, request)
    provider_result = provider.retrieve(request) if provider is not None else None
    core_candidates = tuple(
        Candidate(
            reference=item.reference,
            score=item.score,
            provider=item.provider,
            quote=item.quote,
            channel_evidence=(
                item.channel_evidence
                or (ChannelScoreEvidence(item.provider, index + 1, item.score),)
            ),
        )
        for index, item in enumerate(core_result.candidates)
    )
    candidates = core_candidates
    if provider_result is not None and provider_result.status is RetrievalStatus.OK:
        provider_candidates = tuple(
            item
            if item.channel_evidence
            else Candidate(
                reference=item.reference,
                score=item.score,
                provider=item.provider,
                quote=item.quote,
                channel_evidence=(
                    ChannelScoreEvidence(item.provider, index + 1, item.score),
                ),
            )
            for index, item in enumerate(provider_result.candidates)
        )
        candidates += provider_candidates
    if not candidates:
        if provider_result is not None and provider_result.status is RetrievalStatus.UNAVAILABLE:
            return provider_result
        return core_result
    typed_channel_contract = any(item.channel_evidence for item in candidates)
    fused = fuse_candidates(candidates, fusion_config or RankFusionConfig())
    verified_list: list[VerifiedResult] = []
    reopen_failed = False
    for candidate in fused:
        evidence, reason = _reopen_candidate(store, candidate, request)
        if evidence is None:
            reopen_failed = reopen_failed or reason == "authority_reopen_failed"
            continue
        verified_list.append(VerifiedResult(candidate, evidence))
    if reopen_failed and typed_channel_contract:
        return RetrievalResult.unavailable("authority_reopen_failed")
    verified = tuple(verified_list)
    if not verified and typed_channel_contract:
        return RetrievalResult.abstained("policy_filtered")
    return verified[: request.limit]


__all__ = [
    "AuthorityReference", "Candidate", "NotConfiguredProvider", "RetrievalProvider",
    "RetrievalRequest", "RetrievalResult", "RetrievalStatus", "VerifiedResult",
    "reopen_candidate", "retrieve_lexical", "retrieve_with_authority",
    "verified_background_records",
]
