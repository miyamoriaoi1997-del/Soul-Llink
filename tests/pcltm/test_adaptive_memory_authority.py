from __future__ import annotations

from dataclasses import replace
from pathlib import Path

from pcltm.adaptive_memory import RankFusionConfig
from pcltm.retrieval_provider import (
    AuthorityReference,
    Candidate,
    ChannelScoreEvidence,
    RetrievalRequest,
    RetrievalResult,
    RetrievalStatus,
)
from pcltm.projections.transcript_chunks import TranscriptChunkProjector
from pcltm.store import EventStore
from pcltm.tiered_retrieval import retrieve_with_authority, verified_background_records


def _candidate_for(store: EventStore, event_id: int, provider: str, quote: str) -> Candidate:
    event = store.get_event(event_id)
    chunk = store._conn.execute(
        "SELECT * FROM event_chunks WHERE event_id=? ORDER BY chunk_ordinal LIMIT 1", (event_id,)
    ).fetchone()
    return Candidate(
        reference=AuthorityReference(
            event_id=event_id,
            chunk_id=int(chunk["chunk_id"]),
            chunk_ordinal=int(chunk["chunk_ordinal"]),
            source_revision=int(event["source_revision"]),
            start_char=0,
            end_char=len(quote),
            payload_sha256=event["payload_sha256"],
            chain_hash=event["chain_hash"],
            chunk_sha256=chunk["chunk_sha256"],
        ),
        score=0.9,
        provider=provider,
        quote=quote,
    )


def _store(tmp_path: Path) -> EventStore:
    store = EventStore(tmp_path / "authority.db")
    store.append_event(
        session_id="s", conversation_id="c", platform="desktop", source="chat", role="user",
        content="unified authority needle", category="raw_conversation", subcategory="user",
        inject_policy="retrieve_only",
    )
    TranscriptChunkProjector(store, worker_id="test").run_once(
        now="2026-07-17T05:01:00Z", lease_until="2026-07-17T05:02:00Z"
    )
    return store


def test_unified_entrypoint_fuses_core_and_provider_and_reopens_once(tmp_path: Path) -> None:
    store = _store(tmp_path)
    try:
        core = _candidate_for(store, 1, "core.lexical", "unified authority needle")
        provider = replace(core, provider="optional.provider")

        class Provider:
            def retrieve(self, request: RetrievalRequest) -> RetrievalResult:
                assert request.session_id == "s"
                return RetrievalResult.ok((provider,))

        result = retrieve_with_authority(
            store, RetrievalRequest("unified authority needle", session_id="s"), Provider(),
            fusion_config=RankFusionConfig(),
        )
    finally:
        store.close()

    assert len(result) == 1
    assert result[0].candidate.provider == "soullink.rank_fusion"
    assert {item.channel for item in result[0].candidate.channel_evidence} == {
        "core.lexical", "optional.provider",
    }
    assert result[0].evidence.verified is True


def test_unavailable_provider_does_not_replace_valid_core_result(tmp_path: Path) -> None:
    from pcltm.retrieval_provider import NotConfiguredProvider

    store = _store(tmp_path)
    try:
        result = retrieve_with_authority(
            store, RetrievalRequest("unified", session_id="s"), NotConfiguredProvider(),
        )
    finally:
        store.close()

    assert len(result) == 1
    assert result[0].candidate.reference.event_id == 1


def test_provider_scope_is_rebound_at_authority_reopen(tmp_path: Path) -> None:
    store = _store(tmp_path)
    try:
        candidate = replace(
            _candidate_for(store, 1, "optional.provider", "unified authority needle"),
            channel_evidence=(ChannelScoreEvidence("optional.provider", 1, 0.9),),
        )

        class Provider:
            def retrieve(self, request: RetrievalRequest) -> RetrievalResult:
                return RetrievalResult.ok((candidate,))

        result = retrieve_with_authority(
            store,
            RetrievalRequest("ignored", session_id="other-session"),
            Provider(),
        )
    finally:
        store.close()

    assert isinstance(result, RetrievalResult)
    assert result.status is RetrievalStatus.ABSTAINED
    assert result.reason == "policy_filtered"


def test_core_candidates_preserve_their_lexical_channel_ranks(tmp_path: Path) -> None:
    store = _store(tmp_path)
    store.append_event(
        session_id="s", conversation_id="c", platform="desktop", source="chat", role="user",
        content="unified second authority needle", category="raw_conversation", subcategory="user",
        inject_policy="retrieve_only",
    )
    TranscriptChunkProjector(store, worker_id="test-ranks").run_once(
        now="2026-07-17T05:03:00Z", lease_until="2026-07-17T05:04:00Z"
    )
    try:
        result = retrieve_with_authority(
            store,
            RetrievalRequest("unified", session_id="s", limit=2),
        )
    finally:
        store.close()

    assert isinstance(result, tuple)
    assert [item.candidate.channel_evidence[0].channel_rank for item in result] == [1, 2]


def test_all_candidates_rejected_by_authority_reopen_returns_typed_abstention(tmp_path: Path) -> None:
    store = _store(tmp_path)
    try:
        forged = replace(
            _candidate_for(store, 1, "optional.provider", "unified authority needle"),
            quote="not present in authority",
            channel_evidence=(ChannelScoreEvidence("optional.provider", 1, 0.9),),
        )

        class Provider:
            def retrieve(self, request: RetrievalRequest) -> RetrievalResult:
                return RetrievalResult.ok((forged,))

        result = retrieve_with_authority(
            store,
            RetrievalRequest("no lexical match", session_id="s"),
            Provider(),
        )
    finally:
        store.close()

    assert isinstance(result, RetrievalResult)
    assert result.status is RetrievalStatus.UNAVAILABLE
    assert result.reason == "authority_reopen_failed"


def test_unified_entrypoint_enforces_global_limit_after_fusion(tmp_path: Path) -> None:
    store = _store(tmp_path)
    second_id = store.append_event(
        session_id="s", conversation_id="c", platform="desktop", source="chat", role="user",
        content="provider-only second needle", category="raw_conversation", subcategory="user",
        inject_policy="retrieve_only",
    )
    TranscriptChunkProjector(store, worker_id="test-limit").run_once(
        now="2026-07-17T05:05:00Z", lease_until="2026-07-17T05:06:00Z"
    )
    try:
        provider_candidate = replace(
            _candidate_for(store, second_id, "optional.provider", "provider-only second needle"),
            channel_evidence=(ChannelScoreEvidence("optional.provider", 1, 0.9),),
        )

        class Provider:
            def retrieve(self, request: RetrievalRequest) -> RetrievalResult:
                return RetrievalResult.ok((provider_candidate,))

        result = retrieve_with_authority(
            store, RetrievalRequest("unified", session_id="s", limit=1), Provider()
        )
    finally:
        store.close()

    assert isinstance(result, tuple)
    assert len(result) == 1


def test_blank_query_abstains_without_calling_provider(tmp_path: Path) -> None:
    store = _store(tmp_path)
    called = False
    try:
        class Provider:
            def retrieve(self, request: RetrievalRequest) -> RetrievalResult:
                nonlocal called
                called = True
                return RetrievalResult.ok(
                    (_candidate_for(store, 1, "optional.provider", "unified authority needle"),)
                )

        result = retrieve_with_authority(
            store, RetrievalRequest("   ", session_id="s"), Provider()
        )
    finally:
        store.close()

    assert isinstance(result, RetrievalResult)
    assert result.status is RetrievalStatus.ABSTAINED
    assert result.reason == "no_answer"
    assert called is False


def test_declared_optional_source_commitments_are_reopened(tmp_path: Path) -> None:
    store = _store(tmp_path)
    try:
        candidate = _candidate_for(store, 1, "optional.provider", "unified authority needle")
        forged = replace(
            candidate,
            reference=replace(
                candidate.reference,
                source_hash="forged-source-hash",
                source_created_at="forged-source-created-at",
            ),
            channel_evidence=(ChannelScoreEvidence("optional.provider", 1, 0.9),),
        )

        class Provider:
            def retrieve(self, request: RetrievalRequest) -> RetrievalResult:
                return RetrievalResult.ok((forged,))

        result = retrieve_with_authority(
            store, RetrievalRequest("absent", session_id="s"), Provider()
        )
    finally:
        store.close()

    assert isinstance(result, RetrievalResult)
    assert result.status is RetrievalStatus.UNAVAILABLE
    assert result.reason == "authority_reopen_failed"


def test_authoritative_optional_commitments_cannot_be_omitted(tmp_path: Path) -> None:
    store = EventStore(tmp_path / "committed-authority.db")
    event_id, inserted = store.ingest_external_event(
        external_id="committed:1", source_hash="source-hash", kind="chat_message",
        payload_metadata={"timestamp": "2026-07-17T05:00:00Z"},
        session_id="s", conversation_id="c", platform="desktop", source="chat", role="user",
        content="committed authority needle", category="raw_conversation", subcategory="user",
        inject_policy="retrieve_only",
    )
    assert inserted is True
    TranscriptChunkProjector(store, worker_id="test-commitments").run_once(
        now="2026-07-17T05:01:00Z", lease_until="2026-07-17T05:02:00Z"
    )
    try:
        omitted = replace(
            _candidate_for(store, event_id, "optional.provider", "committed authority needle"),
            channel_evidence=(ChannelScoreEvidence("optional.provider", 1, 0.9),),
        )

        class Provider:
            def retrieve(self, request: RetrievalRequest) -> RetrievalResult:
                return RetrievalResult.ok((omitted,))

        result = retrieve_with_authority(
            store, RetrievalRequest("absent", session_id="s"), Provider()
        )
    finally:
        store.close()

    assert isinstance(result, RetrievalResult)
    assert result.status is RetrievalStatus.UNAVAILABLE
    assert result.reason == "authority_reopen_failed"


def test_core_lexical_carries_external_ingest_commitments_through_reopen(tmp_path: Path) -> None:
    store = EventStore(tmp_path / "external-core.db")
    event_id, inserted = store.ingest_external_event(
        external_id="external-core:1", source_hash="external-source-hash", kind="chat_message",
        payload_metadata={"timestamp": "2026-07-17T05:00:00Z"},
        session_id="s", conversation_id="c", platform="desktop", source="chat", role="user",
        content="external committed lexical needle", category="raw_conversation",
        subcategory="user", inject_policy="retrieve_only",
    )
    assert inserted is True
    TranscriptChunkProjector(store, worker_id="test-external-core").run_once(
        now="2026-07-17T05:01:00Z", lease_until="2026-07-17T05:02:00Z"
    )
    try:
        result = retrieve_with_authority(
            store, RetrievalRequest("external committed", session_id="s")
        )
    finally:
        store.close()

    assert isinstance(result, tuple)
    assert len(result) == 1
    reference = result[0].candidate.reference
    assert reference.event_id == event_id
    assert reference.source_hash == "external-source-hash"
    assert reference.source_created_at == "2026-07-17T05:00:00Z"


def test_verified_background_records_preserve_external_source_hash(tmp_path: Path) -> None:
    store = EventStore(tmp_path / "external-background.db")
    store.ingest_external_event(
        external_id="external-background:1", source_hash="background-source-hash",
        kind="chat_message", payload_metadata={"timestamp": "2026-07-17T05:00:00Z"},
        session_id="s", conversation_id="c", platform="desktop", source="chat", role="user",
        content="external background commitment needle", category="raw_conversation",
        subcategory="user", inject_policy="retrieve_only",
    )
    TranscriptChunkProjector(store, worker_id="test-external-background").run_once(
        now="2026-07-17T05:01:00Z", lease_until="2026-07-17T05:02:00Z"
    )
    try:
        records = verified_background_records(
            store, RetrievalRequest("external background", session_id="s")
        )
    finally:
        store.close()

    assert len(records) == 1
    assert records[0]["source_hash"] == "background-source-hash"


def test_legacy_provider_candidate_order_becomes_channel_rank_evidence(tmp_path: Path) -> None:
    store = _store(tmp_path)
    second_id = store.append_event(
        session_id="s", conversation_id="c", platform="desktop", source="chat", role="user",
        content="second provider ordered needle", category="raw_conversation", subcategory="user",
        inject_policy="retrieve_only",
    )
    TranscriptChunkProjector(store, worker_id="test-provider-order").run_once(
        now="2026-07-17T05:03:00Z", lease_until="2026-07-17T05:04:00Z"
    )
    try:
        first = _candidate_for(store, second_id, "legacy.provider", "second provider ordered needle")
        second = _candidate_for(store, 1, "legacy.provider", "unified authority needle")

        class Provider:
            def retrieve(self, request: RetrievalRequest) -> RetrievalResult:
                return RetrievalResult.ok((first, second))

        result = retrieve_with_authority(
            store, RetrievalRequest("absent", session_id="s", limit=2), Provider()
        )
    finally:
        store.close()

    assert isinstance(result, tuple)
    assert [item.candidate.reference.event_id for item in result] == [second_id, 1]
    assert [item.candidate.channel_evidence[0].channel_rank for item in result] == [1, 2]
