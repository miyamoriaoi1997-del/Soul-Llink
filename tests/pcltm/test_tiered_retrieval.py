from __future__ import annotations

from dataclasses import replace
import sqlite3
from pathlib import Path

from pcltm.retrieval_provider import (
    AuthorityReference,
    Candidate,
    RetrievalRequest,
    RetrievalResult,
    RetrievalStatus,
)
from pcltm.projections.transcript_chunks import TranscriptChunkProjector
from pcltm.store import EventStore
from pcltm.tiered_retrieval import retrieve_lexical, retrieve_with_authority


def _store(tmp_path: Path) -> EventStore:
    store = EventStore(tmp_path / "core.db")
    store.append_event(
        session_id="s1", conversation_id="c1", platform="desktop", role="user",
        source="chat", content="needle in scope one", category="raw_conversation",
        subcategory="user", inject_policy="retrieve_only",
    )
    store.append_event(
        session_id="s2", conversation_id="c2", platform="desktop", role="user",
        source="chat", content="needle outside scope", category="raw_conversation",
        subcategory="user", inject_policy="retrieve_only",
    )
    for index in range(2):
        TranscriptChunkProjector(store, worker_id=f"test-{index}").run_once(
            now=f"2026-07-17T05:0{index}:00Z",
            lease_until=f"2026-07-17T05:0{index + 1}:00Z",
        )
    return store


def test_core_lexical_returns_typed_candidates_with_scope_filter_and_less_than_k(tmp_path: Path) -> None:
    store = _store(tmp_path)
    try:
        result = retrieve_lexical(store, RetrievalRequest("needle", limit=10, session_id="s1"))
    finally:
        store.close()

    assert result.status.value == "ok"
    assert len(result.candidates) == 1
    assert result.candidates[0].reference.event_id == 1
    assert result.candidates[0].provider == "core.lexical"


def test_core_lexical_preserves_fts_case_and_noncontiguous_term_matches(tmp_path: Path) -> None:
    store = EventStore(tmp_path / "core.db")
    try:
        store.append_event(
            session_id="s", conversation_id="c", platform="desktop", role="user",
            source="chat", content="Needle appears inside the requested scope",
            category="raw_conversation", subcategory="user", inject_policy="retrieve_only",
        )
        TranscriptChunkProjector(store, worker_id="test").run_once(
            now="2026-07-17T05:01:00Z", lease_until="2026-07-17T05:02:00Z"
        )
        lower = retrieve_lexical(store, RetrievalRequest("needle", session_id="s"))
        multi = retrieve_lexical(store, RetrievalRequest("needle scope", session_id="s"))
    finally:
        store.close()

    assert len(lower.candidates) == 1
    assert lower.candidates[0].quote == "Needle"
    assert len(multi.candidates) == 1
    assert multi.candidates[0].quote == "Needle appears inside the requested scope"


def test_core_lexical_excludes_sensitive_events(tmp_path: Path) -> None:
    store = EventStore(tmp_path / "core.db")
    try:
        store.append_event(
            session_id="s", conversation_id="c", platform="desktop", role="user",
            source="chat", content="sensitive needle", sensitivity="private",
            category="raw_conversation", subcategory="user", inject_policy="retrieve_only",
        )
        result = retrieve_lexical(store, RetrievalRequest("needle", limit=5))
    finally:
        store.close()

    assert result.candidates == ()
    assert result.status.value == "abstained"


def test_core_lexical_abstains_when_authority_chunk_is_missing(tmp_path: Path) -> None:
    store = EventStore(tmp_path / "core.db")
    try:
        store.append_event(
            session_id="s", conversation_id="c", platform="desktop", role="user",
            source="chat", content="needle without projection", category="raw_conversation",
            subcategory="user", inject_policy="retrieve_only",
        )
        result = retrieve_lexical(store, RetrievalRequest("needle", session_id="s"))
    finally:
        store.close()

    assert result.status.value == "abstained"
    assert result.candidates == ()


def test_authority_reopen_rejects_missing_chunk_reference(tmp_path: Path) -> None:
    store = _store(tmp_path)
    try:
        event = store.get_event(1)
        chunk = store._conn.execute("SELECT * FROM event_chunks WHERE event_id=1").fetchone()
        candidate = Candidate(
            reference=AuthorityReference(
                event_id=1, chunk_id=999_999, chunk_ordinal=int(chunk["chunk_ordinal"]),
                source_revision=int(event["source_revision"]), start_char=0, end_char=6,
                payload_sha256=event["payload_sha256"], chain_hash=event["chain_hash"],
                chunk_sha256=chunk["chunk_sha256"],
            ),
            score=0.9, provider="fake.neural", quote="needle",
        )

        class FakeProvider:
            def retrieve(self, request: RetrievalRequest) -> RetrievalResult:
                return RetrievalResult.ok((candidate,))

        result = retrieve_with_authority(
            store, RetrievalRequest("absent", session_id="s1"), FakeProvider()
        )
    finally:
        store.close()

    assert isinstance(result, RetrievalResult)
    assert result.status is RetrievalStatus.UNAVAILABLE
    assert result.reason == "authority_reopen_failed"


def test_authority_reopen_accepts_current_chunk_and_exact_evidence(tmp_path: Path) -> None:
    store = EventStore(tmp_path / "core.db")
    try:
        event_id = store.append_event(
            session_id="s", conversation_id="c", platform="desktop", role="user",
            source="chat", content="needle in authoritative evidence", category="raw_conversation",
            subcategory="user", inject_policy="retrieve_only",
        )
        TranscriptChunkProjector(store, worker_id="test").run_once(
            now="2026-07-17T05:01:00Z", lease_until="2026-07-17T05:02:00Z"
        )
        result = retrieve_with_authority(store, RetrievalRequest("needle", session_id="s"))
    finally:
        store.close()

    assert len(result) == 1
    assert result[0].candidate.reference.event_id == event_id
    assert result[0].evidence.verified is True


def test_authority_reopen_accepts_exact_evidence_spanning_multiple_chunks(tmp_path: Path) -> None:
    store = EventStore(tmp_path / "core.db")
    quote = "ABCDEFGHIJKL"
    content = f"prefix-{quote}-suffix"
    try:
        event_id = store.append_event(
            session_id="s", conversation_id="c", platform="desktop", role="user",
            source="chat", content=content, category="raw_conversation",
            subcategory="user", inject_policy="retrieve_only",
        )
        TranscriptChunkProjector(
            store, worker_id="test", max_chars=10, overlap_chars=2
        ).run_once(now="2026-07-17T05:01:00Z", lease_until="2026-07-17T05:02:00Z")
        event = store.get_event(event_id)
        start = content.index(quote)
        end = start + len(quote)
        chunk = store._conn.execute(
            """SELECT * FROM event_chunks
               WHERE event_id=? AND start_char < ? AND end_char > ?
               ORDER BY start_char, chunk_ordinal LIMIT 1""",
            (event_id, end, start),
        ).fetchone()
        candidate = Candidate(
            reference=AuthorityReference(
                event_id=event_id, chunk_id=int(chunk["chunk_id"]),
                chunk_ordinal=int(chunk["chunk_ordinal"]),
                source_revision=int(event["source_revision"]), start_char=start, end_char=end,
                payload_sha256=event["payload_sha256"], chain_hash=event["chain_hash"],
                chunk_sha256=chunk["chunk_sha256"],
            ),
            score=0.9, provider="fake.neural", quote=quote,
        )

        class FakeProvider:
            def retrieve(self, request: RetrievalRequest) -> RetrievalResult:
                return RetrievalResult.ok((candidate,))

        result = retrieve_with_authority(
            store, RetrievalRequest("ignored", session_id="s"), FakeProvider()
        )
    finally:
        store.close()

    assert len(result) == 1
    assert result[0].evidence.start_char == start
    assert result[0].evidence.end_char == end


def test_optional_provider_candidates_are_reopened_before_verified_output(tmp_path: Path) -> None:
    store = EventStore(tmp_path / "core.db")
    try:
        event_id = store.append_event(
            session_id="s", conversation_id="c", platform="desktop", role="user",
            source="chat", content="authoritative needle", category="raw_conversation",
            subcategory="user", inject_policy="retrieve_only",
        )
        TranscriptChunkProjector(store, worker_id="test").run_once(
            now="2026-07-17T05:01:00Z", lease_until="2026-07-17T05:02:00Z"
        )
        event = store.get_event(event_id)
        chunk = store._conn.execute("SELECT * FROM event_chunks WHERE event_id=?", (event_id,)).fetchone()
        candidate = Candidate(
            reference=AuthorityReference(
                event_id=event_id, chunk_id=int(chunk["chunk_id"]), chunk_ordinal=0,
                source_revision=int(event["source_revision"]), start_char=0, end_char=20,
                payload_sha256=event["payload_sha256"], chain_hash=event["chain_hash"],
                chunk_sha256=chunk["chunk_sha256"],
            ),
            score=0.9, provider="fake.neural", quote="authoritative needle",
        )

        class FakeProvider:
            def retrieve(self, request: RetrievalRequest) -> RetrievalResult:
                return RetrievalResult.ok((candidate,))

        result = retrieve_with_authority(store, RetrievalRequest("ignored", session_id="s"), FakeProvider())
    finally:
        store.close()

    assert len(result) == 1
    assert result[0].candidate.provider == "fake.neural"
    assert result[0].evidence.verified is True


def test_optional_provider_cannot_cross_requested_authority_scope(tmp_path: Path) -> None:
    store = EventStore(tmp_path / "core.db")
    try:
        event_id = store.append_event(
            session_id="foreign-session", conversation_id="foreign-conversation",
            platform="desktop", role="user", source="chat",
            content="foreign authoritative needle", category="raw_conversation",
            subcategory="user", inject_policy="retrieve_only",
        )
        TranscriptChunkProjector(store, worker_id="test").run_once(
            now="2026-07-17T05:01:00Z", lease_until="2026-07-17T05:02:00Z"
        )
        event = store.get_event(event_id)
        chunk = store._conn.execute("SELECT * FROM event_chunks WHERE event_id=?", (event_id,)).fetchone()
        candidate = Candidate(
            reference=AuthorityReference(
                event_id=event_id, chunk_id=int(chunk["chunk_id"]),
                chunk_ordinal=int(chunk["chunk_ordinal"]),
                source_revision=int(event["source_revision"]), start_char=0,
                end_char=len("foreign authoritative needle"),
                payload_sha256=event["payload_sha256"], chain_hash=event["chain_hash"],
                chunk_sha256=chunk["chunk_sha256"],
            ),
            score=0.9, provider="fake.neural", quote="foreign authoritative needle",
        )

        class FakeProvider:
            def retrieve(self, request: RetrievalRequest) -> RetrievalResult:
                return RetrievalResult.ok((candidate,))

        result = retrieve_with_authority(
            store,
            RetrievalRequest(
                "ignored", session_id="requested-session",
                conversation_id="requested-conversation", platform="desktop", source="chat",
            ),
            FakeProvider(),
        )
    finally:
        store.close()

    assert isinstance(result, RetrievalResult)
    assert result.status is RetrievalStatus.ABSTAINED
    assert result.reason == "policy_filtered"


def test_authority_reopen_rejects_wrong_chunk_ordinal(tmp_path: Path) -> None:
    store = EventStore(tmp_path / "core.db")
    try:
        event_id = store.append_event(
            session_id="s", conversation_id="c", platform="desktop", role="user",
            source="chat", content="ordinal authority needle", category="raw_conversation",
            subcategory="user", inject_policy="retrieve_only",
        )
        TranscriptChunkProjector(store, worker_id="test").run_once(
            now="2026-07-17T05:01:00Z", lease_until="2026-07-17T05:02:00Z"
        )
        event = store.get_event(event_id)
        chunk = store._conn.execute("SELECT * FROM event_chunks WHERE event_id=?", (event_id,)).fetchone()
        candidate = Candidate(
            reference=AuthorityReference(
                event_id=event_id, chunk_id=int(chunk["chunk_id"]), chunk_ordinal=999,
                source_revision=int(event["source_revision"]), start_char=0,
                end_char=len("ordinal authority needle"),
                payload_sha256=event["payload_sha256"], chain_hash=event["chain_hash"],
                chunk_sha256=chunk["chunk_sha256"],
            ),
            score=0.9, provider="fake.neural", quote="ordinal authority needle",
        )

        class FakeProvider:
            def retrieve(self, request: RetrievalRequest) -> RetrievalResult:
                return RetrievalResult.ok((candidate,))

        result = retrieve_with_authority(
            store, RetrievalRequest("ignored", session_id="s"), FakeProvider()
        )
    finally:
        store.close()

    assert isinstance(result, RetrievalResult)
    assert result.status is RetrievalStatus.UNAVAILABLE
    assert result.reason == "authority_reopen_failed"


def test_authority_reopen_rejects_stale_reference_and_revoked_governance(tmp_path: Path) -> None:
    store = EventStore(tmp_path / "core.db")
    try:
        event_id = store.append_event(
            session_id="s", conversation_id="c", platform="desktop", role="user",
            source="chat", content="governed authority needle", category="raw_conversation",
            subcategory="user", inject_policy="retrieve_only",
        )
        TranscriptChunkProjector(store, worker_id="test").run_once(
            now="2026-07-17T05:01:00Z", lease_until="2026-07-17T05:02:00Z"
        )
        event = store.get_event(event_id)
        chunk = store._conn.execute("SELECT * FROM event_chunks WHERE event_id=?", (event_id,)).fetchone()
        current = Candidate(
            reference=AuthorityReference(
                event_id=event_id, chunk_id=int(chunk["chunk_id"]),
                chunk_ordinal=int(chunk["chunk_ordinal"]),
                source_revision=int(event["source_revision"]), start_char=0,
                end_char=len("governed authority needle"),
                payload_sha256=event["payload_sha256"], chain_hash=event["chain_hash"],
                chunk_sha256=chunk["chunk_sha256"],
            ),
            score=0.9, provider="fake.neural", quote="governed authority needle",
        )
        stale = replace(
            current,
            reference=replace(current.reference, payload_sha256="0" * 64),
        )

        class StaleProvider:
            def retrieve(self, request: RetrievalRequest) -> RetrievalResult:
                return RetrievalResult.ok((stale,))

        stale_result = retrieve_with_authority(
            store, RetrievalRequest("ignored", session_id="s"), StaleProvider()
        )
        assert isinstance(stale_result, RetrievalResult)
        assert stale_result.status is RetrievalStatus.UNAVAILABLE
        assert stale_result.reason == "authority_reopen_failed"

        store._conn.execute(
            """INSERT INTO event_governance
               (event_id, action, previous_state, new_state, actor, reason)
               VALUES (?, 'revoke', 'active', 'revoked', 'test', 'contract probe')""",
            (event_id,),
        )
        store._conn.commit()

        class CurrentProvider:
            def retrieve(self, request: RetrievalRequest) -> RetrievalResult:
                return RetrievalResult.ok((current,))

        revoked_result = retrieve_with_authority(
            store, RetrievalRequest("ignored", session_id="s"), CurrentProvider()
        )
        assert isinstance(revoked_result, RetrievalResult)
        assert revoked_result.status is RetrievalStatus.UNAVAILABLE
        assert revoked_result.reason == "authority_reopen_failed"
    finally:
        store.close()


def test_mixed_valid_and_corrupt_provider_candidates_fail_closed(tmp_path: Path) -> None:
    store = EventStore(tmp_path / "core.db")
    try:
        candidates = []
        for index in range(2):
            content = f"opaque-{index}-needle"
            event_id = store.append_event(
                session_id="s", conversation_id="c", platform="desktop", role="user",
                source="chat", content=content, category="raw_conversation",
                subcategory="user", inject_policy="retrieve_only",
            )
            TranscriptChunkProjector(store, worker_id=f"test-{index}").run_once(
                now=f"2026-07-17T05:0{index}:00Z",
                lease_until=f"2026-07-17T05:0{index + 1}:00Z",
            )
            event = store.get_event(event_id)
            chunk = store._conn.execute(
                "SELECT * FROM event_chunks WHERE event_id = ?", (event_id,),
            ).fetchone()
            candidates.append(Candidate(
                reference=AuthorityReference(
                    event_id=event_id, chunk_id=int(chunk["chunk_id"]),
                    chunk_ordinal=int(chunk["chunk_ordinal"]),
                    source_revision=int(event["source_revision"]), start_char=0,
                    end_char=len(content), payload_sha256=event["payload_sha256"],
                    chain_hash=event["chain_hash"], chunk_sha256=chunk["chunk_sha256"],
                ),
                score=1.0 - index / 10, provider="fake.neural", quote=content,
            ))
        candidates[1] = replace(
            candidates[1],
            reference=replace(candidates[1].reference, payload_sha256="0" * 64),
        )

        class MixedProvider:
            def retrieve(self, request: RetrievalRequest) -> RetrievalResult:
                return RetrievalResult.ok(tuple(candidates))

        result = retrieve_with_authority(
            store, RetrievalRequest("absent-from-core", session_id="s"), MixedProvider(),
        )
    finally:
        store.close()

    assert isinstance(result, RetrievalResult)
    assert result.status is RetrievalStatus.UNAVAILABLE
    assert result.reason == "authority_reopen_failed"


def test_authority_reopen_is_zero_write(tmp_path: Path) -> None:
    store = EventStore(tmp_path / "core.db")
    db = tmp_path / "core.db"
    try:
        store.append_event(
            session_id="s", conversation_id="c", platform="desktop", role="user",
            source="chat", content="zero write authority needle", category="raw_conversation",
            subcategory="user", inject_policy="retrieve_only",
        )
        TranscriptChunkProjector(store, worker_id="test").run_once(
            now="2026-07-17T05:01:00Z", lease_until="2026-07-17T05:02:00Z"
        )
        before = db.read_bytes()
        before_outbox = store._conn.execute(
            "SELECT projection_kind, status, attempt_count FROM projection_outbox ORDER BY outbox_id"
        ).fetchall()
        result = retrieve_with_authority(store, RetrievalRequest("needle", session_id="s"))
        after = db.read_bytes()
        after_outbox = store._conn.execute(
            "SELECT projection_kind, status, attempt_count FROM projection_outbox ORDER BY outbox_id"
        ).fetchall()
    finally:
        store.close()

    assert len(result) == 1
    assert before == after
    assert before_outbox == after_outbox


def test_not_configured_provider_does_not_replace_core_authority_path(tmp_path: Path) -> None:
    from pcltm.retrieval_provider import NotConfiguredProvider

    store = EventStore(tmp_path / "core.db")
    try:
        store.append_event(
            session_id="s", conversation_id="c", platform="desktop", role="user",
            source="chat", content="core survives optional provider", category="raw_conversation",
            subcategory="user", inject_policy="retrieve_only",
        )
        TranscriptChunkProjector(store, worker_id="test").run_once(
            now="2026-07-17T05:01:00Z", lease_until="2026-07-17T05:02:00Z"
        )
        result = retrieve_with_authority(
            store,
            RetrievalRequest("core", session_id="s"),
            NotConfiguredProvider(),
        )
    finally:
        store.close()

    assert len(result) == 1
    assert result[0].candidate.provider == "core.lexical"
    assert result[0].evidence.verified is True


def test_core_and_not_configured_paths_are_independent(tmp_path: Path) -> None:
    from pcltm.retrieval_provider import NotConfiguredProvider

    store = _store(tmp_path)
    try:
        lexical = retrieve_lexical(store, RetrievalRequest("needle", session_id="s1"))
        unavailable = NotConfiguredProvider().retrieve(RetrievalRequest("needle"))
    finally:
        store.close()

    assert lexical.status.value == "ok"
    assert unavailable.status.value == "unavailable"
    assert unavailable.reason == "not_configured"


def test_no_answer_abstains_without_forced_fill(tmp_path: Path) -> None:
    store = _store(tmp_path)
    try:
        result = retrieve_lexical(store, RetrievalRequest("absent", limit=10))
    finally:
        store.close()

    assert result.status.value == "abstained"
    assert result.candidates == ()


def test_core_retrieval_is_zero_write(tmp_path: Path) -> None:
    store = _store(tmp_path)
    db = tmp_path / "core.db"
    try:
        before = db.read_bytes()
        before_data_version = store._conn.execute("PRAGMA data_version").fetchone()[0]
        before_outbox = store._conn.execute(
            "SELECT projection_kind, status, attempt_count FROM projection_outbox ORDER BY outbox_id"
        ).fetchall()
        result = retrieve_lexical(store, RetrievalRequest("needle", session_id="s1"))
        after = db.read_bytes()
        after_data_version = store._conn.execute("PRAGMA data_version").fetchone()[0]
        after_outbox = store._conn.execute(
            "SELECT projection_kind, status, attempt_count FROM projection_outbox ORDER BY outbox_id"
        ).fetchall()
    finally:
        store.close()

    assert result.candidates
    assert before == after
    assert before_data_version == after_data_version
    assert before_outbox == after_outbox
    assert not any(name in retrieve_lexical.__code__.co_names for name in ("sync_memory_tool_write", "record_retrieval"))
