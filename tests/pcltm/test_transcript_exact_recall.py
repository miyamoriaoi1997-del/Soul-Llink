from __future__ import annotations

from pathlib import Path

from pcltm.memory_contracts import PersonaMode
from pcltm.projections.transcript_chunks import TranscriptChunkProjector
from pcltm.store import EventStore
from pcltm.transcript_search import search_exact_evidence



def test_exact_recall_returns_verified_e0_quote_and_offsets(tmp_path: Path) -> None:
    store = EventStore(tmp_path / "pcltm.db")
    text = "前文。老师明确要求永久保存每一句原文，并能精准召回。后文。"
    try:
        event_id, _ = store.ingest_external_event(
            external_id="source:1", source_hash="hash", kind="chat_message",
            payload_metadata={"created_at": "2026-07-17T05:00:00Z"},
            session_id="s", conversation_id="c", platform="desktop",
            role="user", source="test", content=text,
            category="raw_conversation", subcategory="user", inject_policy="retrieve_only",
        )
        TranscriptChunkProjector(store, worker_id="worker", max_chars=20, overlap_chars=8).run_once(
            now="2026-07-17T05:01:00Z", lease_until="2026-07-17T05:02:00Z"
        )
        hits = search_exact_evidence(store, "永久保存每一句原文", limit=5)
    finally:
        store.close()

    assert len(hits) == 1
    hit = hits[0]
    assert hit.evidence_level == "E0"
    assert hit.event_id == event_id
    assert hit.quote == "永久保存每一句原文"
    assert text[hit.start_char:hit.end_char] == hit.quote
    assert hit.verified is True
    assert hit.integrity_scope == "l1_local_consistency"
    assert hit.payload_sha256
    assert hit.source_created_at == "2026-07-17T05:00:00Z"


def test_exact_recall_uses_one_snapshot_for_chain_verification_and_query(tmp_path: Path, monkeypatch) -> None:
    db = tmp_path / "pcltm.db"
    writer = EventStore(db)
    reader = EventStore(db)
    try:
        writer.append_event(
            session_id="s", conversation_id="c", platform="desktop",
            role="user", source="test", content="verified phrase",
            category="raw_conversation", subcategory="user", inject_policy="retrieve_only",
        )
        original_verify = reader.verify_event_chain

        def verify_then_write():
            report = original_verify()
            writer.append_event(
                session_id="s", conversation_id="c", platform="desktop",
                role="assistant", source="test", content="race phrase",
                category="raw_conversation", subcategory="assistant", inject_policy="retrieve_only",
            )
            return report

        monkeypatch.setattr(reader, "verify_event_chain", verify_then_write)
        hits = search_exact_evidence(reader, "race phrase", limit=5)
    finally:
        reader.close()
        writer.close()

    assert hits == []


def test_exact_recall_refuses_any_result_when_event_chain_is_invalid(tmp_path: Path) -> None:
    store = EventStore(tmp_path / "pcltm.db")
    try:
        event_id, _ = store.ingest_external_event(
            external_id="source:tamper", source_hash="hash", kind="chat_message",
            session_id="s", conversation_id="c", platform="desktop",
            role="user", source="test", content="authoritative phrase",
            category="raw_conversation", subcategory="user", inject_policy="retrieve_only",
        )
        store._conn.execute("DROP TRIGGER protect_events_update")
        store._conn.execute(
            "UPDATE events SET content='forged phrase', payload_sha256=? WHERE event_id=?",
            (__import__('hashlib').sha256(b'forged phrase').hexdigest(), event_id),
        )
        store._conn.commit()
        assert store.verify_event_chain()["ok"] is False
        hits = search_exact_evidence(store, "forged phrase", limit=5)
    finally:
        store.close()

    assert hits == []


def test_exact_recall_requires_a_chunk_covering_the_quote(tmp_path: Path) -> None:
    store = EventStore(tmp_path / "pcltm.db")
    try:
        store.ingest_external_event(
            external_id="source:no-chunk", source_hash="hash", kind="chat_message",
            session_id="s", conversation_id="c", platform="desktop",
            role="user", source="test", content="exact phrase without projection",
            category="raw_conversation", subcategory="user", inject_policy="retrieve_only",
        )
        hits = search_exact_evidence(store, "exact phrase", limit=5)
    finally:
        store.close()

    assert hits == []


def test_exact_recall_uses_latest_governance_state(tmp_path: Path) -> None:
    store = EventStore(tmp_path / "pcltm.db")
    try:
        event_id, _ = store.ingest_external_event(
            external_id="source:restore", source_hash="hash", kind="chat_message",
            session_id="s", conversation_id="c", platform="desktop",
            role="user", source="test", content="restored exact phrase",
            category="raw_conversation", subcategory="user", inject_policy="retrieve_only",
        )
        TranscriptChunkProjector(store, worker_id="worker").run_once(
            now="2026-07-17T05:01:00Z", lease_until="2026-07-17T05:02:00Z"
        )
        store._conn.execute(
            """INSERT INTO event_governance
               (event_id, action, previous_state, new_state, actor, reason)
               VALUES (?, 'redact', 'active', 'redacted', 'test', 'temporary')""",
            (event_id,),
        )
        store._conn.execute(
            """INSERT INTO event_governance
               (event_id, action, previous_state, new_state, actor, reason)
               VALUES (?, 'restore', 'redacted', 'active', 'test', 'approved')""",
            (event_id,),
        )
        store._conn.commit()
        hits = search_exact_evidence(store, "restored exact phrase", limit=5)
    finally:
        store.close()

    assert len(hits) == 1
    assert hits[0].event_id == event_id


def test_exact_recall_refuses_tampered_chunk(tmp_path: Path) -> None:
    store = EventStore(tmp_path / "pcltm.db")
    try:
        store.ingest_external_event(
            external_id="source:1", source_hash="hash", kind="chat_message",
            session_id="s", conversation_id="c", platform="desktop",
            role="user", source="test", content="需要验证的精确原文",
            category="raw_conversation", subcategory="user", inject_policy="retrieve_only",
        )
        TranscriptChunkProjector(store, worker_id="worker").run_once(
            now="2026-07-17T05:01:00Z", lease_until="2026-07-17T05:02:00Z"
        )
        store._conn.execute("UPDATE event_chunks SET chunk_text='伪造的精确原文'")
        store._conn.commit()
        hits = search_exact_evidence(store, "伪造的精确原文", limit=5)
    finally:
        store.close()

    assert hits == []


def test_exact_recall_finds_phrase_crossing_chunk_boundary(tmp_path: Path) -> None:
    store = EventStore(tmp_path / "pcltm.db")
    text = "甲乙丙丁戊己庚辛壬癸永久证据精确召回子丑寅卯辰巳午未申酉"
    try:
        event_id, _ = store.ingest_external_event(
            external_id="source:boundary", source_hash="hash", kind="chat_message",
            session_id="s", conversation_id="c", platform="desktop",
            role="user", source="test", content=text,
            category="raw_conversation", subcategory="user", inject_policy="retrieve_only",
        )
        TranscriptChunkProjector(store, worker_id="worker", max_chars=12, overlap_chars=2).run_once(
            now="2026-07-17T05:01:00Z", lease_until="2026-07-17T05:02:00Z"
        )
        hits = search_exact_evidence(store, "癸永久证据精确召回子", limit=5)
    finally:
        store.close()

    assert len(hits) == 1
    assert hits[0].event_id == event_id
    assert hits[0].verified is True
    assert text[hits[0].start_char:hits[0].end_char] == hits[0].quote


def test_exact_recall_applies_event_sensitivity_and_persona_mode_policy(tmp_path: Path) -> None:
    store = EventStore(tmp_path / "pcltm.db")
    try:
        private_id, _ = store.ingest_external_event(
            external_id="source:private", source_hash="hash-private", kind="chat_message",
            session_id="s", conversation_id="c", platform="desktop",
            role="user", source="test", content="私密精确原文",
            sensitivity="private", persona_mode="daily",
            category="raw_conversation", subcategory="user", inject_policy="retrieve_only",
        )
        restricted_id, _ = store.ingest_external_event(
            external_id="source:restricted", source_hash="hash-restricted", kind="chat_message",
            session_id="s", conversation_id="c", platform="desktop",
            role="user", source="test", content="受限精确原文",
            sensitivity="restricted", persona_mode="daily",
            category="raw_conversation", subcategory="user", inject_policy="retrieve_only",
        )
        projector = TranscriptChunkProjector(store, worker_id="worker")
        projector.run_once(now="2026-07-17T05:01:00Z", lease_until="2026-07-17T05:02:00Z")
        projector.run_once(now="2026-07-17T05:03:00Z", lease_until="2026-07-17T05:04:00Z")

        assert search_exact_evidence(
            store, "私密精确原文", limit=5, persona_mode=PersonaMode.WORK,
        ) == []
        private_daily = search_exact_evidence(
            store, "私密精确原文", limit=5, persona_mode=PersonaMode.DAILY,
        )
        assert [item.event_id for item in private_daily] == [private_id]
        assert search_exact_evidence(
            store, "受限精确原文", limit=5, persona_mode=PersonaMode.DAILY,
        ) == []
        assert restricted_id > private_id
    finally:
        store.close()
