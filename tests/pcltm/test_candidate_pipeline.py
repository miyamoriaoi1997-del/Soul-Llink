"""End-to-end tests for the candidate pipeline: conversation -> event -> candidate -> claim.

Verifies the fix for "extractor built but never wired":
1. HermesHistoryIngestor now classifies user messages as candidate_only via
   EventClassifier instead of hardcoding retrieve_only.
2. PersonaCandidateExtractor accepts hermes_state_db events.
3. CandidatePromotionService applies guardrails (>=0.85 activate, 0.6-0.85
   pending, <0.6 drop) and promotes to governed claims.
4. Promoted claims are recallable after projection drain.
"""
from __future__ import annotations

import math
import sqlite3
from pathlib import Path

import pytest

from pcltm.candidate_promotion import CandidatePromotionService
from pcltm.candidates import PersonaCandidateExtractor
from pcltm.hermes_history import HermesHistoryIngestor
from pcltm.memory_contracts import PersonaMode
from pcltm.memory_retrieval import GovernedMemorySearchRequest, search_governed_memories
from pcltm.projections.memory_runtime import drain_memory_projections
from pcltm.store import EventStore


def _build_hermes_db(path: Path) -> None:
    conn = sqlite3.connect(path)
    conn.execute(
        "CREATE TABLE sessions (id TEXT PRIMARY KEY, source TEXT, parent_session_id TEXT, "
        "started_at TEXT, ended_at TEXT, end_reason TEXT, archived INTEGER, rewind_count INTEGER, "
        "system_prompt TEXT)"
    )
    conn.execute(
        "CREATE TABLE messages (id INTEGER PRIMARY KEY, session_id TEXT, role TEXT, content TEXT, "
        "timestamp TEXT, active INTEGER, compacted INTEGER, observed INTEGER, token_count INTEGER, "
        "finish_reason TEXT, platform_message_id TEXT, tool_call_id TEXT, tool_name TEXT, tool_calls TEXT)"
    )
    conn.execute(
        "INSERT INTO sessions VALUES ('s1','desktop',NULL,'2026-08-01T10:00:00Z',NULL,NULL,0,0,'secret system prompt')"
    )
    conn.executemany(
        "INSERT INTO messages (id, session_id, role, content, timestamp, active) VALUES (?,?,?,?,?,1)",
        [
            (1, "s1", "user", "[memory] OPAQUE_DURABLE_TOKEN", "2026-08-01T10:00:01Z"),
            (2, "s1", "assistant", "好的，记住了。", "2026-08-01T10:00:02Z"),
        ],
    )
    conn.commit()
    conn.close()


@pytest.fixture()
def store(tmp_path: Path):
    store = EventStore(tmp_path / "pcltm.db")
    yield store
    store.close()


def _ingest(store: EventStore, tmp_path: Path, *, persona_mode: str) -> dict:
    hermes_db = tmp_path / "state.db"
    _build_hermes_db(hermes_db)
    return HermesHistoryIngestor(store, hermes_db).ingest(persona_mode=persona_mode)


def _append_durable_event(store: EventStore, content: str) -> dict:
    session_id = f"semantic-{store._conn.execute('SELECT COUNT(*) FROM events').fetchone()[0]}"
    event_id = store.append_event(
        session_id=session_id, conversation_id=session_id, platform="test",
        role="user", source="chat", content=content, persona_mode="daily",
    )
    return {"event_id": event_id, "session_id": session_id}


def test_ingest_classifies_user_message_as_candidate_only(store: EventStore, tmp_path: Path) -> None:
    _ingest(store, tmp_path, persona_mode="work")
    user_events = [e for e in store.list_events(limit=20) if e["role"] == "user"]
    assert user_events
    ue = user_events[0]
    assert ue["inject_policy"] == "candidate_only"
    assert ue["classification_confidence"] == 0.95
    assert ue["category"] == "work"
    assert ue["source"] == "hermes_state_db"
    # assistant stays retrieve-only
    assistant = [e for e in store.list_events(limit=20) if e["role"] == "assistant"][0]
    assert assistant["inject_policy"] == "retrieve_only"


def test_ordinary_task_does_not_produce_a_candidate_or_active_claim(store: EventStore) -> None:
    event_id = store.append_event(
        session_id="opaque-session", conversation_id="opaque-session", platform="test",
        role="user", source="chat", content="OPAQUE_ONE_SHOT_TASK", persona_mode="work",
    )
    assert event_id > 0
    assert PersonaCandidateExtractor(store).extract(scope={"session_id": "opaque-session"}) == []
    assert store._conn.execute("SELECT COUNT(*) AS n FROM memory_claims").fetchone()["n"] == 0


@pytest.mark.parametrize(
    ("content", "expected_kind", "expected_target"),
    [
        ("我长期偏好简洁、结论先行的报告。", "user_preference", "user"),
        ("我以后喜欢深色界面。", "user_preference", "user"),
        ("我的名字是 Alice。", "identity_fact", "user"),
        ("我的职业是工程师。", "identity_fact", "user"),
        ("以后所有项目都不要自动提交代码。", "system_convention", "memory"),
        ("默认所有报告都必须附带验证证据。", "system_convention", "memory"),
    ],
)
def test_stable_natural_language_fact_auto_activates(
    store: EventStore, content: str, expected_kind: str, expected_target: str,
) -> None:
    event_id = store.append_event(
        session_id="stable-natural", conversation_id="stable-natural", platform="test",
        role="user", source="chat", content=content, persona_mode="work",
    )
    event = store.get_event(event_id)
    assert event["inject_policy"] == "candidate_only"
    candidates = PersonaCandidateExtractor(store).extract(scope={"session_id": "stable-natural"})
    assert len(candidates) == 1
    assert candidates[0]["kind"] == expected_kind
    assert candidates[0]["target_file"] == expected_target
    assert candidates[0]["memory_worthiness"] == "high"
    report = CandidatePromotionService(store).promote(candidates)
    assert report.activated == 1
    current = store._conn.execute(
        """SELECT c.memory_type, c.target, v.content, v.lineage_kind
           FROM memory_current mc
           JOIN memory_claims c ON c.claim_id=mc.claim_id
           JOIN memory_claim_versions v ON v.claim_version_id=mc.claim_version_id"""
    ).fetchone()
    assert (current["memory_type"], current["target"]) == (expected_kind, expected_target)
    assert current["content"] == content.rstrip("。")
    assert current["lineage_kind"] == "event_derived"


def test_stable_natural_language_fact_from_hermes_state_db_auto_activates(
    store: EventStore,
) -> None:
    event_id = store.append_event(
        session_id="stable-hermes-history",
        conversation_id="stable-hermes-history",
        platform="hermes",
        role="user",
        source="hermes_state_db",
        content="我长期偏好简洁报告。",
        persona_mode="work",
    )

    event = store.get_event(event_id)
    assert event["inject_policy"] == "candidate_only"
    candidates = PersonaCandidateExtractor(store).extract(
        scope={"session_id": "stable-hermes-history"}
    )
    assert len(candidates) == 1
    assert candidates[0]["kind"] == "user_preference"
    assert CandidatePromotionService(store).promote(candidates).activated == 1


@pytest.mark.parametrize(
    "content",
    [
        "你觉得我应该喜欢简洁报告吗？",
        "我今天想吃面。",
        "我可能更喜欢蓝色。",
        "刚才测试通过了。",
        "把这个文件改一下。",
        "[ASYNC DELEGATION BATCH COMPLETE] background task finished",
    ],
)
def test_transient_ambiguous_or_process_chat_never_auto_promotes(
    store: EventStore, content: str,
) -> None:
    event_id = store.append_event(
        session_id="not-stable", conversation_id="not-stable", platform="test",
        role="user", source="chat", content=content, persona_mode="work",
    )
    assert store.get_event(event_id)["inject_policy"] == "retrieve_only"
    assert PersonaCandidateExtractor(store).extract(scope={"session_id": "not-stable"}) == []


def test_stable_identity_conflict_goes_pending_without_silent_overwrite(store: EventStore) -> None:
    first_id = store.append_event(
        session_id="identity-conflict-1", conversation_id="identity-conflict-1", platform="test",
        role="user", source="chat", content="我的职业是工程师。", persona_mode="work",
    )
    first = PersonaCandidateExtractor(store).extract(scope={"session_id": "identity-conflict-1"})
    assert CandidatePromotionService(store).promote(first).activated == 1

    second_id = store.append_event(
        session_id="identity-conflict-2", conversation_id="identity-conflict-2", platform="test",
        role="user", source="chat", content="我的职业是教师。", persona_mode="work",
    )
    assert second_id > first_id
    second = PersonaCandidateExtractor(store).extract(scope={"session_id": "identity-conflict-2"})
    report = CandidatePromotionService(store).promote(second)
    assert report.pending == 1
    assert report.outcomes[0].decision == "conflict"
    current = store._conn.execute(
        """SELECT v.content FROM memory_current mc
           JOIN memory_claim_versions v ON v.claim_version_id=mc.claim_version_id"""
    ).fetchone()
    assert current["content"] == "我的职业是工程师"


def test_human_confirmation_is_a_strong_gate_even_at_high_confidence(store: EventStore, tmp_path: Path) -> None:
    _ingest(store, tmp_path, persona_mode="work")
    candidate = PersonaCandidateExtractor(store).extract(scope={"session_id": "s1"})[0]
    candidate = {**candidate, "candidate_id": "opaque-confirmation", "requires_human_confirmation": True, "confidence": 1.0}
    report = CandidatePromotionService(store).promote([candidate])
    assert report.pending == 1
    assert report.activated == 0
    assert store._conn.execute("SELECT COUNT(*) AS n FROM memory_claims").fetchone()["n"] == 0


def test_derived_claim_keeps_reopenable_event_lineage(store: EventStore, tmp_path: Path) -> None:
    _ingest(store, tmp_path, persona_mode="work")
    candidate = PersonaCandidateExtractor(store).extract(scope={"session_id": "s1"})[0]
    report = CandidatePromotionService(store).promote([candidate])
    assert report.activated == 1
    row = store._conn.execute(
        """SELECT v.lineage_kind, s.event_id, s.event_revision, s.event_payload_sha256
           FROM memory_current mc
           JOIN memory_claim_versions v ON v.claim_version_id = mc.claim_version_id
           JOIN memory_claim_sources s ON s.claim_version_id = v.claim_version_id"""
    ).fetchone()
    assert row["lineage_kind"] == "event_derived"
    source = store.get_event(int(row["event_id"]))
    assert int(row["event_revision"]) == int(source["source_revision"])
    assert row["event_payload_sha256"] == source["payload_sha256"]
    assert store._conn.execute("SELECT COUNT(*) AS n FROM events WHERE source = 'memory_assertion'").fetchone()["n"] == 0


def test_ingest_without_persona_mode_stays_retrieve_only(store: EventStore, tmp_path: Path) -> None:
    _ingest(store, tmp_path, persona_mode=None)
    user_events = [e for e in store.list_events(limit=20) if e["role"] == "user"]
    assert user_events
    assert user_events[0]["inject_policy"] == "retrieve_only"


def test_extractor_finds_hermes_state_db_candidates(store: EventStore, tmp_path: Path) -> None:
    _ingest(store, tmp_path, persona_mode="work")
    candidates = PersonaCandidateExtractor(store).extract(scope={"session_id": "s1"}, limit=50)
    assert len(candidates) == 1
    cand = candidates[0]
    assert cand["kind"] == "system_convention"
    assert cand["target_file"] == "MEMORY.md"
    assert cand["confidence"] == 0.95
    assert cand["mode"] == "work"


def test_guardrails_activate_pending_drop(store: EventStore, tmp_path: Path) -> None:
    _ingest(store, tmp_path, persona_mode="work")
    candidates = PersonaCandidateExtractor(store).extract(scope={"session_id": "s1"}, limit=50)
    base = candidates[0]
    high = dict(base)
    high["candidate_id"] = "high1"
    mid = dict(base)
    mid.update(candidate_id="mid1", content="中置信内容", confidence=0.7)
    low = dict(base)
    low.update(candidate_id="low1", content="低置信内容", confidence=0.5)

    report = CandidatePromotionService(store).promote([high, mid, low])
    assert report.activated == 1
    assert report.pending == 1
    assert report.dropped == 1
    queue = store.list_candidate_queue(status="pending")
    assert [q["content"] for q in queue] == ["中置信内容"]


def test_promotion_outcome_has_one_durable_replay_receipt(store: EventStore) -> None:
    event = _append_durable_event(store, "[memory:receipt] OPAQUE_RECEIPT_VALUE")
    candidates = PersonaCandidateExtractor(store).extract(scope={"session_id": event["session_id"]})
    service = CandidatePromotionService(store)

    original = service.promote(candidates).outcomes[0]
    replay = service.promote(candidates).outcomes[0]

    assert replay == original
    receipt = store._conn.execute(
        "SELECT * FROM candidate_promotion_receipts WHERE candidate_id=?",
        (candidates[0]["candidate_id"],),
    ).fetchone()
    assert receipt is not None
    assert receipt["decision"] == "activated"
    assert store._conn.execute("SELECT count(*) FROM candidate_promotion_receipts").fetchone()[0] == 1


def test_candidate_id_payload_drift_fails_closed_without_mutation(store: EventStore) -> None:
    event = _append_durable_event(store, "[memory:receipt] OPAQUE_RECEIPT_VALUE")
    candidate = PersonaCandidateExtractor(store).extract(scope={"session_id": event["session_id"]})[0]
    service = CandidatePromotionService(store)
    assert service.promote([candidate]).activated == 1
    before = store._conn.execute("SELECT count(*) FROM memory_claim_versions").fetchone()[0]

    drifted = {**candidate, "content": "DRIFTED_VALUE"}
    outcome = service.promote([drifted]).outcomes[0]

    assert (outcome.decision, outcome.reason) == ("error", "idempotency_conflict")
    assert store._conn.execute("SELECT count(*) FROM memory_claim_versions").fetchone()[0] == before


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("canonical_key", "persona:USER.md:user_preference:semantic:attacker"),
        ("identity_action", "unknown"),
        ("mode", "work"),
        ("sensitivity", "private"),
        ("target_file", "MEMORY.md"),
        ("kind", "system_convention"),
        ("confidence", 1.0),
    ],
)
def test_automatic_promotion_rejects_candidate_semantic_drift(
    store: EventStore, field: str, value: str,
) -> None:
    event = _append_durable_event(store, "[memory:bound] OPAQUE_BOUND_VALUE")
    candidate = PersonaCandidateExtractor(store).extract(scope={"session_id": event["session_id"]})[0]

    outcome = CandidatePromotionService(store).promote([{**candidate, field: value}]).outcomes[0]

    assert (outcome.decision, outcome.reason) == ("error", "candidate_authority_mismatch")
    assert store._conn.execute("SELECT count(*) FROM memory_claims").fetchone()[0] == 0


def test_malformed_candidate_does_not_abort_unrelated_batch_item(store: EventStore) -> None:
    event = _append_durable_event(store, "[memory:batch] OPAQUE_BATCH_VALUE")
    valid = PersonaCandidateExtractor(store).extract(scope={"session_id": event["session_id"]})[0]

    report = CandidatePromotionService(store).promote([{**valid, "candidate_id": ""}, valid])

    assert [(item.decision, item.reason) for item in report.outcomes] == [
        ("error", "malformed_candidate"), ("activated", "write_allowed"),
    ]
    assert store._conn.execute("SELECT count(*) FROM memory_claims").fetchone()[0] == 1


@pytest.mark.parametrize("candidate_id", [1, 1.5, True, None, ""])
def test_non_string_candidate_id_does_not_create_receipt_or_abort_batch(
    store: EventStore, candidate_id: object,
) -> None:
    first_event = _append_durable_event(store, "[memory:id-type] FIRST")
    malformed = PersonaCandidateExtractor(store).extract(
        scope={"session_id": first_event["session_id"]},
    )[0]
    second_event = _append_durable_event(store, "[memory:id-type-2] SECOND")
    valid = PersonaCandidateExtractor(store).extract(
        scope={"session_id": second_event["session_id"]},
    )[0]
    report = CandidatePromotionService(store).promote([
        {**malformed, "candidate_id": candidate_id}, valid,
    ])
    assert (report.outcomes[0].decision, report.outcomes[0].reason) == (
        "error", "malformed_candidate",
    )
    assert report.outcomes[1].decision == "activated"
    assert store._conn.execute("SELECT count(*) FROM candidate_promotion_receipts").fetchone()[0] == 1


@pytest.mark.parametrize("malformed", [["bad"], {"bad": 1}, 7])
def test_malformed_source_refs_do_not_abort_unrelated_batch_item(
    store: EventStore, malformed: object,
) -> None:
    event = _append_durable_event(store, "[memory:batch-ref] OPAQUE_BATCH_REF_VALUE")
    valid = PersonaCandidateExtractor(store).extract(scope={"session_id": event["session_id"]})[0]
    report = CandidatePromotionService(store).promote([
        {**valid, "candidate_id": "bad-ref", "source_refs": malformed}, valid,
    ])
    assert report.outcomes[0].reason == "malformed_candidate"
    assert report.outcomes[1].decision == "activated"


def test_non_mapping_candidate_does_not_abort_unrelated_batch_item(store: EventStore) -> None:
    event = _append_durable_event(store, "[memory:batch-map] OPAQUE_BATCH_MAP_VALUE")
    valid = PersonaCandidateExtractor(store).extract(scope={"session_id": event["session_id"]})[0]
    report = CandidatePromotionService(store).promote([["bad"], valid])
    assert report.outcomes[0].reason == "malformed_candidate"
    assert report.outcomes[1].decision == "activated"


@pytest.mark.parametrize("field", ["object_version", "payload_sha256"])
def test_forget_rejects_source_commitment_drift(store: EventStore, field: str) -> None:
    event = _append_durable_event(store, "[memory:forget-bound] BEFORE")
    candidate = PersonaCandidateExtractor(store).extract(scope={"session_id": event["session_id"]})[0]
    assert CandidatePromotionService(store).promote([candidate]).activated == 1
    forget_event = _append_durable_event(store, "[forget:forget-bound]")
    forget = PersonaCandidateExtractor(store).extract(scope={"session_id": forget_event["session_id"]})[0]
    ref = forget["source_refs"][0]
    drifted_ref = type(ref)(
        ref.authority_kind, ref.object_id,
        ref.object_version + 1 if field == "object_version" else ref.object_version,
        "0" * 64 if field == "payload_sha256" else ref.payload_sha256,
    )
    outcome = CandidatePromotionService(store).promote([
        {**forget, "source_refs": (drifted_ref,)},
    ]).outcomes[0]
    assert (outcome.decision, outcome.reason) == ("error", "candidate_authority_mismatch")
    assert store._conn.execute("SELECT lifecycle_state FROM memory_current").fetchone()[0] == "active"


def test_promotion_recovers_original_outcome_after_receipt_finalize_crash(store: EventStore) -> None:
    event = _append_durable_event(store, "[memory:crash-receipt] OPAQUE_CRASH_VALUE")
    candidate = PersonaCandidateExtractor(store).extract(scope={"session_id": event["session_id"]})[0]

    class CrashBeforeFinalize(CandidatePromotionService):
        def _finalize_receipt(self, candidate_id, request_sha256, outcome):
            raise RuntimeError("injected receipt finalize crash")

    with pytest.raises(RuntimeError, match="injected receipt finalize crash"):
        CrashBeforeFinalize(store).promote([candidate])

    processing = store._conn.execute(
        "SELECT decision FROM candidate_promotion_receipts WHERE candidate_id=?",
        (candidate["candidate_id"],),
    ).fetchone()
    assert processing["decision"] == "processing"
    assert store._conn.execute("SELECT count(*) FROM memory_claim_versions").fetchone()[0] == 1

    recovered = CandidatePromotionService(store).promote([candidate]).outcomes[0]
    assert (recovered.decision, recovered.reason) == ("activated", "write_allowed")
    assert store._conn.execute("SELECT count(*) FROM memory_claim_versions").fetchone()[0] == 1


def test_processing_receipt_reexecutes_only_through_lower_idempotency(store: EventStore) -> None:
    event = _append_durable_event(store, "[memory:owned-receipt] OPAQUE_OWNER_VALUE")
    candidate = PersonaCandidateExtractor(store).extract(scope={"session_id": event["session_id"]})[0]
    request_sha256 = CandidatePromotionService._request_sha256(candidate)
    store._conn.execute(
        """INSERT INTO candidate_promotion_receipts(
               candidate_id, request_sha256, decision, reason, target_file
           ) VALUES (?, ?, 'processing', 'in_progress', ?)""",
        (candidate["candidate_id"], request_sha256, candidate["target_file"]),
    )
    store._conn.commit()

    first = CandidatePromotionService(store).promote([candidate]).outcomes[0]
    replay = CandidatePromotionService(store).promote([candidate]).outcomes[0]

    assert replay == first
    assert (first.decision, first.reason) == ("activated", "write_allowed")
    assert store._conn.execute("SELECT count(*) FROM memory_claim_versions").fetchone()[0] == 1


def test_lost_receipt_create_race_recovers_owner_outcome(store: EventStore, monkeypatch) -> None:
    event = _append_durable_event(store, "[memory:lost-create] OPAQUE_RACE_VALUE")
    candidate = PersonaCandidateExtractor(store).extract(scope={"session_id": event["session_id"]})[0]
    service = CandidatePromotionService(store)

    class IgnoredInsert:
        rowcount = 0

    class RaceConnection:
        def __init__(self, connection):
            self.connection = connection
            self.injected = False

        def execute(self, sql, parameters=()):
            if "INSERT OR IGNORE INTO candidate_promotion_receipts" in sql and not self.injected:
                self.injected = True
                self.connection.execute(sql.replace("INSERT OR IGNORE", "INSERT"), parameters)
                return IgnoredInsert()
            return self.connection.execute(sql, parameters)

        def __getattr__(self, name):
            return getattr(self.connection, name)

    monkeypatch.setattr(store, "_conn", RaceConnection(store._conn))
    outcome = service.promote([candidate]).outcomes[0]

    assert (outcome.decision, outcome.reason) == ("activated", "write_allowed")
    assert store._conn.execute("SELECT count(*) FROM memory_claim_versions").fetchone()[0] == 1


def test_candidate_review_promotes_through_governed_event_authority(store: EventStore) -> None:
    session_id = "review-authority"
    event_id = store.append_event(
        session_id=session_id, conversation_id=session_id, platform="test",
        role="user", source="chat", content="[memory:review] OPAQUE_REVIEW_VALUE",
        persona_mode="daily", category="daily", inject_policy="candidate_only",
        classification_confidence=0.7,
    )
    event = {"event_id": event_id, "session_id": session_id}
    candidate = PersonaCandidateExtractor(store).extract(scope={"session_id": event["session_id"]})[0]
    pending = CandidatePromotionService(store).promote([candidate]).outcomes[0]
    record_id = int(pending.reason.rsplit("=", 1)[1])

    promoted = store.review_candidate(
        record_id,
        decision="approved",
        reviewer="tester",
        decision_reason="explicit human approval",
    )

    assert promoted["status"] == "promoted"
    assert store._conn.execute("SELECT count(*) FROM memory_claims").fetchone()[0] == 1
    source = store._conn.execute("SELECT source_kind, event_id FROM memory_claim_sources").fetchone()
    assert (source["source_kind"], source["event_id"]) == ("event", event["event_id"])

    replay = store.review_candidate(
        record_id, decision="approved", reviewer="tester",
        decision_reason="explicit human approval",
    )
    assert replay == promoted
    assert store._conn.execute("SELECT count(*) FROM memory_claim_versions").fetchone()[0] == 1


def test_candidate_review_rejects_queue_tampering_before_governed_write(store: EventStore) -> None:
    session_id = "review-tamper"
    store.append_event(
        session_id=session_id, conversation_id=session_id, platform="test",
        role="user", source="chat", content="[memory:review] OPAQUE_ORIGINAL_VALUE",
        persona_mode="daily", category="daily", inject_policy="candidate_only",
        classification_confidence=0.7,
    )
    candidate = PersonaCandidateExtractor(store).extract(scope={"session_id": session_id})[0]
    pending = CandidatePromotionService(store).promote([candidate]).outcomes[0]
    record_id = int(pending.reason.rsplit("=", 1)[1])
    store._conn.execute("UPDATE memory_records SET content='TAMPERED_VALUE' WHERE record_id=?", (record_id,))
    store._conn.commit()

    with pytest.raises(RuntimeError, match="queue commitment mismatch"):
        store.review_candidate(
            record_id, decision="approved", reviewer="tester",
            decision_reason="must fail closed",
        )
    assert store._conn.execute("SELECT count(*) FROM memory_claims").fetchone()[0] == 0


def test_legacy_candidate_without_reopenable_provenance_cannot_be_approved(store: EventStore) -> None:
    record_id, _ = store.add_memory_record(
        candidate_id="legacy-no-source", kind="user_preference", target_file="USER.md",
        content="OPAQUE_LEGACY_VALUE", confidence=0.7, sensitivity="normal",
        status="pending",
    )
    with pytest.raises(RuntimeError, match="reopenable event provenance"):
        store.review_candidate(
            record_id, decision="approved", reviewer="tester",
            decision_reason="must fail closed",
        )

    rejected = store.review_candidate(
        record_id, decision="rejected", reviewer="tester",
        decision_reason="explicit rejection remains supported",
    )
    assert rejected["status"] == "rejected"


def test_promoted_claim_is_recallable(store: EventStore, tmp_path: Path) -> None:
    _ingest(store, tmp_path, persona_mode="work")
    candidates = PersonaCandidateExtractor(store).extract(scope={"session_id": "s1"}, limit=50)
    report = CandidatePromotionService(store).promote(candidates)
    assert report.activated == 1
    drain_memory_projections(store, memfs_root=tmp_path / "memfs")

    result = search_governed_memories(
        store,
        GovernedMemorySearchRequest(query="OPAQUE_DURABLE_TOKEN", persona_mode=PersonaMode.WORK, limit=5),
    )
    assert result.status.value == "ok"
    assert any("OPAQUE_DURABLE_TOKEN" in item.content for item in result.items)


def test_rag_half_full_sync_turn_sequence(store: EventStore, tmp_path: Path) -> None:
    """The RAG half: promotion followed by memory projection drain must make the
    claim recallable through the governed retrieval path (what sync_turn does).

    Without the drain, search_governed_memories would report
    unavailable("projection_unavailable") even though the claim was promoted.
    """
    _ingest(store, tmp_path, persona_mode="work")
    candidates = PersonaCandidateExtractor(store).extract(scope={"session_id": "s1"}, limit=50)
    assert candidates

    # Same sequence as sync_turn: extract -> promote -> drain memory projections.
    CandidatePromotionService(store).promote(candidates)
    drain_memory_projections(store, memfs_root=tmp_path / "memfs")

    # RAG/retrieval half: governed search must now see the promoted claim.
    result = search_governed_memories(
        store,
        GovernedMemorySearchRequest(query="OPAQUE_DURABLE_TOKEN", persona_mode=PersonaMode.WORK, limit=5),
    )
    assert result.status.value == "ok", f"RAG recall failed: {result.status} {result.reason}"
    assert any("OPAQUE_DURABLE_TOKEN" in item.content for item in result.items)

    # The claim also lands in the memfs layer (memory_memfs projection), so the
    # background-memory surface can render it.
    memfs = tmp_path / "memfs"
    assert memfs.exists()
    memfs_files = list(memfs.rglob("*.md"))
    assert memfs_files, "memory_memfs projection produced no files"
    assert any("OPAQUE_DURABLE_TOKEN" in f.read_text(encoding="utf-8") for f in memfs_files)


def test_identical_content_replay_is_idempotent(store: EventStore, tmp_path: Path) -> None:
    _ingest(store, tmp_path, persona_mode="work")
    candidates = PersonaCandidateExtractor(store).extract(scope={"session_id": "s1"}, limit=50)
    service = CandidatePromotionService(store)
    assert service.promote(candidates).activated == 1
    replay = dict(candidates[0])
    replay["candidate_id"] = "replay"
    report = service.promote([replay])
    assert report.outcomes[0].decision == "duplicate"
    conn = store._conn
    rows = conn.execute("SELECT COUNT(*) AS n FROM memory_claims").fetchone()
    assert rows["n"] == 1


def test_same_semantic_key_same_content_is_duplicate(store: EventStore) -> None:
    first = _append_durable_event(store, "[memory:drink] OPAQUE_VALUE")
    first_report = CandidatePromotionService(store).promote(
        PersonaCandidateExtractor(store).extract(scope={"session_id": first["session_id"]})
    )
    second = _append_durable_event(store, "[memory:drink] OPAQUE_VALUE")
    second_report = CandidatePromotionService(store).promote(
        PersonaCandidateExtractor(store).extract(scope={"session_id": second["session_id"]})
    )

    assert first_report.activated == 1
    assert second_report.outcomes[0].decision == "duplicate"
    assert store._conn.execute("SELECT COUNT(*) AS n FROM memory_claim_versions").fetchone()["n"] == 1


def test_same_semantic_key_different_content_is_pending_conflict(store: EventStore) -> None:
    first = _append_durable_event(store, "[memory:drink] OPAQUE_VALUE_A")
    CandidatePromotionService(store).promote(
        PersonaCandidateExtractor(store).extract(scope={"session_id": first["session_id"]})
    )
    second = _append_durable_event(store, "[memory:drink] OPAQUE_VALUE_B")
    report = CandidatePromotionService(store).promote(
        PersonaCandidateExtractor(store).extract(scope={"session_id": second["session_id"]})
    )

    assert report.outcomes[0].decision == "conflict"
    assert report.pending == 1
    assert store._conn.execute("SELECT COUNT(*) AS n FROM memory_claim_versions").fetchone()["n"] == 1


def test_explicit_replace_updates_existing_semantic_identity(store: EventStore) -> None:
    first = _append_durable_event(store, "[memory:drink] OPAQUE_VALUE_A")
    CandidatePromotionService(store).promote(
        PersonaCandidateExtractor(store).extract(scope={"session_id": first["session_id"]})
    )
    second = _append_durable_event(store, "[replace:drink] OPAQUE_VALUE_B")
    report = CandidatePromotionService(store).promote(
        PersonaCandidateExtractor(store).extract(scope={"session_id": second["session_id"]})
    )

    assert report.superseded == 1
    current = store._conn.execute(
        "SELECT v.content FROM memory_current mc JOIN memory_claim_versions v ON v.claim_version_id=mc.claim_version_id"
    ).fetchone()
    assert current["content"] == "OPAQUE_VALUE_B"
    lineage = store._conn.execute(
        """SELECT v.lineage_kind, s.event_id, s.event_revision, s.event_payload_sha256
           FROM memory_current mc JOIN memory_claim_versions v ON v.claim_version_id=mc.claim_version_id
           JOIN memory_claim_sources s ON s.claim_version_id=v.claim_version_id"""
    ).fetchone()
    assert lineage["lineage_kind"] == "event_derived"
    source = store.get_event(int(lineage["event_id"]))
    assert int(lineage["event_revision"]) == int(source["source_revision"])
    assert lineage["event_payload_sha256"] == source["payload_sha256"]
    governance = store._conn.execute(
        """SELECT governance_id FROM event_governance
           WHERE event_id=? ORDER BY governance_id DESC LIMIT 1""",
        (int(lineage["event_id"]),),
    ).fetchone()
    assert governance is not None
    assert int(governance["governance_id"]) != 1


def test_explicit_replace_replay_returns_same_promotion_outcome(store: EventStore) -> None:
    first = _append_durable_event(store, "[memory:drink] OPAQUE_VALUE_A")
    CandidatePromotionService(store).promote(
        PersonaCandidateExtractor(store).extract(scope={"session_id": first["session_id"]})
    )
    second = _append_durable_event(store, "[replace:drink] OPAQUE_VALUE_B")
    candidates = PersonaCandidateExtractor(store).extract(scope={"session_id": second["session_id"]})
    service = CandidatePromotionService(store)
    original = service.promote(candidates).outcomes[0]
    replay = service.promote(candidates).outcomes[0]

    assert replay == original
    assert replay.decision == "superseded"
    assert store._conn.execute("SELECT count(*) FROM memory_claim_versions").fetchone()[0] == 2


def test_explicit_forget_retires_existing_semantic_identity(store: EventStore) -> None:
    first = _append_durable_event(store, "[memory:drink] OPAQUE_VALUE")
    CandidatePromotionService(store).promote(
        PersonaCandidateExtractor(store).extract(scope={"session_id": first["session_id"]})
    )
    second = _append_durable_event(store, "[forget:drink]")
    report = CandidatePromotionService(store).promote(
        PersonaCandidateExtractor(store).extract(scope={"session_id": second["session_id"]})
    )

    assert report.outcomes[0].decision == "retracted"
    assert store._conn.execute("SELECT lifecycle_state FROM memory_current").fetchone()["lifecycle_state"] == "retired"


def test_forget_replay_and_nan_confidence_are_fail_safe(store: EventStore) -> None:
    first = _append_durable_event(store, "[memory:drink] OPAQUE_VALUE")
    CandidatePromotionService(store).promote(
        PersonaCandidateExtractor(store).extract(scope={"session_id": first["session_id"]})
    )
    forget = _append_durable_event(store, "[forget:drink]")
    candidates = PersonaCandidateExtractor(store).extract(scope={"session_id": forget["session_id"]})
    service = CandidatePromotionService(store)
    assert service.promote(candidates).outcomes[0].decision == "retracted"
    assert service.promote(candidates).outcomes[0].decision == "retracted"

    report = service.promote([{
        "candidate_id": "nan", "kind": "preference", "target_file": "USER.md",
        "content": "OPAQUE", "confidence": float("nan"), "mode": "daily",
        "sensitivity": "normal", "identity_action": "memory",
    }])
    assert report.outcomes[0].decision == "error"


@pytest.mark.parametrize(
    "confidence",
    ["not-a-number", "0.8", "1.0", "", None, object(), True, False],
)
def test_malformed_confidence_is_a_typed_candidate_error(store: EventStore, confidence: object) -> None:
    report = CandidatePromotionService(store).promote([{
        "candidate_id": "malformed-confidence", "kind": "preference",
        "target_file": "USER.md", "content": "OPAQUE", "confidence": confidence,
        "mode": "daily", "sensitivity": "normal", "identity_action": "memory",
    }])
    assert report.outcomes[0].decision == "error"
    assert report.outcomes[0].reason == "invalid_confidence"
    assert store._conn.execute("SELECT count(*) FROM memory_claims").fetchone()[0] == 0
    assert store._conn.execute("SELECT count(*) FROM candidate_promotion_receipts").fetchone()[0] == 0


def test_automatic_candidate_without_source_refs_is_rejected(store: EventStore) -> None:
    report = CandidatePromotionService(store).promote([{
        "candidate_id": "unbound-auto-candidate", "kind": "preference",
        "target_file": "USER.md", "content": "OPAQUE", "confidence": 1.0,
        "mode": "daily", "sensitivity": "normal", "identity_action": "memory",
        "source_refs": (),
    }])
    assert report.outcomes[0].decision == "rejected"
    assert report.outcomes[0].reason == "source_snapshot_missing"
    assert store._conn.execute("SELECT count(*) FROM memory_claims").fetchone()[0] == 0
    assert store._conn.execute("SELECT count(*) FROM candidate_promotion_receipts").fetchone()[0] == 0


@pytest.mark.parametrize("missing", ["source_event_ids", "source_node_ids"])
def test_missing_queue_fields_do_not_create_receipt_or_abort_batch(
    store: EventStore, missing: str,
) -> None:
    first_event = _append_durable_event(store, "[memory:pending-shape] FIRST")
    malformed = PersonaCandidateExtractor(store).extract(
        scope={"session_id": first_event["session_id"]},
    )[0]
    malformed = {key: value for key, value in malformed.items() if key != missing}
    malformed["confidence"] = 0.7
    second_event = _append_durable_event(store, "[memory:pending-shape-2] SECOND")
    valid = PersonaCandidateExtractor(store).extract(
        scope={"session_id": second_event["session_id"]},
    )[0]

    report = CandidatePromotionService(store).promote([malformed, valid])

    assert (report.outcomes[0].decision, report.outcomes[0].reason) == (
        "error", "malformed_candidate",
    )
    assert report.outcomes[1].decision == "activated"
    receipts = store._conn.execute(
        "SELECT candidate_id FROM candidate_promotion_receipts ORDER BY candidate_id",
    ).fetchall()
    assert [row[0] for row in receipts] == [valid["candidate_id"]]


@pytest.mark.parametrize("confidence", [-0.01, 1.01])
def test_out_of_range_confidence_is_rejected(store: EventStore, confidence: float) -> None:
    report = CandidatePromotionService(store).promote([{
        "candidate_id": "out-of-range", "kind": "preference", "target_file": "USER.md",
        "content": "OPAQUE", "confidence": confidence, "mode": "daily",
        "sensitivity": "normal", "identity_action": "memory",
    }])
    assert report.outcomes[0].decision == "error"
    assert report.outcomes[0].reason == "invalid_confidence"


@pytest.mark.parametrize(
    ("field", "value"),
    [("mode", "unknown-mode"), ("sensitivity", "unknown-sensitivity")],
)
def test_unknown_candidate_enums_fail_closed(store: EventStore, field: str, value: str) -> None:
    candidate = {
        "candidate_id": "unknown-enum", "kind": "preference", "target_file": "USER.md",
        "content": "OPAQUE", "confidence": 1.0, "mode": "daily",
        "sensitivity": "normal", "identity_action": "memory",
    }
    candidate[field] = value
    report = CandidatePromotionService(store).promote([candidate])
    assert report.outcomes[0].decision == "error"
    assert report.outcomes[0].reason == f"invalid_{field}"
    assert store._conn.execute("SELECT count(*) FROM memory_claims").fetchone()[0] == 0


def test_legacy_unbound_explicit_key_replace_is_retired(store: EventStore, tmp_path: Path) -> None:
    """Unbound dictionary input cannot claim explicit-user authority."""
    from pcltm.memory_contracts import LineageKind, Sensitivity
    from pcltm.memory_write_service import MemoryWriteRequest, MemoryWriteService

    shared_key = "persona:USER.md:user_preference:preferred-drink"
    # Pre-create the first version of the claim.
    service = MemoryWriteService(store)
    first = service.write(MemoryWriteRequest(
        idempotency_key="seed",
        content="第一版偏好",
        canonical_key=shared_key,
        target="USER.md",
        memory_type="user_preference",
        sensitivity=Sensitivity.NORMAL,
        mode_scope=(PersonaMode.DAILY,),
        injection_policy="allow",
        lineage_kind=LineageKind.EXPLICIT_USER_ASSERTION,
    ))
    assert first.success

    # Same canonical key, different content, but no authority source: reject.
    base = {
        "candidate_id": "k1",
        "kind": "user_preference",
        "target_file": "USER.md",
        "content": "第二版偏好",
        "confidence": 0.95,
        "mode": "daily",
        "sensitivity": "normal",
        "canonical_key": shared_key,
        "identity_action": "replace",
    }
    report = CandidatePromotionService(store).promote([base])
    assert report.rejected == 1
    assert report.outcomes[0].reason == "source_snapshot_missing"
    conn = store._conn
    versions = conn.execute(
        "SELECT COUNT(*) AS n FROM memory_claim_versions WHERE claim_id = ?",
        (first.claim_id,),
    ).fetchone()
    assert versions["n"] == 1
    current = conn.execute(
        "SELECT v.content FROM memory_current mc "
        "JOIN memory_claim_versions v ON v.claim_version_id = mc.claim_version_id "
        "WHERE mc.claim_id = ?",
        (first.claim_id,),
    ).fetchone()
    assert current["content"] == "第一版偏好"


def test_extract_scans_newest_events_not_oldest_window(store: EventStore) -> None:
    """Regression gate: extract() must see the NEWEST candidate_only events.

    list_events() orders by event_id ASC with a bounded LIMIT; on a long session
    the newest events fall outside the ASC window and the extractor silently
    yields zero candidates (production: 54k events, sync_turn always empty).
    extract() passes order="desc" so the limit window covers the newest events.
    """
    session = "long-session"
    for i in range(60):
        store.upsert_external_event(
            external_id=f"old-{i}", source_hash=f"h{i}", kind="hermes_message",
            session_id=session, conversation_id=session, platform="desktop",
            role="assistant", source="hermes_state_db", content=f"旧内容{i}",
            persona_mode="daily", route_bucket=None, model_hint=None,
            sensitivity="normal", category="raw_conversation", subcategory="chat",
            inject_policy="retrieve_only", classification_confidence=0.2,
            classifier_version="hermes-history-v1",
        )
    store.upsert_external_event(
        external_id="new-1", source_hash="hnew", kind="hermes_message",
        session_id=session, conversation_id=session, platform="desktop",
        role="user", source="hermes_state_db", content="[memory] OPAQUE_DURABLE_TOKEN",
        persona_mode="daily", route_bucket=None, model_hint=None,
        sensitivity="normal", category="daily", subcategory="chat",
        inject_policy="candidate_only", classification_confidence=0.95,
        classifier_version="hermes-history-v1",
    )
    candidates = PersonaCandidateExtractor(store).extract(scope={"session_id": session}, limit=50)
    assert len(candidates) == 1, f"expected 1 candidate, got {len(candidates)}"
    assert candidates[0]["content"] == "OPAQUE_DURABLE_TOKEN"


def test_command_batch_folds_oldest_to_newest_for_forget(store: EventStore) -> None:
    session = "ordered-forget"
    for index, content in enumerate(("[memory:key] alpha", "[forget:key]")):
        store.upsert_external_event(
            external_id=f"forget-{index}", source_hash=f"forget-hash-{index}",
            kind="hermes_message", session_id=session, conversation_id=session,
            platform="desktop", role="user", source="hermes_state_db", content=content,
            persona_mode="daily", category="daily", inject_policy="candidate_only",
            classification_confidence=0.95,
        )
    candidates = PersonaCandidateExtractor(store).extract(scope={"session_id": session})
    report = CandidatePromotionService(store).promote(candidates)
    assert [outcome.decision for outcome in report.outcomes] == ["activated", "retracted"]
    assert store._conn.execute(
        "SELECT lifecycle_state FROM memory_current"
    ).fetchone()["lifecycle_state"] == "retired"


def test_command_batch_folds_oldest_to_newest_for_replace(store: EventStore) -> None:
    session = "ordered-replace"
    for index, content in enumerate(("[memory:key] alpha", "[replace:key] beta")):
        store.upsert_external_event(
            external_id=f"replace-{index}", source_hash=f"replace-hash-{index}",
            kind="hermes_message", session_id=session, conversation_id=session,
            platform="desktop", role="user", source="hermes_state_db", content=content,
            persona_mode="daily", category="daily", inject_policy="candidate_only",
            classification_confidence=0.95,
        )
    candidates = PersonaCandidateExtractor(store).extract(scope={"session_id": session})
    report = CandidatePromotionService(store).promote(candidates)
    assert [outcome.decision for outcome in report.outcomes] == ["activated", "superseded"]
    assert store._conn.execute(
        """SELECT v.content FROM memory_current mc
           JOIN memory_claim_versions v ON v.claim_version_id=mc.claim_version_id"""
    ).fetchone()["content"] == "beta"


@pytest.mark.parametrize("confidence", ["0.95", True, math.nan, math.inf, -0.1, 1.1])
def test_event_store_rejects_invalid_classification_confidence(
    store: EventStore, confidence: object,
) -> None:
    with pytest.raises((TypeError, ValueError), match="classification_confidence"):
        store.append_event(
            session_id="strict", conversation_id="strict", platform="desktop",
            role="user", source="chat", content="[memory:key] value",
            classification_confidence=confidence,
        )


def test_extractor_skips_event_with_missing_payload_commitment(store: EventStore) -> None:
    event_id = store.append_event(
        session_id="missing-hash", conversation_id="missing-hash", platform="desktop",
        role="user", source="chat", content="[memory:key] value", persona_mode="daily",
        category="daily", inject_policy="candidate_only", classification_confidence=0.95,
    )
    store._conn.execute("DROP TRIGGER protect_events_update")
    store._conn.execute("UPDATE events SET payload_sha256='' WHERE event_id=?", (event_id,))
    store._conn.commit()
    with pytest.raises(ValueError, match="payload commitment"):
        PersonaCandidateExtractor(store).extract(scope={"session_id": "missing-hash"})
