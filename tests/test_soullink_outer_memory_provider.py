from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest


def _load_owner_provider_class():
    from soul_link.hermes_plugin.memory_provider import SoulLinkMemoryProvider

    return SoulLinkMemoryProvider


def _load_provider_class():
    return _load_owner_provider_class()


def _configure_provider_store(monkeypatch, tmp_path: Path) -> Path:
    hermes_home = tmp_path / "hermes"
    hermes_home.mkdir()
    (hermes_home / "config.yaml").write_text(
        "plugins:\n  entries:\n    soullink: {}\n",
        encoding="utf-8",
    )
    db_path = tmp_path / "pcltm.db"
    monkeypatch.setenv("HERMES_HOME", str(hermes_home))
    monkeypatch.setenv("HERMES_PCLTM_DB", str(db_path))
    monkeypatch.setenv("HERMES_PCLTM_PERSONA_VIEWS", "1")
    return db_path


def _projected_event(db_path: Path, content: str, *, session_id: str = "source-session") -> None:
    from pcltm.projections.transcript_chunks import TranscriptChunkProjector
    from pcltm.store import EventStore

    store = EventStore(db_path)
    try:
        store.append_event(
            session_id=session_id,
            conversation_id="conversation",
            platform="desktop",
            role="user",
            source="chat",
            content=content,
            category="raw_conversation",
            subcategory="user",
            inject_policy="retrieve_only",
        )
        TranscriptChunkProjector(store, worker_id="provider-test").run_once(
            now="2026-07-25T01:00:00Z",
            lease_until="2026-07-25T01:01:00Z",
        )
    finally:
        store.close()


def test_retired_tiered_core_gate_is_not_exposed(monkeypatch, tmp_path: Path) -> None:
    hermes_home = tmp_path / "hermes"
    hermes_home.mkdir()
    monkeypatch.setenv("HERMES_HOME", str(hermes_home))
    from soul_link.hermes_plugin import memory_provider

    assert not hasattr(memory_provider, "_tiered_core_enabled")


def test_production_provider_rejects_ambient_db_outside_canonical_root(monkeypatch, tmp_path: Path) -> None:
    from soul_link.hermes_plugin import memory_provider

    hermes_home = tmp_path / "hermes"
    hermes_home.mkdir()
    production_root = hermes_home / "plugins" / "Soul-Llink"
    (production_root / "var").mkdir(parents=True)
    wrong_db = tmp_path / "stale" / "other.db"
    monkeypatch.setattr(memory_provider, "_soullink_root", lambda: tmp_path / "candidate")
    monkeypatch.setattr(memory_provider, "_hermes_home", lambda: hermes_home)
    monkeypatch.setenv("HERMES_PCLTM_DB", str(wrong_db))

    with pytest.raises(RuntimeError, match="canonical production DB"):
        memory_provider._validate_production_db_binding()


def test_provider_initialize_fails_closed_when_identity_takeover_fails(monkeypatch) -> None:
    from soul_link.hermes_plugin import memory_provider

    provider = memory_provider.SoulLinkMemoryProvider()
    monkeypatch.setattr(memory_provider, "_ensure_paths", lambda: None)
    monkeypatch.setattr(memory_provider, "_validate_production_db_binding", lambda: None)
    monkeypatch.setattr(
        memory_provider, "ensure_soullink_managed_soul",
        lambda: (_ for _ in ()).throw(PermissionError("identity anchor locked")),
    )
    router_calls = []
    monkeypatch.setattr(
        memory_provider, "ensure_inprocess_model_router",
        lambda: router_calls.append("started") or object(),
    )
    monkeypatch.setattr("pcltm.cli.init_runtime", lambda: None)

    with pytest.raises(RuntimeError, match="identity takeover failed"):
        provider.initialize("identity-fail")

    assert provider._soul_manifest["error_type"] == "PermissionError"
    assert router_calls == []


def test_event_store_read_only_open_is_byte_stable(tmp_path: Path) -> None:
    from pcltm.store import EventStore

    db_path = tmp_path / "pcltm.db"
    writer = EventStore(db_path)
    try:
        writer.append_event(
            session_id="s", conversation_id="c", platform="desktop",
            role="user", source="chat", content="read only authority needle",
        )
    finally:
        writer.close()
    before = hashlib.sha256(db_path.read_bytes()).hexdigest()

    reader = EventStore(db_path, read_only=True)
    try:
        assert reader.search_events("authority needle")
        assert reader._conn.execute("PRAGMA query_only").fetchone()[0] == 1
    finally:
        reader.close()

    assert hashlib.sha256(db_path.read_bytes()).hexdigest() == before


def test_owner_prefetch_does_not_inject_legacy_curated_or_retired_core_records(
    monkeypatch, tmp_path: Path,
) -> None:
    db_path = _configure_provider_store(monkeypatch, tmp_path)
    _projected_event(db_path, "immutable event needle from another session")
    from pcltm.store import EventStore

    store = EventStore(db_path)
    try:
        store.add_memory_record(
            candidate_id="curated-provider-test",
            kind="memory_note",
            target_file="MEMORY.md",
            content="curated continuity remains available",
            confidence=1.0,
            sensitivity="normal",
            status="approved",
        )
    finally:
        store.close()

    provider = _load_owner_provider_class()()
    provider._active_mode = "work"
    rendered = provider.prefetch("immutable needle", session_id="current-session")
    assert rendered == ""
    assert "curated continuity remains available" not in rendered
    assert "immutable event needle" not in rendered


def test_owner_prefetch_retired_core_cannot_be_restored_by_config(
    monkeypatch, tmp_path: Path,
) -> None:
    db_path = _configure_provider_store(monkeypatch, tmp_path)
    _projected_event(db_path, "rollback gate needle")
    provider = _load_owner_provider_class()()
    provider._active_mode = "work"

    assert provider.prefetch("rollback gate needle", session_id="current-session") == ""


def test_owner_prefetch_missing_canonical_store_fails_closed_without_curated_fallback(
    monkeypatch, tmp_path: Path,
) -> None:
    _configure_provider_store(monkeypatch, tmp_path)
    provider = _load_owner_provider_class()()
    provider._active_mode = "work"
    try:
        provider.prefetch("query", session_id="current-session")
    except FileNotFoundError:
        pass
    else:
        raise AssertionError("missing canonical store must not use legacy curated fallback")


def test_owner_final_forward_never_observes_retired_core_as_memory_selection(
    monkeypatch, tmp_path: Path,
) -> None:
    db_path = _configure_provider_store(monkeypatch, tmp_path)
    _projected_event(db_path, "final boundary needle")
    provider = _load_owner_provider_class()()
    provider._active_mode = "work"
    provider._runtime_capture_payload = {}
    assert provider.prefetch("final boundary needle", session_id="current-session") == ""
    assert provider._turn_memory_selection_observation["authority"] == "pcltm.memory_current"
    assert provider._turn_memory_selection_observation["status"] == "abstained"


def test_provider_wrapper_is_reserved_inside_final_memory_budget() -> None:
    from pcltm.injection.candidate import estimate_token_cost

    prefix = "<pcltm_context>\nsource: pcltm.memory_current\n"
    suffix = "\n</pcltm_context>"
    total_budget = 800

    assert total_budget - estimate_token_cost(prefix + suffix) < total_budget
    assert estimate_token_cost(prefix + ("x" * 4000) + suffix) > total_budget


def test_provider_real_governed_context_includes_wrapper_within_final_budget(monkeypatch, tmp_path: Path) -> None:
    from pcltm.injection.candidate import estimate_token_cost
    from pcltm.memory_contracts import PersonaMode, Sensitivity
    from pcltm.memory_write_service import MemoryWriteRequest, MemoryWriteService
    from pcltm.projections.memory_fts import MemoryFtsProjector
    from pcltm.store import EventStore

    db_path = _configure_provider_store(monkeypatch, tmp_path)
    store = EventStore(db_path)
    try:
        receipt = MemoryWriteService(store).write(
            MemoryWriteRequest(
                idempotency_key="provider:budget",
                content="provider final budget governed needle",
                canonical_key="provider:budget",
                target="profile",
                memory_type="preference",
                sensitivity=Sensitivity.NORMAL,
                mode_scope=(PersonaMode.WORK,),
                injection_policy="allow",
            )
        )
        assert receipt.success is True
        assert MemoryFtsProjector(store, worker_id="provider-budget").run_once(
            now="2026-08-03T00:00:00Z",
            lease_until="2026-08-03T00:01:00Z",
        )["applied"] == 1
    finally:
        store.close()

    provider = _load_owner_provider_class()()
    context = provider._load_memory_context(
        query="provider final budget governed needle",
        active_mode="work",
    )

    assert context.startswith("<pcltm_context>\nsource: pcltm.memory_current\n")
    assert context.endswith("\n</pcltm_context>")
    assert estimate_token_cost(context) <= 800


def test_owner_sync_turn_owns_projection_drain(monkeypatch, tmp_path: Path) -> None:
    _configure_provider_store(monkeypatch, tmp_path)

    provider = _load_owner_provider_class()()
    calls = []

    import pcltm.hermes_history as history
    import pcltm.projections.runtime as runtime

    monkeypatch.setattr(history.HermesHistoryIngestor, "ingest", lambda self, **kwargs: {"inserted": 1})
    monkeypatch.setattr(runtime, "drain_transcript_projections", lambda store: calls.append(store))

    provider.sync_turn("user", "assistant", session_id="session")

    assert len(calls) == 1
    assert provider._last_candidate_promotion == {
        "status": "completed", "scanned": 0, "activated": 0, "pending": 0,
        "dropped": 0, "superseded": 0, "rejected": 0, "failed": 0,
        "outcomes": [],
    }


def test_owner_sync_turn_reports_typed_promotion_without_candidate_content(
    monkeypatch, tmp_path: Path,
) -> None:
    _configure_provider_store(monkeypatch, tmp_path)
    provider = _load_owner_provider_class()()
    provider._runtime_capture_payload = {}

    from pcltm.candidate_promotion import PromotionOutcome, PromotionReport
    import pcltm.candidate_promotion as promotion
    import pcltm.candidates as candidates
    import pcltm.hermes_history as history
    import pcltm.projections.runtime as runtime

    monkeypatch.setattr(history.HermesHistoryIngestor, "ingest", lambda self, **kwargs: {"inserted": 1})
    monkeypatch.setattr(runtime, "drain_transcript_projections", lambda store: None)
    monkeypatch.setattr(candidates.PersonaCandidateExtractor, "extract", lambda self, **kwargs: [{"content": "SECRET_BODY"}])
    monkeypatch.setattr(
        promotion.CandidatePromotionService, "promote",
        lambda self, items: PromotionReport(
            scanned=1, pending=1,
            outcomes=(PromotionOutcome("opaque-id", "conflict", "semantic_identity_conflict:record=1", 7, 2, "USER.md"),),
        ),
    )

    provider.sync_turn("ignored", "ignored", session_id="session")

    report = provider._last_candidate_promotion
    assert report["outcomes"][0]["decision"] == "conflict"
    assert "SECRET_BODY" not in json.dumps(report)
    assert provider._runtime_capture_payload["candidate_promotion"] == report


def test_owner_sync_turn_does_not_promote_raw_history_into_memory_prefetch(
    monkeypatch, tmp_path: Path,
) -> None:
    db_path = _configure_provider_store(monkeypatch, tmp_path)
    state_db = tmp_path / "hermes" / "state.db"
    import sqlite3

    with sqlite3.connect(state_db) as conn:
        conn.executescript(
            """
            CREATE TABLE sessions (
                id TEXT PRIMARY KEY, source TEXT NOT NULL, parent_session_id TEXT,
                started_at REAL NOT NULL, ended_at REAL, end_reason TEXT,
                archived INTEGER NOT NULL DEFAULT 0, rewind_count INTEGER NOT NULL DEFAULT 0,
                system_prompt TEXT
            );
            CREATE TABLE messages (
                id INTEGER PRIMARY KEY, session_id TEXT NOT NULL, role TEXT NOT NULL,
                content TEXT, tool_call_id TEXT, tool_calls TEXT, tool_name TEXT,
                timestamp REAL NOT NULL, token_count INTEGER, finish_reason TEXT,
                reasoning TEXT, reasoning_content TEXT, reasoning_details TEXT,
                codex_reasoning_items TEXT, codex_message_items TEXT,
                platform_message_id TEXT, observed INTEGER DEFAULT 0,
                active INTEGER NOT NULL DEFAULT 1, compacted INTEGER NOT NULL DEFAULT 0
            );
            """
        )
        conn.execute("INSERT INTO sessions(id, source, started_at) VALUES ('s', 'desktop', 1)")
        conn.execute(
            "INSERT INTO messages(id, session_id, role, content, timestamp) VALUES (1, 's', 'user', 'sync-owned needle', 2)"
        )

    provider = _load_owner_provider_class()()
    provider.sync_turn("sync-owned needle", "answer", session_id="s")
    assert provider.prefetch("sync-owned needle", session_id="another-session") == ""


def test_outer_prefetch_mode_hint_separates_work_and_relationship_queries():
    provider = _load_provider_class()()

    assert provider._prefetch_mode_for_query("检查soullink运行情况") == "work"
    assert provider._prefetch_mode_for_query("看看情绪值") == "work"
    assert provider._prefetch_mode_for_query("恋爱时候恋爱，工作时候工作，不会互相干扰了对吗") == "work"
    assert provider._prefetch_mode_for_query("我爱你") == "daily"
    assert provider._prefetch_mode_for_query("揉揉你") == "daily"


def test_outer_prefetch_mode_hint_keeps_ambiguous_short_turn_default():
    provider = _load_provider_class()()

    assert provider._prefetch_mode_for_query("那你做吧") is None


def test_outer_prefetch_mode_hint_allows_explicit_adult_boundary_mode():
    provider = _load_provider_class()()

    assert provider._prefetch_mode_for_query("我们做爱") == "sex"


def test_provider_reuses_only_captured_recall_intent_from_same_session(monkeypatch):
    provider = _load_provider_class()()
    provider._active_mode = "work"
    provider._runtime_capture_payload = None
    calls = []
    observations = iter((
        {"context_sha256": hashlib.sha256(b"memory-1").hexdigest(), "recall_intent": {"intent": "memory_retrieval_diagnostics"}},
        {"context_sha256": hashlib.sha256(b"memory-2").hexdigest(), "recall_intent": {"intent": "memory_retrieval_diagnostics"}},
        {"context_sha256": hashlib.sha256(b"memory-3").hexdigest(), "recall_intent": {"intent": "default"}},
    ))

    def fake_load_memory_context(**kwargs):
        calls.append(kwargs)
        return f"memory-{len(calls)}"

    monkeypatch.setattr(provider, "_load_memory_context", fake_load_memory_context)
    monkeypatch.setattr(provider, "_load_memory_selection_observation", lambda: next(observations))

    provider.prefetch("优化长期记忆检索精准度", session_id="session-a")
    provider.prefetch("也就是说现在达到预期了吗", session_id="session-a")
    provider.prefetch("也就是说现在达到预期了吗", session_id="session-b")

    assert calls[0]["continuity_evidence"] is None
    assert calls[1]["continuity_evidence"].session_id == "session-a"
    assert calls[1]["continuity_evidence"].prior_intent.value == "memory_retrieval_diagnostics"
    assert calls[2]["continuity_evidence"] is None


def test_provider_rejects_stale_recall_observation(monkeypatch):
    provider = _load_provider_class()()
    provider._active_mode = "work"
    provider._runtime_capture_payload = None
    monkeypatch.setattr(provider, "_load_memory_context", lambda **kwargs: "current-memory")
    monkeypatch.setattr(
        provider,
        "_load_memory_selection_observation",
        lambda: {
            "context_sha256": hashlib.sha256(b"previous-memory").hexdigest(),
            "recall_intent": {"intent": "memory_retrieval_diagnostics"},
        },
    )

    provider.prefetch("普通当前问题", session_id="session-a")

    assert "session-a" not in provider._recall_intents_by_session


def test_provider_does_not_advance_continuity_when_load_fails(monkeypatch):
    provider = _load_provider_class()()
    provider._active_mode = "work"
    provider._runtime_capture_payload = None
    provider._recall_intents_by_session["session-a"] = "memory_retrieval_diagnostics"

    def fail(**kwargs):
        raise RuntimeError("load failed")

    monkeypatch.setattr(provider, "_load_memory_context", fail)

    try:
        provider.prefetch("切换到代码测试", session_id="session-a")
    except RuntimeError:
        pass
    else:
        raise AssertionError("expected load failure")

    assert provider._recall_intents_by_session["session-a"] == "memory_retrieval_diagnostics"


def test_owner_provider_loaded_by_installed_shim_has_session_recall_continuity(monkeypatch):
    provider = _load_owner_provider_class()()
    provider._active_mode = "work"
    provider._runtime_capture_payload = None
    provider._turn_emotion_context = ""
    calls = []

    def fake_load_memory_context(**kwargs):
        calls.append(kwargs)
        return f"owner-memory-{len(calls)}"

    observations = iter((
        {
            "context_sha256": hashlib.sha256(b"owner-memory-1").hexdigest(),
            "recall_intent": {"intent": "memory_retrieval_diagnostics"},
        },
        {
            "context_sha256": hashlib.sha256(b"owner-memory-2").hexdigest(),
            "recall_intent": {"intent": "memory_retrieval_diagnostics"},
        },
    ))
    monkeypatch.setattr(provider, "_load_memory_context", fake_load_memory_context)
    monkeypatch.setattr(provider, "_load_memory_selection_observation", lambda: next(observations))

    provider.prefetch("诊断长期记忆召回的准确性", session_id="owner-session")
    provider.prefetch("也就是说现在达到预期了吗", session_id="owner-session")

    assert calls[0]["continuity_evidence"] is None
    assert calls[1]["continuity_evidence"].session_id == "owner-session"
    assert calls[1]["continuity_evidence"].prior_intent.value == "memory_retrieval_diagnostics"
