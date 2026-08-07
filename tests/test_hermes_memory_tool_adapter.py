from __future__ import annotations

import json
import sqlite3
from pathlib import Path

import pcltm.memory_adapter as memory_adapter
import pcltm.transcript_search as transcript_search


def _load_provider_class():
    from soul_link.hermes_plugin.memory_provider import SoulLinkMemoryProvider

    return SoulLinkMemoryProvider


def test_hermes_provider_binds_canonical_governed_memory_tools(
    tmp_path: Path, monkeypatch,
) -> None:
    from pcltm.memory_retrieval import MemoryRetrievalStatus
    from pcltm.projections.memory_runtime import drain_memory_projections
    from pcltm.runtime_paths import resolve_db_path
    from pcltm.store import EventStore

    db = tmp_path / "authority.db"
    memfs = tmp_path / "memfs"
    monkeypatch.setenv("HERMES_PCLTM_DB", str(db))
    monkeypatch.setenv("HERMES_PCLTM_MEMFS_ROOT", str(memfs))
    exact_calls = []
    monkeypatch.setattr(
        transcript_search,
        "search_exact_evidence",
        lambda store, query, limit, persona_mode: exact_calls.append({
            "query": query, "limit": limit, "persona_mode": persona_mode.value,
        }) or [],
    )
    provider = _load_provider_class()()
    provider._active_mode = "work"

    assert [schema["name"] for schema in provider.get_tool_schemas()] == [
        "soullink_memory_search",
        "soullink_memory_recall_exact",
        "soullink_memory_open",
        "soullink_memory_remember",
        "soullink_identity_status",
    ]
    remembered = json.loads(provider.handle_tool_call(
        "soullink_memory_remember",
        {"content": "stable canonical authority token", "target": "memory"},
    ))
    assert remembered["success"] is True
    claim_id = remembered["claim_id"]
    assert remembered["status"] == "active"
    assert remembered["projection_status"] == "applied"
    assert remembered["recall_ready"] is True

    store = EventStore(resolve_db_path())
    try:
        assert store._conn.execute("SELECT count(*) FROM memory_claims").fetchone()[0] == 1
        assert store._conn.execute("SELECT count(*) FROM memory_records").fetchone()[0] == 0
        assert drain_memory_projections(store, memfs_root=memfs) == {
            "memory_fts": 0,
            "memory_memfs": 0,
        }
    finally:
        store.close()

    searched = json.loads(provider.handle_tool_call(
        "soullink_memory_search",
        {"query": "canonical authority token", "mode": "work"},
    ))
    assert searched["success"] is True
    assert searched["status"] == MemoryRetrievalStatus.OK.value
    assert searched["results"][0]["memory_id"] == f"claim/{claim_id}"
    assert searched["results"][0]["authority_verified"] is True

    opened = json.loads(provider.handle_tool_call(
        "soullink_memory_open",
        {"memory_id": f"claim/{claim_id}", "mode": "work"},
    ))
    assert opened["success"] is True
    assert opened["status"] == MemoryRetrievalStatus.OK.value
    assert opened["memory"]["body"] == "stable canonical authority token"

    exact = json.loads(provider.handle_tool_call(
        "soullink_memory_recall_exact", {"query": "alpha"},
    ))
    assert exact["success"] is True
    assert exact["results"] == []
    assert exact_calls == [{"query": "alpha", "limit": 8, "persona_mode": "work"}]


def test_hermes_provider_does_not_fall_back_to_legacy_memory_records(
    tmp_path: Path, monkeypatch,
) -> None:
    db = tmp_path / "authority.db"
    monkeypatch.setenv("HERMES_PCLTM_DB", str(db))
    monkeypatch.setenv("HERMES_PCLTM_MEMFS_ROOT", str(tmp_path / "memfs"))
    provider = _load_provider_class()()
    provider.handle_tool_call(
        "soullink_memory_remember",
        {"content": "canonical only token", "target": "memory"},
    )

    con = sqlite3.connect(db)
    try:
        con.execute(
            """
            INSERT INTO memory_records(
                candidate_id, kind, target_file, content, confidence, sensitivity,
                source_event_ids, source_node_ids, status, metadata
            ) VALUES ('legacy-only', 'memory_note', 'MEMORY.md', 'legacy fallback token',
                      1.0, 'normal', '[]', '[]', 'approved', '{}')
            """
        )
        con.commit()
    finally:
        con.close()

    searched = json.loads(provider.handle_tool_call(
        "soullink_memory_search", {"query": "legacy fallback token", "mode": "work"},
    ))
    assert searched == {
        "success": True,
        "status": "abstained",
        "reason": "no_answer",
        "results": [],
    }
    opened = json.loads(provider.handle_tool_call(
        "soullink_memory_open", {"memory_id": "db/MEMORY.md/1"},
    ))
    assert opened == {
        "success": True,
        "status": "abstained",
        "reason": "invalid_memory_id",
        "memory": None,
    }


def test_hermes_prefetch_injects_only_reopened_canonical_claims(
    tmp_path: Path, monkeypatch,
) -> None:
    db = tmp_path / "authority.db"
    monkeypatch.setenv("HERMES_PCLTM_DB", str(db))
    monkeypatch.setenv("HERMES_PCLTM_MEMFS_ROOT", str(tmp_path / "memfs"))
    provider = _load_provider_class()()
    remembered = json.loads(provider.handle_tool_call(
        "soullink_memory_remember",
        {"content": "prefetch governed authority token", "target": "memory"},
    ))
    assert remembered["success"] is True

    rendered = provider.prefetch("governed authority token", session_id="phase6")

    assert "<pcltm_context>" in rendered
    assert "prefetch governed authority token" in rendered
    assert "pcltm.memory_current" in rendered
    assert "memory_records" not in rendered
    observation = provider._turn_memory_selection_observation
    assert observation["status"] == "captured"
    assert observation["authority"] == "pcltm.memory_current"
    assert observation["selected_count"] == 1


def test_hermes_prefetch_abstains_without_legacy_prompt_fallback(
    tmp_path: Path, monkeypatch,
) -> None:
    db = tmp_path / "authority.db"
    monkeypatch.setenv("HERMES_PCLTM_DB", str(db))
    monkeypatch.setenv("HERMES_PCLTM_MEMFS_ROOT", str(tmp_path / "memfs"))
    provider = _load_provider_class()()
    provider.handle_tool_call(
        "soullink_memory_remember",
        {"content": "bootstrap schema", "target": "memory"},
    )
    con = sqlite3.connect(db)
    try:
        con.execute(
            """
            INSERT INTO memory_records(
                candidate_id, kind, target_file, content, confidence, sensitivity,
                source_event_ids, source_node_ids, status, metadata
            ) VALUES ('legacy-prefetch', 'memory_note', 'MEMORY.md',
                      'legacy prefetch forbidden token', 1.0, 'normal', '[]', '[]',
                      'approved', '{}')
            """
        )
        con.commit()
    finally:
        con.close()

    assert provider.prefetch("legacy prefetch forbidden token", session_id="phase6") == ""
    observation = provider._turn_memory_selection_observation
    assert observation["status"] == "abstained"
    assert observation["authority"] == "pcltm.memory_current"
    assert observation["reason"] == "no_answer"
    assert observation["selected_count"] == 0
    assert observation["selected_records"] == []


def test_hermes_open_uses_current_mode_when_tool_mode_is_omitted(
    tmp_path: Path, monkeypatch,
) -> None:
    monkeypatch.setenv("HERMES_PCLTM_DB", str(tmp_path / "authority.db"))
    monkeypatch.setenv("HERMES_PCLTM_MEMFS_ROOT", str(tmp_path / "memfs"))
    provider = _load_provider_class()()
    provider._active_mode = "work"
    remembered = json.loads(provider.handle_tool_call(
        "soullink_memory_remember",
        {"content": "work scoped open token", "target": "memory"},
    ))

    opened = json.loads(provider.handle_tool_call(
        "soullink_memory_open", {"memory_id": f"claim/{remembered['claim_id']}"},
    ))

    assert opened["status"] == "ok"
    assert opened["memory"]["body"] == "work scoped open token"


def test_hermes_remember_preserves_target_type_and_current_mode_scope(
    tmp_path: Path, monkeypatch,
) -> None:
    db = tmp_path / "authority.db"
    monkeypatch.setenv("HERMES_PCLTM_DB", str(db))
    monkeypatch.setenv("HERMES_PCLTM_MEMFS_ROOT", str(tmp_path / "memfs"))
    provider = _load_provider_class()()
    provider._active_mode = "work"
    provider.handle_tool_call(
        "soullink_memory_remember", {"content": "user type token", "target": "user"},
    )
    provider.handle_tool_call(
        "soullink_memory_remember", {"content": "memory type token", "target": "memory"},
    )

    con = sqlite3.connect(db)
    try:
        rows = con.execute(
            """
            SELECT c.target, c.memory_type, v.mode_scope
            FROM memory_claims c JOIN memory_claim_versions v USING(claim_id)
            ORDER BY c.claim_id
            """
        ).fetchall()
    finally:
        con.close()

    assert rows == [
        ("user", "user_preference", '["work"]'),
        ("memory", "memory_note", '["work"]'),
    ]


def test_provider_memory_write_hook_uses_canonical_authority_without_legacy_double_write(
    tmp_path: Path, monkeypatch,
) -> None:
    db = tmp_path / "authority.db"
    monkeypatch.setenv("HERMES_PCLTM_DB", str(db))
    monkeypatch.setenv("HERMES_PCLTM_MEMFS_ROOT", str(tmp_path / "memfs"))
    provider = _load_provider_class()()
    provider._active_mode = "work"

    provider.on_memory_write("add", "memory", "hook canonical token")

    con = sqlite3.connect(db)
    try:
        assert con.execute("SELECT count(*) FROM memory_claims").fetchone()[0] == 1
        assert con.execute("SELECT count(*) FROM memory_records").fetchone()[0] == 0
    finally:
        con.close()
