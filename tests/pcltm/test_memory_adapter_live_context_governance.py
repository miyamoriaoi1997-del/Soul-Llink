from __future__ import annotations

import sqlite3
from pathlib import Path

import pytest

from pcltm import memory_adapter


SCHEMA = """
CREATE TABLE memory_records (
    record_id INTEGER PRIMARY KEY AUTOINCREMENT,
    candidate_id TEXT UNIQUE,
    kind TEXT,
    target_file TEXT,
    content TEXT,
    confidence REAL,
    sensitivity TEXT,
    source_event_ids TEXT,
    source_node_ids TEXT,
    status TEXT,
    reviewer TEXT,
    reviewed_at TEXT,
    decision_reason TEXT,
    patch_suggestion TEXT,
    metadata TEXT,
    created_at TEXT DEFAULT CURRENT_TIMESTAMP,
    retrieval_count INTEGER DEFAULT 0,
    last_retrieved_at TEXT,
    citation_count INTEGER DEFAULT 0,
    last_cited_at TEXT
)
"""


@pytest.fixture
def governed_memory_db(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    db = tmp_path / "pcltm.db"
    con = sqlite3.connect(db)
    con.execute(SCHEMA)
    con.execute(
        """
        INSERT INTO memory_records (
            candidate_id, kind, target_file, content, confidence, sensitivity,
            source_event_ids, source_node_ids, status, metadata
        ) VALUES (?, 'memory_note', 'MEMORY.md', ?, 1.0, 'normal', '[]', '[]', 'approved', ?)
        """,
        (
            "runtime-long-context",
            "PCLTM runtime boundary and context budget governance " + ("低优先级历史 " * 1400),
            '{"governor_category":"runtime_boundary"}',
        ),
    )
    con.execute(
        """
        INSERT INTO memory_records (
            candidate_id, kind, target_file, content, confidence, sensitivity,
            source_event_ids, source_node_ids, status, metadata
        ) VALUES (?, 'memory_note', 'USER.md', ?, 1.0, 'normal', '[]', '[]', 'approved', ?)
        """,
        (
            "user-context-pref",
            "User prefers PCLTM-native context governance and no Hermes built-in compression.",
            '{"governor_category":"user_preference"}',
        ),
    )
    con.commit()
    con.close()
    monkeypatch.setenv("HERMES_PCLTM_DB", str(db))
    monkeypatch.setattr(memory_adapter, "DEFAULT_DB", db)
    monkeypatch.setattr(memory_adapter, "MEMFS_ROOT", tmp_path / "memfs")
    monkeypatch.setenv("HERMES_PCLTM_DISABLE", "0")
    monkeypatch.setenv("HERMES_PCLTM_PERSONA_VIEWS", "1")
    monkeypatch.setattr(memory_adapter, "_semantic_scores_for_query", lambda *args, **kwargs: {})
    return db


def test_load_prompt_context_is_governed_but_keeps_single_pcltm_envelope(governed_memory_db: Path) -> None:
    rendered = memory_adapter.load_prompt_context(
        mode="work",
        query="把当前上下文链路和PCLTM预算治理整理出来",
        memory_limit=20000,
        user_limit=200,
    )

    assert rendered.startswith("<pcltm_context>")
    assert rendered.endswith("</pcltm_context>")
    assert rendered.count("<pcltm_context>") == 1
    assert rendered.count("</pcltm_context>") == 1
    assert "【state_machine_mode】work" in rendered
    assert "【pcltm_mode】work" in rendered
    assert "【mode_sync】consistent" in rendered
    assert "【governed_memory_view】" in rendered
    assert "【recall_intent】" in rendered
    assert "context_diagnostics" in rendered
    assert len(rendered) <= memory_adapter._live_context_total_chars(20000, 200)
    telemetry = memory_adapter.last_live_context_telemetry()
    assert telemetry["within_budget"] is True
    assert telemetry["total_chars"] == len(rendered)
    assert telemetry["recall_intent"]["intent"] == "context_diagnostics"


def test_context_diagnostics_intent_filters_disallowed_user_preferences(governed_memory_db: Path) -> None:
    con = sqlite3.connect(governed_memory_db)
    con.execute(
        "UPDATE memory_records SET metadata = ? WHERE candidate_id = 'runtime-long-context'",
        ('{"governor_category":"emotion_boundary"}',),
    )
    con.execute(
        """
        INSERT INTO memory_records (
            candidate_id, kind, target_file, content, confidence, sensitivity,
            source_event_ids, source_node_ids, status, metadata
        ) VALUES ('unrelated-creative', 'memory_note', 'MEMORY.md',
                  '漫画角色母版与中文嵌字工作流', 1.0, 'normal', '[]', '[]',
                  'approved', '{"governor_category":"emotion_boundary"}')
        """
    )
    con.commit()
    con.close()

    rendered = memory_adapter.load_prompt_context(
        mode="work",
        query="检查 PCLTM 上下文预算链路",
        memory_limit=20000,
        user_limit=2000,
    )

    assert "PCLTM runtime boundary and context budget governance" in rendered
    assert "User prefers PCLTM-native context governance" not in rendered
    assert "漫画角色母版" not in rendered


def test_runtime_maintenance_intent_keeps_user_preferences_when_allowed(governed_memory_db: Path) -> None:
    con = sqlite3.connect(governed_memory_db)
    con.execute(
        "UPDATE memory_records SET metadata = ? WHERE candidate_id = 'user-context-pref'",
        ('{"governor_category":"emotion_boundary"}',),
    )
    con.commit()
    con.close()

    rendered = memory_adapter.load_prompt_context(
        mode="work",
        query="检查 SoulLink runtime provider doctor",
        memory_limit=20000,
        user_limit=2000,
    )

    assert "User prefers PCLTM-native context governance" in rendered


def test_load_prompt_context_hard_caps_uncompacted_raw_context(monkeypatch: pytest.MonkeyPatch) -> None:
    raw_context = "<pcltm_context>\n" + ("bulky runtime context\n" * 500) + "</pcltm_context>"
    monkeypatch.setattr(
        memory_adapter,
        "_render_prompt_context",
        lambda *args, **kwargs: raw_context,
    )
    monkeypatch.setattr(memory_adapter, "_fetch_rows", lambda target_file: [])
    monkeypatch.setattr(memory_adapter, "_load_system_core_entries", lambda **kwargs: ["system core"])
    monkeypatch.setattr(memory_adapter, "enabled", lambda: True)

    rendered = memory_adapter.load_prompt_context(
        mode="work",
        query="PCLTM context budget diagnostics",
        memory_limit=20000,
        user_limit=200,
    )

    assert rendered.startswith("<pcltm_context>")
    assert rendered.endswith("</pcltm_context>")
    assert rendered.count("<pcltm_context>") == 1
    assert rendered.count("</pcltm_context>") == 1
    assert "[omitted" in rendered
    assert len(rendered) <= memory_adapter._live_context_total_chars(20000, 200)
