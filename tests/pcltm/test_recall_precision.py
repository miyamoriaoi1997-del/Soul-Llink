from __future__ import annotations

import json
import sqlite3
from pathlib import Path

import pytest

from pcltm import memory_adapter
from pcltm.live_context_governor import (
    RecallContinuityEvidence,
    RecallIntent,
    classify_recall_intent,
)


SCHEMA = """
CREATE TABLE memory_records (
    record_id INTEGER PRIMARY KEY AUTOINCREMENT,
    candidate_id TEXT UNIQUE, kind TEXT, target_file TEXT, content TEXT,
    confidence REAL, sensitivity TEXT, source_event_ids TEXT, source_node_ids TEXT,
    status TEXT, metadata TEXT, created_at TEXT DEFAULT CURRENT_TIMESTAMP,
    retrieval_count INTEGER DEFAULT 0, last_retrieved_at TEXT,
    citation_count INTEGER DEFAULT 0, last_cited_at TEXT
)
"""


def insert_record(con: sqlite3.Connection, record_id: int, target: str, content: str, metadata: dict) -> None:
    con.execute(
        """INSERT INTO memory_records(record_id, candidate_id, kind, target_file, content,
           confidence, sensitivity, source_event_ids, source_node_ids, status, metadata)
           VALUES (?, ?, 'memory_note', ?, ?, 1, 'normal', '[]', '[]', 'approved', ?)""",
        (record_id, f"candidate-{record_id}", target, content, json.dumps(metadata, ensure_ascii=False)),
    )


@pytest.fixture
def recall_db(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    db = tmp_path / "pcltm.db"
    con = sqlite3.connect(db)
    con.execute(SCHEMA)
    con.commit()
    con.close()
    monkeypatch.setattr(memory_adapter, "DEFAULT_DB", db)
    monkeypatch.setenv("HERMES_PCLTM_DB", str(db))
    monkeypatch.setattr(memory_adapter, "MEMFS_ROOT", tmp_path / "memfs")
    monkeypatch.setattr(memory_adapter, "_semantic_scores_for_query", lambda *args, **kwargs: {})
    return db


@pytest.mark.parametrize(
    "query",
    (
        "诊断长期记忆召回的相关性和准确性",
        "优化长期记忆检索精准度",
        "improve long-term memory retrieval relevance and precision",
    ),
)
def test_recall_precision_intent_has_dedicated_non_relationship_bucket(query: str) -> None:
    decision = classify_recall_intent(query)

    assert decision.intent is RecallIntent.MEMORY_RETRIEVAL_DIAGNOSTICS
    assert "memory_retrieval" in decision.allowed_buckets
    assert decision.allow_user_preferences is True
    assert not {"relationship", "emotion_boundary"} & decision.allowed_buckets


def test_recall_precision_gates_unrelated_user_preferences(recall_db: Path) -> None:
    con = sqlite3.connect(recall_db)
    insert_record(con, 1, "USER.md", "用户偏好蓝色界面。", {"bucket": "user_preference"})
    insert_record(con, 2, "USER.md", "用户要求检索精准正确时宁可少召回。", {"bucket": "user_preference"})
    con.commit()
    con.close()

    rendered = memory_adapter.load_prompt_context(
        mode="work", query="优化长期记忆检索的精准和正确性", memory_limit=1000, user_limit=1000
    )

    assert "检索精准正确时宁可少召回" in rendered
    assert "蓝色界面" not in rendered


def test_unrelated_high_governor_score_cannot_outrank_or_fill_recall_precision_query(recall_db: Path) -> None:
    con = sqlite3.connect(recall_db)
    insert_record(con, 1, "MEMORY.md", "关系互动和成人层创作偏好。", {"bucket": "relationship", "gov_score": 99})
    insert_record(con, 2, "MEMORY.md", "长期记忆检索必须以精准召回和正确性为先。", {"bucket": "memory_retrieval", "gov_score": 1})
    con.commit()
    con.close()

    query = "长期记忆检索精准召回正确性"
    intent = classify_recall_intent(query)
    allowed = memory_adapter._rows_allowed_by_recall_intent(
        "MEMORY.md", memory_adapter._fetch_rows("MEMORY.md"), intent, query
    )
    ranked = memory_adapter._rank_rows("MEMORY.md", allowed, "work", query=query)

    assert [row["record_id"] for row in ranked] == [2]


def test_allowed_bucket_does_not_admit_unrelated_high_score_runtime_record(recall_db: Path) -> None:
    con = sqlite3.connect(recall_db)
    insert_record(
        con,
        1,
        "MEMORY.md",
        "Hermes Telegram gateway 使用本机代理。",
        {"bucket": "runtime_boundary", "gov_score": 100},
    )
    insert_record(
        con,
        2,
        "MEMORY.md",
        "长期记忆检索必须按相关性保障精准正确召回。",
        {"bucket": "memory_retrieval", "gov_score": 1},
    )
    con.commit()
    con.close()

    rendered = memory_adapter.load_prompt_context(
        mode="work", query="优化长期记忆检索的精准和正确性", memory_limit=2000, user_limit=1000
    )

    assert "精准正确召回" in rendered
    assert "Telegram gateway" not in rendered


def test_continuity_hint_requires_explicit_resume_signal() -> None:
    diagnostic = "优化长期记忆检索的精准和正确性"

    assert memory_adapter._continuity_query_hint(diagnostic) == diagnostic
    assert "continuity capsule" in memory_adapter._continuity_query_hint("继续恢复之前的任务")


def test_load_prompt_context_recall_precision_excludes_unrelated_domains(recall_db: Path) -> None:
    con = sqlite3.connect(recall_db)
    insert_record(con, 1, "MEMORY.md", "PCLTM 长期记忆检索需用相关性门禁保障精准正确召回。", {"bucket": "memory_retrieval", "gov_score": 1})
    insert_record(con, 2, "USER.md", "用户要求召回不相关时宁可留空。", {"bucket": "user_preference", "gov_score": 1})
    insert_record(con, 3, "MEMORY.md", "关系互动的偏好记录。", {"bucket": "relationship", "gov_score": 100})
    insert_record(con, 4, "MEMORY.md", "AI 创作的分级验证流程。", {"bucket": "workflow", "gov_score": 100})
    insert_record(con, 5, "MEMORY.md", "漫画模型的训练参数。", {"bucket": "generic", "gov_score": 100})
    con.commit()
    con.close()

    rendered = memory_adapter.load_prompt_context(
        mode="work", query="需要优化，目标是，精准召回，正确召回", memory_limit=2000, user_limit=1000
    )

    assert "相关性门禁保障精准正确召回" in rendered
    assert "不相关时宁可留空" in rendered
    assert "关系互动" not in rendered
    assert "AI 创作" not in rendered
    assert "漫画模型" not in rendered


def test_elliptical_followup_inherits_diagnostic_intent_only_with_session_evidence() -> None:
    evidence = RecallContinuityEvidence(
        prior_intent=RecallIntent.MEMORY_RETRIEVAL_DIAGNOSTICS,
        confidence=0.95,
        source="session_turn",
        session_id="session-a",
    )

    decision = classify_recall_intent(
        "这个优化符合预期吗", continuity_evidence=evidence, session_id="session-a"
    )

    assert decision.intent is RecallIntent.MEMORY_RETRIEVAL_DIAGNOSTICS
    assert decision.reason == "inherited memory retrieval diagnostics from session continuity"


def test_elliptical_followup_without_evidence_stays_default() -> None:
    decision = classify_recall_intent("也就是说现在达到预期了吗")

    assert decision.intent is RecallIntent.DEFAULT


def test_elliptical_followup_rejects_evidence_from_another_session() -> None:
    evidence = RecallContinuityEvidence(
        prior_intent=RecallIntent.MEMORY_RETRIEVAL_DIAGNOSTICS,
        confidence=0.95,
        source="session_turn",
        session_id="session-a",
    )

    decision = classify_recall_intent(
        "这个优化符合预期吗", continuity_evidence=evidence, session_id="session-b"
    )

    assert decision.intent is RecallIntent.DEFAULT


@pytest.mark.parametrize(
    ("query", "expected"),
    (
        ("我们聊聊关系吧", RecallIntent.RELATIONSHIP),
        ("Git 分支现在怎么处理", RecallIntent.GIT_WORKFLOW),
        ("修复这段代码并跑测试", RecallIntent.CODING),
        ("检查 runtime provider 状态", RecallIntent.RUNTIME_MAINTENANCE),
        ("检查 PCLTM 上下文预算链路", RecallIntent.CONTEXT_DIAGNOSTICS),
    ),
)
def test_explicit_topic_overrides_diagnostic_continuity(query: str, expected: RecallIntent) -> None:
    evidence = RecallContinuityEvidence(
        prior_intent=RecallIntent.MEMORY_RETRIEVAL_DIAGNOSTICS,
        confidence=0.95,
        source="session_turn",
        session_id="session-a",
    )

    assert classify_recall_intent(
        query, continuity_evidence=evidence, session_id="session-a"
    ).intent is expected


def test_raw_followup_retrieval_keeps_only_quality_related_records(recall_db: Path) -> None:
    con = sqlite3.connect(recall_db)
    insert_record(con, 1, "MEMORY.md", "长期记忆召回优化的相关性和精准度已通过测试。", {"bucket": "memory_retrieval"})
    insert_record(con, 2, "USER.md", "安全复审、宿主安装和运行维护记录。", {"bucket": "runtime_boundary"})
    insert_record(con, 3, "MEMORY.md", "关系互动和成人创作偏好。", {"bucket": "relationship"})
    insert_record(con, 4, "MEMORY.md", "漫画模型和 Cua Driver 的参数。", {"bucket": "generic"})
    con.commit()
    con.close()

    evidence = RecallContinuityEvidence(
        prior_intent=RecallIntent.MEMORY_RETRIEVAL_DIAGNOSTICS,
        confidence=0.95,
        source="session_turn",
        session_id="session-a",
    )
    rendered = memory_adapter.load_prompt_context(
        mode="work",
        query="也就是说现在达到预期了吗",
        continuity_evidence=evidence,
        session_id="session-a",
        memory_limit=2000,
        user_limit=1000,
    )

    assert "长期记忆召回优化" in rendered
    assert "安全复审" not in rendered
    assert "关系互动" not in rendered
    assert "漫画模型" not in rendered


def test_default_raw_query_does_not_reinject_unrelated_user_or_memory(recall_db: Path) -> None:
    con = sqlite3.connect(recall_db)
    insert_record(con, 1, "USER.md", "用户偏好安全复审和宿主安装。", {"bucket": "user_preference"})
    insert_record(con, 2, "MEMORY.md", "关系互动、成人创作和漫画模型记录。", {"bucket": "generic"})
    con.commit()
    con.close()

    rendered = memory_adapter.load_prompt_context(
        mode="work",
        query="这个优化符合预期吗",
        memory_limit=2000,
        user_limit=1000,
    )

    assert "安全复审" not in rendered
    assert "关系互动" not in rendered
    assert "漫画模型" not in rendered