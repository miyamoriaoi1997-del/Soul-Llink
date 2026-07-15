from __future__ import annotations

import json
import sqlite3
from pathlib import Path

from pcltm import memory_adapter


def _row(con: sqlite3.Connection, record_id: int, content: str, metadata: dict | None = None) -> None:
    con.execute(
        """
        INSERT INTO memory_records(record_id, candidate_id, kind, target_file, content, confidence, sensitivity, source_event_ids, source_node_ids, status, metadata, created_at)
        VALUES (?, ?, 'user_preference', 'USER.md', ?, 0.9, 'low', '[]', '[]', 'approved', ?, '2026-01-01T00:00:00Z')
        """,
        (record_id, f"cand-{record_id}", content, json.dumps(metadata or {}, ensure_ascii=False)),
    )


def test_query_exact_phrase_beats_high_governor_incidental_ngram(tmp_path, monkeypatch):
    db = tmp_path / "pcltm.db"
    con = sqlite3.connect(db)
    con.execute(
        """
        CREATE TABLE memory_records (
            record_id INTEGER PRIMARY KEY,
            candidate_id TEXT NOT NULL,
            kind TEXT NOT NULL,
            target_file TEXT NOT NULL,
            content TEXT NOT NULL,
            confidence REAL NOT NULL,
            sensitivity TEXT NOT NULL,
            source_event_ids TEXT NOT NULL,
            source_node_ids TEXT NOT NULL,
            status TEXT NOT NULL,
            metadata TEXT NOT NULL,
            created_at TEXT
        )
        """
    )
    _row(con, 1, "用户喜欢普通长期计划和日程安排", {"gov_score": 9.0})
    _row(con, 2, "用户要求长期记忆召回要优先匹配完整话题，而不是只靠短字词重合。", {"gov_score": 6.0})
    con.commit()
    con.close()

    monkeypatch.setattr(memory_adapter, "DEFAULT_DB", db)
    monkeypatch.setenv("HERMES_PCLTM_DB", str(db))
    monkeypatch.setattr(memory_adapter, "_semantic_scores_for_query", lambda *args, **kwargs: {})

    rows = memory_adapter._rank_rows("USER.md", memory_adapter._fetch_rows("USER.md"), "work", query="长期记忆召回")

    assert [row["record_id"] for row in rows[:2]] == [2, 1]


def test_bucket_metadata_can_make_query_relevant_without_content_hit(tmp_path, monkeypatch):
    db = tmp_path / "pcltm.db"
    con = sqlite3.connect(db)
    con.execute(
        """
        CREATE TABLE memory_records (
            record_id INTEGER PRIMARY KEY,
            candidate_id TEXT NOT NULL,
            kind TEXT NOT NULL,
            target_file TEXT NOT NULL,
            content TEXT NOT NULL,
            confidence REAL NOT NULL,
            sensitivity TEXT NOT NULL,
            source_event_ids TEXT NOT NULL,
            source_node_ids TEXT NOT NULL,
            status TEXT NOT NULL,
            metadata TEXT NOT NULL,
            created_at TEXT
        )
        """
    )
    _row(con, 1, "普通偏好：回复要清楚。", {"gov_score": 5.0})
    _row(con, 2, "读取文件后返回详细数据。", {"gov_score": 5.0, "buckets": ["emotion_boundary"], "tags": ["emotion"]})
    con.commit()
    con.close()

    monkeypatch.setattr(memory_adapter, "DEFAULT_DB", db)
    monkeypatch.setenv("HERMES_PCLTM_DB", str(db))
    monkeypatch.setattr(memory_adapter, "_semantic_scores_for_query", lambda *args, **kwargs: {})

    rows = memory_adapter._rank_rows("USER.md", memory_adapter._fetch_rows("USER.md"), "work", query="emotion_boundary")

    assert rows[0]["record_id"] == 2

def test_functional_cjk_continuation_words_do_not_swamp_topical_recall(tmp_path, monkeypatch):
    db = tmp_path / "pcltm.db"
    con = sqlite3.connect(db)
    con.execute(
        """
        CREATE TABLE memory_records (
            record_id INTEGER PRIMARY KEY,
            candidate_id TEXT NOT NULL,
            kind TEXT NOT NULL,
            target_file TEXT NOT NULL,
            content TEXT NOT NULL,
            confidence REAL,
            sensitivity TEXT,
            source_event_ids TEXT,
            source_node_ids TEXT,
            status TEXT,
            metadata TEXT,
            created_at TEXT
        )
        """
    )
    _row(
        con,
        1,
        "用户说可以继续做，但这条只是泛用任务确认，没有具体技术主题。",
        {"gov_score": 0.97, "bucket": "active_task"},
    )
    _row(
        con,
        2,
        "PCLTM 长期记忆召回需要完整短语、metadata bucket 与语义索引共同参与排序。",
        {"gov_score": 0.52, "bucket": "runtime_boundary"},
    )
    con.row_factory = sqlite3.Row
    rows = con.execute(
        """
        SELECT record_id, kind, content, metadata, created_at, NULL AS reviewed_at
        FROM memory_records
        ORDER BY record_id ASC
        """
    ).fetchall()
    con.commit()
    con.close()

    rows = memory_adapter._rank_rows(
        "MEMORY.md",
        rows,
        "work",
        query="可以，继续做，优化 PCLTM 长期记忆召回",
    )

    assert rows[0]["content"].startswith("PCLTM 长期记忆召回")


def test_synthetic_rows_do_not_consume_scores_from_default_db_semantic_index(monkeypatch):
    con = sqlite3.connect(":memory:")
    con.row_factory = sqlite3.Row
    con.execute("CREATE TABLE rows (record_id INTEGER, content TEXT, metadata TEXT)")
    con.executemany(
        "INSERT INTO rows VALUES (?, ?, ?)",
        [(1, "topical exact phrase", '{"gov_score": 0.8}'),
         (2, "unrelated row", '{"gov_score": 0.7}')],
    )
    rows = con.execute("SELECT * FROM rows").fetchall()
    monkeypatch.setattr(
        memory_adapter,
        "_semantic_scores_for_query",
        lambda *a, **k: (_ for _ in ()).throw(AssertionError("default DB index was queried")),
    )

    ranked = memory_adapter._rank_rows("USER.md", rows, "work", query="topical exact phrase")

    assert ranked[0]["record_id"] == 1

