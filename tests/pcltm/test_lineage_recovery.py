from __future__ import annotations

import json
import sqlite3
from pathlib import Path

import pytest

from pcltm.lineage_recovery import LineageRecovery, parse_legacy_time


def _db(path: Path) -> None:
    with sqlite3.connect(path) as conn:
        conn.executescript(
            """
            CREATE TABLE memory_records (
                record_id INTEGER PRIMARY KEY, content TEXT, status TEXT,
                sensitivity TEXT, created_at TEXT, source_event_ids TEXT,
                source_node_ids TEXT, metadata TEXT
            );
            CREATE TABLE dac_raw_messages (
                raw_id INTEGER PRIMARY KEY, session_id TEXT, role TEXT,
                content TEXT, created_at TEXT
            );
            CREATE TABLE short_term_events (
                short_event_id INTEGER PRIMARY KEY, session_id TEXT, role TEXT,
                source TEXT, content TEXT, created_at TEXT
            );
            CREATE TABLE dac_summary_nodes (
                node_id INTEGER PRIMARY KEY, session_id TEXT, summary TEXT,
                source_type TEXT, source_ids TEXT, created_at TEXT
            );
            """
        )
        conn.execute("INSERT INTO memory_records VALUES (1,?,?,?,?,?,?,?)", (
            "老师偏好数据库优先保存完整历史记录", "approved", "normal",
            "2026-01-03T00:00:00Z", "[]", "[]", "{}",
        ))
        conn.execute("INSERT INTO memory_records VALUES (2,?,?,?,?,?,?,?)", (
            "没有可靠来源的结论", "approved", "normal",
            "2026-01-03T00:00:00Z", "[]", "[]", "{}",
        ))
        conn.execute("INSERT INTO dac_raw_messages VALUES (10,'s1','user',?,?)", (
            "我希望数据库优先保存完整的历史记录，不要只留下精选记忆。",
            "2026-01-02T00:00:00Z",
        ))
        conn.execute("INSERT INTO short_term_events VALUES (20,'short-space-9','user','gateway',?,?)", (
            "我希望数据库优先保存完整的历史记录，不要只留下精选记忆。",
            "2026-01-02T00:00:00Z",
        ))
        conn.execute("INSERT INTO dac_summary_nodes VALUES (30,'short-space-9',?,'short_term_events','[20]',?)", (
            "用户要求数据库优先保存完整历史。", "2026-01-02T00:01:00Z",
        ))


def test_recovers_user_lineage_with_summary_chain_without_exposing_bodies(tmp_path: Path) -> None:
    db = tmp_path / "legacy.db"
    _db(db)
    recovery = LineageRecovery(db)

    result = recovery.recover(limit=30)

    assert result["eligible_count"] == 1
    evidence = result["eligible"][0]
    assert evidence["record_id"] == 1
    assert evidence["raw_ids"] == [10]
    assert evidence["short_event_ids"] == [20]
    assert evidence["summary_node_ids"] == [30]
    assert evidence["roles"] == {"user": 1}
    assert evidence["memory_sha256"]
    assert evidence["evidence_sha256"]
    serialized = json.dumps(result, ensure_ascii=False)
    assert "老师偏好" not in serialized
    assert "我希望数据库" not in serialized


def test_does_not_promote_unmatched_or_assistant_only_evidence(tmp_path: Path) -> None:
    db = tmp_path / "legacy.db"
    _db(db)
    with sqlite3.connect(db) as conn:
        conn.execute("INSERT INTO dac_raw_messages VALUES (11,'s2','assistant','没有可靠来源的结论','2026-01-02T00:00:00Z')")
    result = LineageRecovery(db).recover(limit=30)

    assert [row["record_id"] for row in result["eligible"]] == [1]
    assert result["rejected_reason_counts"]["no_user_evidence"] == 1


def test_user_corroborated_requires_high_score_and_unambiguous_margin(tmp_path: Path) -> None:
    db = tmp_path / "legacy.db"
    _db(db)
    result = LineageRecovery(db).corroborate(limit=30, minimum_score=0.60, minimum_margin=0.10)

    assert result["corroborated_count"] == 1
    row = result["corroborated"][0]
    assert row["record_id"] == 1
    assert row["raw_id"] == 10
    assert row["evidence_level"] == "user_corroborated"
    assert row["status"] == "pending_human_review"
    serialized = json.dumps(result, ensure_ascii=False)
    assert "老师偏好" not in serialized
    assert "我希望数据库" not in serialized


def test_parse_legacy_time_supports_iso_and_unix_epoch() -> None:
    iso = parse_legacy_time("2026-05-09T05:49:49.417748Z")
    epoch = parse_legacy_time("1778305789.417748")

    assert iso is not None
    assert epoch is not None
    assert iso == epoch
    assert parse_legacy_time("not-a-time") is None


def test_limit_is_bounded_and_database_is_read_only(tmp_path: Path) -> None:
    db = tmp_path / "legacy.db"
    _db(db)
    before = db.read_bytes()

    result = LineageRecovery(db).recover(limit=1)

    assert len(result["eligible"]) <= 1
    assert db.read_bytes() == before
    with pytest.raises(ValueError):
        LineageRecovery(db).recover(limit=31)
