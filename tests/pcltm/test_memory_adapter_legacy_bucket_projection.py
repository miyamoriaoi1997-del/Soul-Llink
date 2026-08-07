from __future__ import annotations

import json
import sqlite3

from pcltm import memory_adapter


def _legacy_row(*, target_file: str, content: str, metadata: dict | None = None) -> sqlite3.Row:
    con = sqlite3.connect(":memory:")
    con.row_factory = sqlite3.Row
    con.execute(
        """
        CREATE TABLE records (
            record_id INTEGER,
            kind TEXT,
            target_file TEXT,
            content TEXT,
            metadata TEXT,
            status TEXT,
            created_at TEXT,
            reviewed_at TEXT
        )
        """
    )
    kind = "user_profile" if target_file == "USER.md" else "memory_note"
    con.execute(
        "INSERT INTO records VALUES (1, ?, ?, ?, ?, 'approved', '', '')",
        (kind, target_file, content, json.dumps(metadata or {}, ensure_ascii=False)),
    )
    row = con.execute("SELECT * FROM records").fetchone()
    con.close()
    assert row is not None
    return row


def test_legacy_user_preference_with_boundary_word_is_not_emotion_bucket():
    row = _legacy_row(
        target_file="USER.md",
        content="用户要求架构边界与运行证据必须先完成审查，再允许修改。",
    )

    assert memory_adapter._bucket_for(row, "USER.md") == "user_preference"


def test_legacy_creative_tool_preference_is_not_runtime_bucket():
    row = _legacy_row(
        target_file="USER.md",
        content="User wants local image understanding for manga reference analysis, character consistency, and prompt assistance.",
    )

    assert memory_adapter._bucket_for(row, "USER.md") == "user_preference"


def test_legacy_runtime_memory_wins_over_incidental_emotion_word():
    row = _legacy_row(
        target_file="MEMORY.md",
        content="The runtime provider stores emotion telemetry while preserving the production boundary.",
    )

    assert memory_adapter._bucket_for(row, "MEMORY.md") == "runtime_boundary"


def test_legacy_project_path_wins_over_incidental_boundary_word():
    row = _legacy_row(
        target_file="MEMORY.md",
        content="项目路径：C:\\workspace\\boundary-review；这是本地工作目录。",
    )

    assert memory_adapter._bucket_for(row, "MEMORY.md") == "project_path"


def test_explicit_emotion_bucket_remains_authoritative():
    row = _legacy_row(
        target_file="USER.md",
        content="用户偏好在亲密关系中直接表达情绪与边界。",
        metadata={"buckets": ["emotion_boundary"]},
    )

    assert memory_adapter._bucket_for(row, "USER.md") == "emotion_boundary"


def test_legacy_intimacy_preference_remains_emotion_bucket():
    row = _legacy_row(
        target_file="USER.md",
        content="用户希望亲密关系中可以坦率表达在乎与同意。",
    )

    assert memory_adapter._bucket_for(row, "USER.md") == "emotion_boundary"
