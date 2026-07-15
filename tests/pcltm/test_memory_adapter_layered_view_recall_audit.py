from __future__ import annotations

import json
import sqlite3

from pcltm import memory_adapter


def _init_db(path):
    con = sqlite3.connect(path)
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
    return con


def _row(con: sqlite3.Connection, record_id: int, target_file: str, content: str, metadata: dict) -> None:
    con.execute(
        """
        INSERT INTO memory_records(record_id, candidate_id, kind, target_file, content, confidence, sensitivity,
                                   source_event_ids, source_node_ids, status, metadata, created_at)
        VALUES (?, ?, 'fact', ?, ?, 0.9, 'normal', '[]', '[]', 'approved', ?, '2026-01-01T00:00:00Z')
        """,
        (record_id, f"cand-{record_id}", target_file, content, json.dumps(metadata, ensure_ascii=False)),
    )


def test_layered_fallback_preserves_source_record_metadata_for_audits(tmp_path, monkeypatch):
    db = tmp_path / "pcltm.db"
    con = _init_db(db)
    _row(
        con,
        42,
        "USER.md",
        "用户希望长期记忆召回要完整命中，不要只靠短词撞中。",
        {"gov_score": 4.2, "buckets": ["memory_quality"], "tags": ["recall"], "type": "UserPreference"},
    )
    con.commit()
    con.close()

    monkeypatch.setattr(memory_adapter, "DEFAULT_DB", db)
    monkeypatch.setenv("HERMES_PCLTM_DB", str(db))
    monkeypatch.setenv("HERMES_PCLTM_MEMFS_ROOT", str(tmp_path / "missing-memfs"))
    monkeypatch.setattr(memory_adapter, "_semantic_scores_for_query", lambda *args, **kwargs: {})

    view = memory_adapter._fallback_layered_prompt_context(mode="work", query="长期记忆召回", budgets={"system": 1000, "pinned": 1500, "episodic": 1000, "transient": 500}, policy=memory_adapter.DEFAULT_VIEW_POLICY)

    item = view.pinned.items[0]
    assert item.id == "42"
    assert item.path.startswith("pinned/")
    assert item.buckets == ("memory_quality",)
    assert item.mode_scope == ("daily", "work", "sex")
    assert item.score == 4.2
    assert item.memory_type == "UserPreference"
    assert item.metadata["source"] == "db_fallback"
    assert item.metadata["target_file"] == "USER.md"
    assert item.metadata["bucket"] == "memory_quality"
    assert item.body == "用户希望长期记忆召回要完整命中，不要只靠短词撞中。"

    snapshot = view.context_selection_snapshot()
    pinned_audit = next(layer for layer in snapshot.layers if layer["layer"] == "pinned")
    assert pinned_audit["selected_items"][0]["id"] == "42"
    assert pinned_audit["selected_items"][0]["buckets"] == ["memory_quality"]
