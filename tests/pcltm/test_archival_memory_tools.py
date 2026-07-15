from __future__ import annotations

import sqlite3
from pathlib import Path

import pytest

from pcltm import memory_adapter


def write_memfs_file(root: Path, rel: str, *, description: str, authority: str, mode_scope: list[str], buckets: list[str], body: str) -> None:
    path = root / rel
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "---\n"
        f"description: {description!r}\n"
        f"authority: {authority!r}\n"
        f"mode_scope: {mode_scope!r}\n"
        f"buckets: {buckets!r}\n"
        "source: test\n"
        "metadata:\n"
        "  record_id: 999\n"
        "  candidate_id: should-not-leak\n"
        "  gov_score: 0.99\n"
        "---\n"
        f"{body}\n",
        encoding="utf-8",
    )


@pytest.fixture
def memfs_root(tmp_path, monkeypatch):
    root = tmp_path / "memfs"
    monkeypatch.setattr(memory_adapter, "MEMFS_ROOT", root)
    return root


def test_archival_search_returns_short_reference_not_full_body(memfs_root):
    long_body = "recent work episode about stateful archival retrieval " + ("FULL_BODY_SHOULD_NOT_LEAK " * 50)
    write_memfs_file(
        memfs_root,
        "episodic/2026/05/stateful.md",
        description="stateful archival retrieval episode",
        authority="episodic",
        mode_scope=["work"],
        buckets=["current_task"],
        body=long_body,
    )

    results = memory_adapter.search_archival_memories(
        "stateful retrieval",
        mode="work",
        layers=["episodic"],
        buckets=["current_task"],
        limit=3,
        excerpt_chars=80,
    )

    assert len(results) == 1
    result = results[0]
    assert result["memory_id"] == "episodic/2026/05/stateful.md"
    assert result["reference_only"] is True
    assert result["layer"] == "episodic"
    assert len(result["excerpt"]) <= 80
    rendered = str(result)
    assert "record_id" not in rendered
    assert "candidate_id" not in rendered
    assert "gov_score" not in rendered
    assert rendered.count("FULL_BODY_SHOULD_NOT_LEAK") < 3


def test_archival_open_returns_full_body_only_by_explicit_id(memfs_root):
    body = "complete stateful archival memory body with details"
    write_memfs_file(
        memfs_root,
        "episodic/2026/05/open.md",
        description="Openable archival memory",
        authority="episodic",
        mode_scope=["work"],
        buckets=["current_task"],
        body=body,
    )

    opened = memory_adapter.open_archival_memory("episodic/2026/05/open.md", body_limit=4000)

    assert opened["memory_id"] == "episodic/2026/05/open.md"
    assert opened["reference_only"] is True
    assert opened["body"] == body
    rendered = str(opened)
    assert "record_id" not in rendered
    assert "candidate_id" not in rendered
    assert "gov_score" not in rendered


def test_archival_open_truncates_large_body(memfs_root):
    body = "X" * 200
    write_memfs_file(
        memfs_root,
        "episodic/2026/05/large.md",
        description="Large archival memory",
        authority="episodic",
        mode_scope=["work"],
        buckets=["current_task"],
        body=body,
    )

    opened = memory_adapter.open_archival_memory("episodic/2026/05/large.md", body_limit=50)

    assert opened["truncated"] is True
    assert len(opened["body"]) <= 51


def test_archival_search_dedupes_memfs_record_from_db_fallback(tmp_path, monkeypatch):
    root = tmp_path / "memfs"
    db = tmp_path / "pcltm.db"
    body = "shared archival memory about checkpoint licensing and VRAM"
    con = sqlite3.connect(db)
    con.execute(
        """
        CREATE TABLE memory_records (
            record_id INTEGER PRIMARY KEY,
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
            created_at TEXT
        )
        """
    )
    con.execute(
        """
        INSERT INTO memory_records (
            record_id, candidate_id, kind, target_file, content, confidence,
            sensitivity, source_event_ids, source_node_ids, status, metadata
        ) VALUES (999, 'shared-candidate', 'memory_note', 'MEMORY.md', ?, 1.0,
                  'normal', '[]', '[]', 'approved', '{}')
        """,
        (body,),
    )
    con.commit()
    con.close()
    write_memfs_file(
        root,
        "episodic/shared.md",
        description="shared archival memory",
        authority="episodic",
        mode_scope=["work"],
        buckets=["generic"],
        body=body,
    )
    monkeypatch.setattr(memory_adapter, "DEFAULT_DB", db)
    monkeypatch.setenv("HERMES_PCLTM_DB", str(db))
    monkeypatch.setattr(memory_adapter, "MEMFS_ROOT", root)
    monkeypatch.setattr(memory_adapter, "_semantic_scores_for_query", lambda *args, **kwargs: {})

    results = memory_adapter.search_archival_memories(
        "checkpoint licensing VRAM",
        mode="work",
        layers=["episodic"],
        limit=5,
    )

    assert [result["memory_id"] for result in results] == ["episodic/shared.md"]
