from __future__ import annotations

import sqlite3
from pathlib import Path

import pytest

from pcltm import memory_adapter
from pcltm.memory_adapter import load_entries, load_prompt_context, open_archival_memory, search_archival_memories


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
def dirty_memory_db(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    db = tmp_path / "pcltm.db"
    memfs_root = tmp_path / "memfs"
    fake_secret = "PASSWORD=hunter2"
    con = sqlite3.connect(db)
    con.execute(SCHEMA)
    con.execute(
        """
        INSERT INTO memory_records (
            candidate_id, kind, target_file, content, confidence, sensitivity,
            source_event_ids, source_node_ids, status, metadata
        ) VALUES ('dirty-secret', 'memory_note', 'MEMORY.md', ?, 1.0, 'normal', '[]', '[]', 'approved', '{}')
        """,
        (f"Legacy dirty record contains {fake_secret} for testing",),
    )
    con.commit()
    con.close()
    monkeypatch.setenv("HERMES_PCLTM_DB", str(db))
    monkeypatch.setattr(memory_adapter, "DEFAULT_DB", db)
    monkeypatch.setattr(memory_adapter, "MEMFS_ROOT", memfs_root)
    monkeypatch.setenv("HERMES_PCLTM_DISABLE", "0")
    monkeypatch.setenv("HERMES_PCLTM_PERSONA_VIEWS", "1")
    return db


def test_db_backed_memory_outputs_are_redacted(dirty_memory_db: Path) -> None:
    fake_secret = "PASSWORD=hunter2"

    entries = load_entries("memory")
    assert fake_secret not in "\n".join(entries)
    assert "[REDACTED_SECRET]" in "\n".join(entries)

    prompt = load_prompt_context(mode="work", query="dirty record testing")
    assert fake_secret not in prompt
    assert "[REDACTED_SECRET]" in prompt

    results = search_archival_memories("dirty record", mode="work", layers=["episodic"], limit=3)
    assert results
    assert fake_secret not in str(results)
    assert "[REDACTED_SECRET]" in str(results)

    opened = open_archival_memory("db/MEMORY.md/1")
    assert fake_secret not in str(opened)
    assert "[REDACTED_SECRET]" in str(opened)
