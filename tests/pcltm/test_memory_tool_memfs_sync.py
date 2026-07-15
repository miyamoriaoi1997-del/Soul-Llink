from __future__ import annotations

import json
import sqlite3
from pathlib import Path
from unittest.mock import patch

import pytest

from pcltm import memory_adapter
from pcltm.memory_adapter import load_layered_prompt_context, sync_memory_tool_write


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
    created_at TEXT DEFAULT CURRENT_TIMESTAMP
)
"""


@pytest.fixture
def pcltm_paths(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> tuple[Path, Path]:
    db = tmp_path / "pcltm.db"
    memfs_root = tmp_path / "memfs"
    con = sqlite3.connect(db)
    con.execute(SCHEMA)
    con.commit()
    con.close()
    monkeypatch.setenv("HERMES_PCLTM_DB", str(db))
    monkeypatch.setattr(memory_adapter, "DEFAULT_DB", db)
    monkeypatch.setattr(memory_adapter, "MEMFS_ROOT", memfs_root)
    monkeypatch.setenv("HERMES_PCLTM_DISABLE", "0")
    monkeypatch.setenv("HERMES_PCLTM_PERSONA_VIEWS", "1")
    return db, memfs_root


def test_memory_tool_write_materializes_memfs_and_new_session_reads_it(pcltm_paths: tuple[Path, Path]) -> None:
    _, memfs_root = pcltm_paths

    ok = sync_memory_tool_write(
        "user",
        "add",
        content="User prefers cross-session memory to persist through materialized MemFS files.",
    )

    assert ok is True
    files = sorted(memfs_root.rglob("*.md"))
    assert files, "memory-tool writes must create durable MemFS files, not only DB rows"
    assert any("cross-session memory" in path.read_text(encoding="utf-8") for path in files)

    # Simulate the next session: prompt assembly reads from persisted MemFS with no
    # in-process MemoryStore state carried over.
    view = load_layered_prompt_context(
        mode="work",
        query="cross-session memory MemFS",
        layers=["system", "pinned", "episodic", "transient"],
        buckets=["user_preference"],
    )

    rendered = view.render()
    assert view.selection_source == "memfs"
    assert "cross-session memory" in rendered
    assert any(item.metadata.get("source") == "memory_tool" for item in view.pinned.items)


def test_memory_tool_rejects_raw_secret_without_legacy_fallback(pcltm_paths: tuple[Path, Path]) -> None:
    db, memfs_root = pcltm_paths
    fake_key = "sk-test_secret_fake_key_1234567890"

    ok = sync_memory_tool_write("user", "add", content=f"remember {fake_key}")

    assert ok is True
    assert not list(memfs_root.rglob("*.md"))
    con = sqlite3.connect(db)
    try:
        rows = con.execute("SELECT content FROM memory_records").fetchall()
    finally:
        con.close()
    assert rows == []


def test_memory_tool_sanitizes_secret_but_preserves_connection_metadata(pcltm_paths: tuple[Path, Path]) -> None:
    db, memfs_root = pcltm_paths

    ok = sync_memory_tool_write(
        "memory",
        "add",
        content="SSH server host=203.0.113.10 user=ubuntu password=hunter2 path=/srv/soul-link port=22",
    )

    assert ok is True
    files = sorted(memfs_root.rglob("*.md"))
    assert files
    rendered_files = "\n".join(path.read_text(encoding="utf-8") for path in files)
    assert "203.0.113.10" in rendered_files
    assert "ubuntu" in rendered_files
    assert "/srv/soul-link" in rendered_files
    assert "hunter2" not in rendered_files

    con = sqlite3.connect(db)
    con.row_factory = sqlite3.Row
    try:
        row = con.execute("SELECT content, metadata FROM memory_records").fetchone()
    finally:
        con.close()
    assert row is not None
    assert "hunter2" not in row["content"]
    metadata = json.loads(row["metadata"])
    assert metadata["sanitized_from_secret"] is True


def test_memory_tool_rolls_back_database_when_memfs_materialization_fails(
    pcltm_paths: tuple[Path, Path],
) -> None:
    db, memfs_root = pcltm_paths

    with patch.object(memory_adapter, "_materialize_memfs_record", side_effect=OSError("disk full")):
        with pytest.raises(OSError, match="disk full"):
            sync_memory_tool_write("memory", "add", content="Rollback this durable memory if materialization fails.")

    con = sqlite3.connect(db)
    try:
        rows = con.execute(
            "SELECT status, content FROM memory_records WHERE content LIKE '%Rollback this durable memory%'"
        ).fetchall()
    finally:
        con.close()

    assert rows == []
    assert not list(memfs_root.rglob("*.md"))


def test_memory_tool_rolls_back_supersede_when_memfs_removal_fails(
    pcltm_paths: tuple[Path, Path],
) -> None:
    db, _ = pcltm_paths
    content = "Preserve this memory when its materialized file cannot be removed."
    assert sync_memory_tool_write("memory", "add", content=content) is True

    with patch.object(memory_adapter, "_remove_memfs_record_file", side_effect=OSError("locked")):
        with pytest.raises(OSError, match="locked"):
            sync_memory_tool_write("memory", "remove", old_text="Preserve this memory")

    con = sqlite3.connect(db)
    try:
        status = con.execute(
            "SELECT status FROM memory_records WHERE content = ?",
            (content,),
        ).fetchone()[0]
    finally:
        con.close()

    assert status == "approved"


def test_rollback_restores_approved_memfs_frontmatter(pcltm_paths: tuple[Path, Path]) -> None:
    db, memfs_root = pcltm_paths
    content = "Restore approved frontmatter after a later materialization failure."
    assert sync_memory_tool_write("memory", "add", content=content) is True
    original = next(memfs_root.rglob("*.md"))

    real_materialize = memory_adapter._materialize_memfs_record
    calls = 0

    def fail_once_then_materialize(row):
        nonlocal calls
        calls += 1
        if calls == 1:
            raise OSError("later failure")
        return real_materialize(row)

    with patch.object(memory_adapter, "_materialize_memfs_record", side_effect=fail_once_then_materialize):
        with pytest.raises(OSError, match="later failure"):
            sync_memory_tool_write(
                "memory", "replace", old_text="Restore approved frontmatter", content="Replacement must roll back."
            )

    restored = original.read_text(encoding="utf-8")
    assert "lifecycle_state: active" in restored
    assert "status: approved" in restored
    con = sqlite3.connect(db)
    try:
        assert con.execute("SELECT status FROM memory_records WHERE content = ?", (content,)).fetchone()[0] == "approved"
    finally:
        con.close()


def test_add_remove_and_readd_reactivates_record(pcltm_paths: tuple[Path, Path]) -> None:
    db, memfs_root = pcltm_paths
    content = "A durable fact may be removed and explicitly added again."
    assert sync_memory_tool_write("memory", "add", content=content) is True
    assert sync_memory_tool_write("memory", "remove", old_text="durable fact") is True
    assert sync_memory_tool_write("memory", "add", content=content) is True

    con = sqlite3.connect(db)
    try:
        rows = con.execute("SELECT status FROM memory_records WHERE content = ?", (content,)).fetchall()
    finally:
        con.close()
    assert rows == [("approved",)]
    assert len(list(memfs_root.rglob("*.md"))) == 1


def test_commit_failure_restores_database_and_memfs(pcltm_paths: tuple[Path, Path]) -> None:
    db, memfs_root = pcltm_paths
    original_content = "Commit failure must restore this approved memory."
    assert sync_memory_tool_write("memory", "add", content=original_content) is True
    original_path = next(memfs_root.rglob("*.md"))
    real_connect = sqlite3.connect

    class FailingCommitConnection:
        def __init__(self, inner):
            object.__setattr__(self, "inner", inner)

        def __getattr__(self, name):
            return getattr(self.inner, name)

        def __setattr__(self, name, value):
            setattr(self.inner, name, value)

        def commit(self):
            raise sqlite3.OperationalError("injected commit failure")

    with patch.object(memory_adapter.sqlite3, "connect", side_effect=lambda path: FailingCommitConnection(real_connect(path))):
        with pytest.raises(sqlite3.OperationalError, match="injected commit failure"):
            sync_memory_tool_write(
                "memory", "replace", old_text="Commit failure", content="This replacement must disappear."
            )

    con = real_connect(db)
    try:
        rows = con.execute("SELECT content, status FROM memory_records ORDER BY record_id").fetchall()
    finally:
        con.close()
    assert rows == [(original_content, "approved")]
    restored = original_path.read_text(encoding="utf-8")
    assert original_content in restored
    assert "status: approved" in restored
    assert len(list(memfs_root.rglob("*.md"))) == 1


def test_repeated_add_commit_failure_preserves_existing_approved_memfs(
    pcltm_paths: tuple[Path, Path],
) -> None:
    db, memfs_root = pcltm_paths
    content = "Repeated approved add must survive a later commit failure."
    assert sync_memory_tool_write("memory", "add", content=content) is True
    original_path = next(memfs_root.rglob("*.md"))
    original_rendered = original_path.read_text(encoding="utf-8")
    real_connect = sqlite3.connect

    class FailingCommitConnection:
        def __init__(self, inner):
            object.__setattr__(self, "inner", inner)

        def __getattr__(self, name):
            return getattr(self.inner, name)

        def __setattr__(self, name, value):
            setattr(self.inner, name, value)

        def commit(self):
            raise sqlite3.OperationalError("injected repeated-add commit failure")

    with patch.object(
        memory_adapter.sqlite3,
        "connect",
        side_effect=lambda path: FailingCommitConnection(real_connect(path)),
    ):
        with pytest.raises(sqlite3.OperationalError, match="repeated-add commit failure"):
            sync_memory_tool_write("memory", "add", content=content)

    con = real_connect(db)
    try:
        rows = con.execute(
            "SELECT content, status FROM memory_records WHERE content = ?", (content,)
        ).fetchall()
    finally:
        con.close()
    assert rows == [(content, "approved")]
    assert original_path.read_text(encoding="utf-8") == original_rendered
