from __future__ import annotations

import json
import sqlite3
from pathlib import Path

import pytest

from pcltm import memory_adapter


@pytest.fixture
def legacy_projection_fixture(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> tuple[Path, Path]:
    db = tmp_path / "pcltm.db"
    con = sqlite3.connect(db)
    try:
        con.execute(
            "CREATE TABLE memory_records (record_id INTEGER PRIMARY KEY, candidate_id TEXT, "
            "target_file TEXT, content TEXT, status TEXT)"
        )
        con.execute(
            "INSERT INTO memory_records VALUES (1, 'legacy-1', 'USER.md', ?, 'approved')",
            ("LEGACY_DB_BODY_SENTINEL",),
        )
        con.commit()
    finally:
        con.close()
    root = tmp_path / "memfs"
    path = root / "pinned" / "legacy.md"
    path.parent.mkdir(parents=True)
    path.write_text(
        "---\ndescription: legacy\nauthority: pinned\nmode_scope: [work]\n"
        "buckets: [runtime_boundary]\nmetadata:\n  record_id: 1\n---\n"
        "LEGACY_MEMFS_BODY_SENTINEL\n",
        encoding="utf-8",
    )
    monkeypatch.setenv("HERMES_PCLTM_DB", str(db))
    monkeypatch.setattr(memory_adapter, "MEMFS_ROOT", root)
    return db, root


def test_legacy_db_and_memfs_projection_are_not_prompt_authority(
    legacy_projection_fixture: tuple[Path, Path],
) -> None:
    _db, root = legacy_projection_fixture

    view = memory_adapter.load_layered_prompt_context(mode="work", root=root)

    assert view.render() == ""
    assert view.total_items == 0
    assert view.selection_source == "retired_legacy_memfs_prompt"


def test_legacy_materializer_and_live_entries_are_retired(
    legacy_projection_fixture: tuple[Path, Path],
) -> None:
    _db, root = legacy_projection_fixture
    before = {path.relative_to(root).as_posix(): path.read_bytes() for path in root.rglob("*") if path.is_file()}

    assert memory_adapter.materialize_memfs_from_approved_records() == 0
    assert memory_adapter.load_entries("user") == []

    after = {path.relative_to(root).as_posix(): path.read_bytes() for path in root.rglob("*") if path.is_file()}
    assert after == before
