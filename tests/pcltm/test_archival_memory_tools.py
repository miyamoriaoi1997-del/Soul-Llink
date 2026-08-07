from __future__ import annotations

from pathlib import Path

import pytest

from pcltm import memory_adapter


def _write_legacy_archival(root: Path, body: str) -> str:
    memory_id = "episodic/2026/05/legacy.md"
    path = root / memory_id
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "---\ndescription: legacy archival\nauthority: episodic\n"
        "mode_scope: [work]\nbuckets: [current_task]\n---\n"
        f"{body}\n",
        encoding="utf-8",
    )
    return memory_id


def test_archival_search_is_retired_and_does_not_expose_memfs_body(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.setattr(memory_adapter, "MEMFS_ROOT", tmp_path)
    _write_legacy_archival(tmp_path, "ARCHIVAL_SEARCH_BODY_SENTINEL")

    assert memory_adapter.search_archival_memories(
        "sentinel", mode="work", layers=["episodic"], limit=3,
    ) == []


def test_archival_open_is_retired_for_all_noncanonical_memfs_ids(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.setattr(memory_adapter, "MEMFS_ROOT", tmp_path)
    memory_id = _write_legacy_archival(tmp_path, "ARCHIVAL_OPEN_BODY_SENTINEL")

    with pytest.raises(ValueError, match="legacy_memfs_archival_open_retired"):
        memory_adapter.open_archival_memory(memory_id)
    with pytest.raises(ValueError, match="legacy_db_memory_id_retired"):
        memory_adapter.open_archival_memory("db/MEMORY.md/1")
