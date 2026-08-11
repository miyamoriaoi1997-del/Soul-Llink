from __future__ import annotations

import sqlite3

import pytest

import pcltm.memory_adapter as memory_adapter


def test_system_core_entries_surface_memfs_failures(tmp_path, monkeypatch):
    root = tmp_path / "memfs"
    root.mkdir()
    monkeypatch.setattr(memory_adapter, "_memfs_root", lambda: root)

    class BrokenStore:
        def __init__(self, path):
            del path

        def load_layer(self, *args, **kwargs):
            del args, kwargs
            raise ValueError("broken projection")

    import pcltm.memfs_store as memfs_store

    monkeypatch.setattr(memfs_store, "MemFSStore", BrokenStore)
    with pytest.raises(RuntimeError, match="MemFS system layer unavailable"):
        memory_adapter._load_system_core_entries(mode="work", query="q")


def test_retrieval_stats_surface_sqlite_failures(tmp_path, monkeypatch):
    db = tmp_path / "authority.db"
    db.touch()
    monkeypatch.setattr(memory_adapter, "db_path", lambda: db)
    monkeypatch.setattr(
        sqlite3,
        "connect",
        lambda path: (_ for _ in ()).throw(sqlite3.OperationalError("locked")),
    )

    with pytest.raises(RuntimeError, match="memory retrieval stats update failed"):
        memory_adapter._update_retrieval_stats([1])


def test_citation_tracking_surfaces_sqlite_failures(tmp_path, monkeypatch):
    db = tmp_path / "authority.db"
    db.touch()
    monkeypatch.setattr(memory_adapter, "db_path", lambda: db)
    monkeypatch.setattr(
        sqlite3,
        "connect",
        lambda path: (_ for _ in ()).throw(sqlite3.OperationalError("locked")),
    )

    with pytest.raises(RuntimeError, match="memory citation tracking failed"):
        memory_adapter.track_citations("response text", [1])
