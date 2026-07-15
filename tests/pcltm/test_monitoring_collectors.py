from __future__ import annotations

import hashlib
from pathlib import Path

from pcltm.cli import init_runtime
from pcltm.monitoring import collectors


def _tree_hashes(root: Path) -> dict[str, str]:
    return {
        path.relative_to(root).as_posix(): hashlib.sha256(path.read_bytes()).hexdigest()
        for path in root.rglob("*")
        if path.is_file()
    }


OPAQUE_SENTINEL = "opaque-collector-sentinel-4d2a"


def test_runtime_memory_collector_uses_read_only_snapshot(tmp_path: Path, monkeypatch) -> None:
    db = tmp_path / "runtime.db"
    memfs = tmp_path / "memfs"
    init_runtime(db_path=db, memfs_root=memfs)
    real_connect = collectors.sqlite3.connect
    calls = []

    def checked_connect(database, *args, **kwargs):
        calls.append((str(database), dict(kwargs)))
        return real_connect(database, *args, **kwargs)

    monkeypatch.setattr(collectors.sqlite3, "connect", checked_connect)
    report = collectors.collect_runtime_memory(db_path=db, memfs_root=memfs)
    assert calls and calls[0][1].get("uri") is True
    assert "mode=ro" in calls[0][0]
    assert str(db) not in calls[0][0]
    assert report["runtime"]["status"] == "healthy"
    assert report["memory"]["semantic_query_ok"] is True


def test_real_collector_does_not_modify_runtime_tree(tmp_path: Path) -> None:
    db = tmp_path / "var" / "runtime.db"
    memfs = tmp_path / "var" / "memfs"
    init_runtime(db_path=db, memfs_root=memfs)
    before = _tree_hashes(tmp_path)

    report = collectors.collect_runtime_memory(db_path=db, memfs_root=memfs)

    after = _tree_hashes(tmp_path)
    assert after == before
    assert report["runtime"]["status"] == "healthy"
    assert report["memory"]["fts_consistent"] is True
    assert report["issues"] == []


def test_missing_database_degrades_without_creating_files(tmp_path: Path) -> None:
    db = tmp_path / "missing" / "runtime.db"
    report = collectors.collect_runtime_memory(db_path=db, memfs_root=tmp_path / "memfs")
    assert report["runtime"]["status"] == "error"
    assert report["runtime"]["db_exists"] is False
    assert report["issues"][0]["code"] == "MISSING_DB"
    assert not db.exists()
    assert not db.parent.exists()
