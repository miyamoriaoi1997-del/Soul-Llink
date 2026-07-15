from __future__ import annotations

import sqlite3
import threading
import time
from pathlib import Path

from pcltm.monitoring.sqlite_snapshot import stable_sqlite_snapshot


def _create_wal_db(path: Path) -> sqlite3.Connection:
    connection = sqlite3.connect(path, isolation_level=None, check_same_thread=False)
    connection.execute("PRAGMA journal_mode=WAL")
    connection.execute("PRAGMA wal_autocheckpoint=0")
    connection.execute("CREATE TABLE records(value INTEGER)")
    connection.execute("INSERT INTO records VALUES (1)")
    return connection


def test_snapshot_reads_committed_wal_without_touching_source_files(tmp_path: Path) -> None:
    db = tmp_path / "live.db"
    writer = _create_wal_db(db)
    before = {p.name: p.read_bytes() for p in tmp_path.glob("live.db*")}

    with stable_sqlite_snapshot(db) as snapshot:
        reader = sqlite3.connect(f"{snapshot.as_uri()}?mode=ro", uri=True)
        try:
            assert reader.execute("SELECT value FROM records").fetchall() == [(1,)]
        finally:
            reader.close()

    after = {p.name: p.read_bytes() for p in tmp_path.glob("live.db*")}
    writer.close()
    assert after == before


def test_snapshot_retries_when_source_changes_during_copy(tmp_path: Path, monkeypatch) -> None:
    db = tmp_path / "live.db"
    writer = _create_wal_db(db)
    import pcltm.monitoring.sqlite_snapshot as module

    original_copy = module.shutil.copy2
    copied_once = threading.Event()
    release_copy = threading.Event()
    calls = 0

    def racing_copy(source, target, *args, **kwargs):
        nonlocal calls
        result = original_copy(source, target, *args, **kwargs)
        calls += 1
        if calls == 1:
            copied_once.set()
            release_copy.wait(timeout=2)
        return result

    monkeypatch.setattr(module.shutil, "copy2", racing_copy)

    def commit_during_copy() -> None:
        assert copied_once.wait(timeout=2)
        writer.execute("INSERT INTO records VALUES (2)")
        release_copy.set()

    thread = threading.Thread(target=commit_during_copy)
    thread.start()
    with stable_sqlite_snapshot(db, max_attempts=3) as snapshot:
        reader = sqlite3.connect(f"{snapshot.as_uri()}?mode=ro", uri=True)
        try:
            assert reader.execute("SELECT value FROM records ORDER BY value").fetchall() == [(1,), (2,)]
        finally:
            reader.close()
    thread.join(timeout=2)
    writer.close()
    assert calls >= 3
