from __future__ import annotations

import json
import sqlite3
from pathlib import Path

from pcltm.cli import main


def _create_source_db(path: Path) -> None:
    with sqlite3.connect(path) as conn:
        conn.executescript(
            """
            CREATE TABLE sessions (
                id TEXT PRIMARY KEY, source TEXT NOT NULL, parent_session_id TEXT,
                started_at REAL NOT NULL, ended_at REAL, end_reason TEXT,
                archived INTEGER NOT NULL DEFAULT 0, rewind_count INTEGER NOT NULL DEFAULT 0,
                system_prompt TEXT
            );
            CREATE TABLE messages (
                id INTEGER PRIMARY KEY, session_id TEXT NOT NULL, role TEXT NOT NULL,
                content TEXT, timestamp REAL NOT NULL
            );
            INSERT INTO sessions VALUES ('s1','tui',NULL,1,NULL,NULL,0,0,'system');
            INSERT INTO messages VALUES (1,'s1','user','hello',2);
            """
        )


def test_cli_backfills_hermes_history_and_reports_json(tmp_path: Path, capsys) -> None:
    source_db = tmp_path / "state.db"
    target_db = tmp_path / "pcltm.db"
    _create_source_db(source_db)

    code = main([
        "hermes-history-ingest",
        "--source-db", str(source_db),
        "--db", str(target_db),
        "--json",
    ])
    report = json.loads(capsys.readouterr().out)

    assert code == 0
    assert report["ok"] is True
    assert report["scanned"] == 1
    assert report["inserted"] == 2
    assert report["updated"] == 0
    assert report["existing"] == 0
    assert report["source_db"] == str(source_db)
    assert report["db_path"] == str(target_db)
