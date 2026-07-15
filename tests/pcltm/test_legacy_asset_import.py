from __future__ import annotations

import json
import sqlite3
from pathlib import Path

from pcltm.legacy_assets import LegacyAssetImporter
from pcltm.store import EventStore


def _source_db(path: Path) -> None:
    with sqlite3.connect(path) as conn:
        conn.executescript(
            """
            CREATE TABLE memory_records (
                record_id INTEGER PRIMARY KEY, candidate_id TEXT, kind TEXT,
                target_file TEXT, content TEXT, confidence REAL, sensitivity TEXT,
                source_event_ids TEXT, source_node_ids TEXT, status TEXT,
                reviewer TEXT, reviewed_at TEXT, decision_reason TEXT,
                patch_suggestion TEXT, metadata TEXT, created_at TEXT
            );
            CREATE TABLE short_term_events (
                short_event_id INTEGER PRIMARY KEY, session_id TEXT, conversation_id TEXT,
                platform TEXT, role TEXT, source TEXT, content TEXT, persona_mode TEXT,
                route_bucket TEXT, model_hint TEXT, sensitivity TEXT, category TEXT,
                subcategory TEXT, inject_policy TEXT, retention_policy TEXT,
                ttl_hours INTEGER, created_at TEXT
            );
            CREATE TABLE dac_raw_messages (
                raw_id INTEGER PRIMARY KEY, session_id TEXT, turn_id TEXT, role TEXT,
                content TEXT, persona_mode TEXT, source_platform TEXT, sequence INTEGER,
                token_count INTEGER, sensitivity TEXT, inject_policy TEXT,
                metadata TEXT, created_at TEXT
            );
            CREATE TABLE dac_summary_nodes (
                node_id INTEGER PRIMARY KEY, session_id TEXT, node_type TEXT, depth INTEGER,
                summary TEXT, token_count INTEGER, source_token_count INTEGER,
                source_type TEXT, source_ids TEXT, persona_mode TEXT, inject_policy TEXT,
                sensitivity TEXT, metadata TEXT, created_at TEXT, earliest_at TEXT,
                latest_at TEXT, expand_hint TEXT, status TEXT
            );
            CREATE TABLE dac_context_snapshots (
                snapshot_id INTEGER PRIMARY KEY, session_id TEXT, turn_id TEXT, mode TEXT,
                budget_tokens INTEGER, selected_node_ids TEXT, selected_raw_ids TEXT,
                fresh_tail_count INTEGER, metadata TEXT, created_at TEXT
            );
            """
        )
        memories = [
            (1, "old-1", "memory_note", "MEMORY.md", "approved unique", .9, "normal", "[]", "[]", "approved", None, None, None, None, "{}", "2026-01-01"),
            (2, "old-2", "memory_note", "MEMORY.md", "superseded body", .9, "normal", "[]", "[]", "superseded", None, None, None, None, "{}", "2026-01-02"),
            (3, "old-3", "user_profile", "USER.md", "private body", .9, "private", "[]", "[]", "approved", None, None, None, None, "{}", "2026-01-03"),
            (4, "old-4", "memory_note", "MEMORY.md", "API_TOKEN=not-for-promotion", .9, "normal", "[]", "[]", "approved", None, None, None, None, "{}", "2026-01-04"),
            (5, "old-5", "memory_note", "MEMORY.md", "already local", .9, "normal", "[]", "[]", "approved", None, None, None, None, "{}", "2026-01-05"),
        ]
        conn.executemany("INSERT INTO memory_records VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)", memories)
        conn.execute("INSERT INTO short_term_events VALUES (1,'s','c','gateway','system','compression','ttl body',NULL,NULL,NULL,'normal','x','x','no_memory','ttl',72,'2026')")
        conn.execute("INSERT INTO dac_raw_messages VALUES (1,'s','t','user','raw body','work','gateway',1,2,'normal','context_only','{}','2026')")
        conn.execute("INSERT INTO dac_summary_nodes VALUES (1,'s','leaf',0,'summary body',2,3,'short_term_events','[1]','work','retrieve_only','normal','{}','2026','2026','2026',NULL,'active')")
        conn.execute("INSERT INTO dac_context_snapshots VALUES (1,'s','t','work',100,'[1]','[1]',1,'{}','2026')")


def _local_db(path: Path) -> None:
    store = EventStore(path)
    try:
        store.add_memory_record(
            candidate_id="local-1", kind="memory_note", target_file="MEMORY.md",
            content="already local", confidence=1.0, sensitivity="normal",
            source_event_ids=[], source_node_ids=[], status="approved",
        )
    finally:
        store.close()


def test_imports_complete_archive_but_only_safe_unique_approved_candidates(tmp_path: Path) -> None:
    source = tmp_path / "server.db"
    local = tmp_path / "local.db"
    shadow = tmp_path / "shadow.db"
    _source_db(source)
    _local_db(local)

    report = LegacyAssetImporter(source, local, shadow, source_name="old-server").run()

    assert report["archived"] == 9
    assert report["candidates"] == 1
    assert report["candidate_exclusions"] == {
        "duplicate_local": 1,
        "not_approved": 1,
        "sensitive": 1,
        "secret_policy": 1,
    }
    with sqlite3.connect(shadow) as conn:
        assert conn.execute("SELECT count(*) FROM legacy_assets").fetchone()[0] == 9
        assert conn.execute("SELECT count(*) FROM legacy_promotion_candidates").fetchone()[0] == 1
        candidate = conn.execute("SELECT body,status FROM legacy_promotion_candidates").fetchone()
        assert candidate == ("approved unique", "pending_review")
        assert conn.execute("SELECT count(*) FROM legacy_assets WHERE body='private body'").fetchone()[0] == 1
        assert conn.execute("SELECT count(*) FROM legacy_assets WHERE body='API_TOKEN=not-for-promotion'").fetchone()[0] == 1
        assert conn.execute("PRAGMA quick_check").fetchone()[0] == "ok"


def test_reimport_is_idempotent_and_preserves_lineage_payload(tmp_path: Path) -> None:
    source = tmp_path / "server.db"
    local = tmp_path / "local.db"
    shadow = tmp_path / "shadow.db"
    _source_db(source)
    _local_db(local)
    importer = LegacyAssetImporter(source, local, shadow, source_name="old-server")

    first = importer.run()
    second = importer.run()

    assert first["inserted"] == 9
    assert second["inserted"] == 0
    assert second["existing"] == 9
    with sqlite3.connect(shadow) as conn:
        raw = conn.execute("SELECT payload_json FROM legacy_assets WHERE source_table='dac_summary_nodes'").fetchone()[0]
        payload = json.loads(raw)
        assert payload["source_ids"] == "[1]"
        assert payload["session_id"] == "s"


def test_shadow_archive_is_searchable_but_not_active_prompt_memory(tmp_path: Path, monkeypatch) -> None:
    from pcltm import memory_adapter

    source = tmp_path / "server.db"
    local = tmp_path / "local.db"
    shadow = tmp_path / "shadow.db"
    _source_db(source)
    _local_db(local)
    LegacyAssetImporter(source, local, shadow, source_name="old-server").run()

    with sqlite3.connect(shadow) as conn:
        rows = conn.execute(
            "SELECT a.body FROM legacy_asset_fts f JOIN legacy_assets a ON a.asset_id=f.rowid WHERE legacy_asset_fts MATCH ?",
            ('"raw body"',),
        ).fetchall()
    monkeypatch.setenv("HERMES_PCLTM_DB", str(local))
    monkeypatch.setattr(memory_adapter, "_load_system_core_entries", lambda **kwargs: [])
    monkeypatch.setattr(memory_adapter, "enabled", lambda: True)

    rendered = memory_adapter.load_prompt_context(mode="work", query="raw body")
    assert rows == [("raw body",)]
    assert "- [memory] raw body" not in rendered
    assert "- [user] raw body" not in rendered
    assert "approved unique" not in rendered
