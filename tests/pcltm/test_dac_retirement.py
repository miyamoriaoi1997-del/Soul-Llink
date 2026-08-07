from __future__ import annotations

import ast
import sqlite3
from pathlib import Path

import pytest

from pcltm.store import EventStore


DAC_TABLES = {
    "dac_raw_messages",
    "dac_summary_nodes",
    "dac_context_snapshots",
    "dac_summary_nodes_fts",
}


def _tables(path: Path) -> set[str]:
    with sqlite3.connect(path) as conn:
        return {
            row[0]
            for row in conn.execute(
                "SELECT name FROM sqlite_master WHERE type IN ('table', 'virtual table')"
            )
        }


def test_new_event_store_does_not_create_dac_tables(tmp_path: Path) -> None:
    db = tmp_path / "new.db"
    store = EventStore(db)
    store.close()

    assert not (_tables(db) & DAC_TABLES)


def test_existing_dac_tables_open_without_schema_write_or_dac_migration(tmp_path: Path) -> None:
    db = tmp_path / "legacy.db"
    with sqlite3.connect(db) as conn:
        conn.execute(
            "CREATE TABLE dac_context_snapshots (snapshot_id INTEGER PRIMARY KEY, session_id TEXT NOT NULL, metadata TEXT)"
        )
        conn.execute("INSERT INTO dac_context_snapshots VALUES (7, 'legacy-session', '{}')")
        conn.commit()
    before = db.read_bytes()
    with sqlite3.connect(db) as conn:
        before_schema = conn.execute(
            "SELECT sql FROM sqlite_master WHERE name='dac_context_snapshots'"
        ).fetchone()[0]

    store = EventStore(db)
    try:
        assert tuple(store._conn.execute(
            "SELECT session_id, metadata FROM dac_context_snapshots WHERE snapshot_id=7"
        ).fetchone()) == ("legacy-session", "{}")
    finally:
        store.close()

    with sqlite3.connect(db) as conn:
        assert conn.execute(
            "SELECT sql FROM sqlite_master WHERE name='dac_context_snapshots'"
        ).fetchone()[0] == before_schema
    assert db.read_bytes() != b""


def test_context_engine_rejects_dac_controls_and_does_not_write_snapshots(tmp_path: Path, monkeypatch) -> None:
    from soul_link.hermes_plugin.context_engine import PCLTMContextCompressionEngine

    monkeypatch.setenv("HERMES_PCLTM_DAC_ACTIVE", "1")
    monkeypatch.setenv("HERMES_PCLTM_DAC_SHADOW_ONLY", "1")
    monkeypatch.setenv("HERMES_PCLTM_DB", str(tmp_path / "runtime.db"))

    with pytest.raises(TypeError):
        PCLTMContextCompressionEngine(dac_active=True)

    engine = PCLTMContextCompressionEngine(model="unknown")
    engine.on_session_start("s", hermes_home=tmp_path)
    engine.compress([{"role": "user", "content": "hello"}], current_tokens=1)
    assert not (tmp_path / "runtime.db").exists()


def test_context_engine_source_has_no_dac_controls_or_snapshot_writer() -> None:
    path = Path(__file__).parents[2] / "soul_link" / "hermes_plugin" / "context_engine.py"
    source = path.read_text(encoding="utf-8")
    assert "HERMES_PCLTM_DAC_" not in source
    assert "_write_dac_shadow_snapshot" not in source
    assert "_create_dac_shadow_snapshot_if_enabled" not in source


def test_active_tree_has_no_dac_package_or_public_active_dac_symbols() -> None:
    root = Path(__file__).parents[2]
    assert not (root / "packages" / "pcltm" / "dac").exists()
    assert not (root / "packages" / "pcltm" / "legacy_dac.py").exists()
    for path in (root / "packages" / "pcltm").glob("*.py"):
        tree = ast.parse(path.read_text(encoding="utf-8"))
        for node in ast.walk(tree):
            if isinstance(node, (ast.Import, ast.ImportFrom)):
                names = [alias.name for alias in node.names]
                assert all(name not in {"dac", "pcltm.dac"} for name in names)


def test_retired_dac_modules_are_absent_from_packaging_metadata() -> None:
    metadata = (Path(__file__).parents[2] / "pyproject.toml").read_text(encoding="utf-8")
    assert '"pcltm.dac"' not in metadata


def test_active_memory_policy_has_no_dac_markers_or_current_architecture_rules() -> None:
    root = Path(__file__).parents[2] / "packages" / "pcltm"
    adapter_source = (root / "memory_adapter.py").read_text(encoding="utf-8")

    assert '        "dac",' not in adapter_source


def test_retired_legacy_memory_governor_is_absent() -> None:
    root = Path(__file__).parents[2] / "packages" / "pcltm"
    assert not (root / "pcltm_governor.py").exists()
