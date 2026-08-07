from __future__ import annotations

import json
import sqlite3

from pcltm.cli import doctor_runtime, init_runtime, live_context_evidence_smoke, live_context_smoke, main
from pcltm.memfs_store import MEMFS_DIRECTORIES
from pcltm.memory_contracts import PersonaMode, Sensitivity
from pcltm.memory_write_service import MemoryWriteRequest, MemoryWriteService
from pcltm.projections.memory_fts import MemoryFtsProjector
from pcltm.runtime_paths import resolve_db_path, resolve_memfs_root
from pcltm.store import CURRENT_SCHEMA_VERSION, EventStore


def _seed_governed_memory(db, *, token: str = "governed-cli-token") -> None:
    store = EventStore(db)
    try:
        receipt = MemoryWriteService(store).write(
            MemoryWriteRequest(
                idempotency_key=f"cli:{token}",
                content=f"governed CLI memory {token}",
                canonical_key=f"cli:{token}",
                target="profile",
                memory_type="preference",
                sensitivity=Sensitivity.NORMAL,
                mode_scope=(PersonaMode.WORK,),
                injection_policy="allow",
            )
        )
        assert receipt.success is True
        outcome = MemoryFtsProjector(store, worker_id=f"cli-{token}").run_once(
            now="2026-07-31T00:00:00Z",
            lease_until="2026-07-31T00:01:00Z",
        )
        assert outcome["applied"] == 1
    finally:
        store.close()


def test_init_runtime_bootstraps_db_and_memfs(tmp_path) -> None:
    db = tmp_path / "var" / "pcltm-prod.db"
    memfs = tmp_path / "var" / "memfs"

    report = init_runtime(db_path=db, memfs_root=memfs)

    assert report["ok"] is True
    assert db.exists()
    with sqlite3.connect(db) as con:
        tables = {row[0] for row in con.execute("select name from sqlite_master where type='table'")}
    assert "events" in tables
    assert "memory_records" in tables
    for name in MEMFS_DIRECTORIES:
        assert (memfs / name).is_dir()


def test_doctor_runtime_reports_missing_then_fix_bootstraps(tmp_path) -> None:
    db = tmp_path / "var" / "pcltm-prod.db"
    memfs = tmp_path / "var" / "memfs"

    missing = doctor_runtime(db_path=db, memfs_root=memfs)
    assert missing["ok"] is False
    assert {issue["code"] for issue in missing["issues"]} == {"missing_db", "missing_memfs_directories"}

    fixed = doctor_runtime(db_path=db, memfs_root=memfs, fix=True)
    assert fixed["ok"] is True
    assert fixed["issues"] == []
    assert db.exists()
    assert all((memfs / name).is_dir() for name in MEMFS_DIRECTORIES)


def test_cli_init_json_uses_explicit_paths(tmp_path, capsys) -> None:
    db = tmp_path / "runtime" / "pcltm-prod.db"
    memfs = tmp_path / "runtime" / "memfs"

    exit_code = main(["init", "--db", str(db), "--memfs", str(memfs), "--json"])

    assert exit_code == 0
    output = json.loads(capsys.readouterr().out)
    assert output["ok"] is True
    assert output["db_path"] == str(db)
    assert output["memfs_root"] == str(memfs)


def test_cli_explicit_paths_do_not_open_unrelated_default_db(tmp_path, capsys, monkeypatch) -> None:
    db = tmp_path / "runtime" / "pcltm-prod.db"
    memfs = tmp_path / "runtime" / "memfs"
    unrelated = tmp_path / "unrelated.db"
    unrelated.write_bytes(b"not-a-sqlite-database")
    monkeypatch.setenv("HERMES_PCLTM_DB", str(unrelated))

    exit_code = main(["init", "--db", str(db), "--memfs", str(memfs), "--json"])

    assert exit_code == 0
    assert unrelated.read_bytes() == b"not-a-sqlite-database"
    output = json.loads(capsys.readouterr().out)
    assert output["db_path"] == str(db)


def test_runtime_path_env_precedence(monkeypatch, tmp_path) -> None:
    db = tmp_path / "custom.db"
    memfs = tmp_path / "custom-memfs"
    monkeypatch.setenv("HERMES_PCLTM_DB", str(db))
    monkeypatch.setenv("HERMES_PCLTM_MEMFS_ROOT", str(memfs))

    assert resolve_db_path() == db
    assert resolve_memfs_root() == memfs


def test_live_context_smoke_reports_governed_prompt_context(monkeypatch, tmp_path) -> None:
    db = tmp_path / "authority.db"
    _seed_governed_memory(db)
    monkeypatch.setenv("HERMES_PCLTM_DB", str(db))

    report = live_context_smoke(mode="work", query="governed-cli-token")

    assert report["ok"] is True
    assert report["has_pcltm_context"] is True
    assert report["single_pcltm_context"] is True
    assert report["telemetry"]["within_budget"] is True
    assert report["telemetry"]["status"] == "ok"


def test_live_context_smoke_reports_runtime_paths(monkeypatch, tmp_path) -> None:
    db = tmp_path / "pcltm-prod.db"
    memfs = tmp_path / "memfs"
    init_runtime(db_path=db, memfs_root=memfs)
    _seed_governed_memory(db, token="runtime-path-token")
    monkeypatch.setenv("HERMES_PCLTM_DB", str(db))
    monkeypatch.setenv("HERMES_PCLTM_MEMFS_ROOT", str(memfs))

    report = live_context_smoke(mode="work", query="runtime-path-token")

    assert report["db_path"] == str(db)
    assert report["schema_version"] == CURRENT_SCHEMA_VERSION
    assert report["memfs_root"] == str(memfs)


def test_cli_live_context_smoke_json(capsys, monkeypatch, tmp_path) -> None:
    db = tmp_path / "cli.db"
    _seed_governed_memory(db, token="cli-json-token")
    monkeypatch.setenv("HERMES_PCLTM_DB", str(db))

    exit_code = main(["live-context", "smoke", "--mode", "work", "--query", "cli-json-token", "--json"])

    assert exit_code == 0
    output = json.loads(capsys.readouterr().out)
    assert output["ok"] is True
    assert output["single_pcltm_context"] is True


def test_live_context_evidence_smoke_reports_capsules() -> None:
    report = live_context_evidence_smoke()

    assert report["ok"] is True
    assert report["evidence"]["capsules"] >= 1
    assert report["governed"]["within_budget"] is True
    assert report["secret_leaked"] is False
    assert report["single_pcltm_context"] is True


def test_cli_live_context_evidence_smoke_json(capsys) -> None:
    exit_code = main(["live-context", "evidence-smoke", "--json"])

    assert exit_code == 0
    output = json.loads(capsys.readouterr().out)
    assert output["ok"] is True
    assert output["secret_leaked"] is False
    assert output["single_pcltm_context"] is True
