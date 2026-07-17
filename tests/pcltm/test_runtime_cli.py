from __future__ import annotations

import json
import sqlite3

from pcltm.cli import doctor_runtime, init_runtime, live_context_evidence_smoke, live_context_smoke, main
from pcltm.memfs_store import MEMFS_DIRECTORIES
from pcltm.runtime_paths import resolve_db_path, resolve_memfs_root


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


def test_runtime_path_env_precedence(monkeypatch, tmp_path) -> None:
    db = tmp_path / "custom.db"
    memfs = tmp_path / "custom-memfs"
    monkeypatch.setenv("HERMES_PCLTM_DB", str(db))
    monkeypatch.setenv("HERMES_PCLTM_MEMFS_ROOT", str(memfs))

    assert resolve_db_path() == db
    assert resolve_memfs_root() == memfs


def test_live_context_smoke_reports_governed_prompt_context(monkeypatch) -> None:
    monkeypatch.setattr("pcltm.cli.load_prompt_context", lambda **kwargs: "<pcltm_context>\nbody\n</pcltm_context>")
    monkeypatch.setattr(
        "pcltm.cli.last_live_context_telemetry",
        lambda: {
            "within_budget": True,
            "total_chars": 36,
            "limit_chars": 900,
            "omitted_chars": 0,
            "actions": [],
            "capsules": {"continuation": 0, "tool_evidence": 0},
            "recall_intent": {"intent": "context_diagnostics"},
        },
    )

    report = live_context_smoke(mode="work", query="PCLTM context budget")

    assert report["ok"] is True
    assert report["has_pcltm_context"] is True
    assert report["single_pcltm_context"] is True
    assert report["telemetry"]["within_budget"] is True
    assert report["telemetry"]["recall_intent"]["intent"] == "context_diagnostics"


def test_live_context_smoke_reports_runtime_paths(monkeypatch, tmp_path) -> None:
    db = tmp_path / "pcltm-prod.db"
    memfs = tmp_path / "memfs"
    init_runtime(db_path=db, memfs_root=memfs)
    monkeypatch.setenv("HERMES_PCLTM_DB", str(db))
    monkeypatch.setenv("HERMES_PCLTM_MEMFS_ROOT", str(memfs))
    monkeypatch.setattr("pcltm.cli.load_prompt_context", lambda **kwargs: "<pcltm_context>\nbody\n</pcltm_context>")
    monkeypatch.setattr("pcltm.cli.last_live_context_telemetry", lambda: {"within_budget": True})

    report = live_context_smoke()

    assert report["db_path"] == str(db)
    assert report["schema_version"] == 9
    assert report["memfs_root"] == str(memfs)


def test_cli_live_context_smoke_json(capsys, monkeypatch) -> None:
    monkeypatch.setattr("pcltm.cli.load_prompt_context", lambda **kwargs: "<pcltm_context>\nbody\n</pcltm_context>")
    monkeypatch.setattr(
        "pcltm.cli.last_live_context_telemetry",
        lambda: {"within_budget": True, "total_chars": 36, "limit_chars": 900, "recall_intent": {"intent": "default"}},
    )

    exit_code = main(["live-context", "smoke", "--query", "hello", "--json"])

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
