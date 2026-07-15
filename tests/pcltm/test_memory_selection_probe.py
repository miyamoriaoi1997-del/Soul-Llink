import json
import os
import sqlite3
import subprocess
import sys

from pcltm.memory_adapter import load_entries
from pcltm.memory_selection import PriorityClass
from pcltm.memory_selection_probe import build_probe_report, explain_memory_records


def _create_db(path):
    con = sqlite3.connect(path)
    try:
        con.execute(
            """
            CREATE TABLE memory_records (
                record_id INTEGER PRIMARY KEY,
                target_file TEXT NOT NULL,
                content TEXT NOT NULL,
                status TEXT NOT NULL,
                kind TEXT,
                metadata TEXT
            )
            """
        )
        rows = [
            (
                1,
                "USER.md",
                "Pinned identity for teacher.",
                "approved",
                "identity",
                '{"canonical_key":"identity.teacher","injection_policy":"pinned","state_affinity":{"modes":["work"]}}',
            ),
            (
                2,
                "USER.md",
                "Pending user preference.",
                "pending",
                "preference",
                '{"canonical_key":"pref.pending","object_type":"preference","injection_policy":"selective","state_affinity":{"modes":["work"]}}',
            ),
            (
                3,
                "USER.md",
                "Quarantined identity evidence.",
                "quarantined",
                "identity",
                '{"canonical_key":"identity.quarantined","injection_policy":"evidence_only"}',
            ),
            (
                4,
                "USER.md",
                "Retired project fact.",
                "retired",
                "project",
                '{"canonical_key":"project.retired","injection_policy":"selective"}',
            ),
            (
                5,
                "USER.md",
                "Daily-only approved preference.",
                "approved",
                "preference",
                '{"canonical_key":"pref.daily","injection_policy":"selective","state_affinity":{"modes":["daily"]}}',
            ),
            (
                6,
                "MEMORY.md",
                "Memory target should not leak into user probe.",
                "approved",
                "memory_note",
                '{"canonical_key":"memory.other","injection_policy":"selective"}',
            ),
        ]
        con.executemany(
            "INSERT INTO memory_records (record_id, target_file, content, status, kind, metadata) VALUES (?, ?, ?, ?, ?, ?)",
            rows,
        )
        con.commit()
    finally:
        con.close()


def test_explain_memory_records_is_sidecar_and_keeps_load_entries_unchanged(tmp_path, monkeypatch):
    db = tmp_path / "pcltm.sqlite3"
    _create_db(db)
    monkeypatch.setenv("HERMES_PCLTM_DB", str(db))

    assert load_entries("user") == [
        "Pinned identity for teacher.",
        "Daily-only approved preference.",
    ]

    probes = explain_memory_records("user", mode="work", emotion_axes=set(), budget_available=1.0)

    assert [probe.record_id for probe in probes] == [1, 2, 3, 4, 5]
    assert load_entries("user") == [
        "Pinned identity for teacher.",
        "Daily-only approved preference.",
    ]


def test_explain_memory_records_reports_selection_and_rejection_reasons(tmp_path, monkeypatch):
    db = tmp_path / "pcltm.sqlite3"
    _create_db(db)
    monkeypatch.setenv("HERMES_PCLTM_DB", str(db))

    probes = {
        probe.memory.canonical_key: probe
        for probe in explain_memory_records("user", mode="work", emotion_axes=set(), budget_available=1.0)
    }

    assert probes["identity.teacher"].decision.selected is True
    assert probes["identity.teacher"].decision.priority_class is PriorityClass.PINNED_IDENTITY
    assert probes["identity.teacher"].decision.reason == "pinned identity"

    assert probes["pref.pending"].decision.selected is False
    assert probes["pref.pending"].decision.rejected_reason == "status=pending"

    assert probes["identity.quarantined"].decision.selected is False
    assert probes["identity.quarantined"].decision.priority_class is PriorityClass.EVIDENCE_ONLY
    assert probes["identity.quarantined"].decision.rejected_reason == "status=quarantined"

    assert probes["project.retired"].decision.selected is False
    assert probes["project.retired"].decision.rejected_reason == "status=retired"

    assert probes["pref.daily"].decision.selected is False
    assert probes["pref.daily"].decision.rejected_reason == "mode_mismatch"


def test_explain_memory_records_returns_empty_for_unknown_target(tmp_path, monkeypatch):
    db = tmp_path / "pcltm.sqlite3"
    _create_db(db)
    monkeypatch.setenv("HERMES_PCLTM_DB", str(db))

    assert explain_memory_records("unknown", mode="work", emotion_axes=set(), budget_available=1.0) == []


def test_build_probe_report_groups_sidecar_results_and_reports_drift(tmp_path, monkeypatch):
    db = tmp_path / "pcltm.sqlite3"
    _create_db(db)
    monkeypatch.setenv("HERMES_PCLTM_DB", str(db))

    report = build_probe_report("user", mode="work", emotion_axes={"focus"}, budget_available=1.0)

    assert report["target"] == "user"
    assert report["mode"] == "work"
    assert report["emotion_axes"] == ["focus"]
    assert [item["memory"]["canonical_key"] for item in report["selected"]] == ["identity.teacher"]
    assert [item["memory"]["canonical_key"] for item in report["skipped"]] == [
        "pref.pending",
        "pref.daily",
    ]
    assert [item["memory"]["canonical_key"] for item in report["quarantined"]] == [
        "identity.quarantined"
    ]
    assert [item["memory"]["canonical_key"] for item in report["retired"]] == [
        "project.retired"
    ]
    assert report["load_entries_baseline"] == [
        "Pinned identity for teacher.",
        "Daily-only approved preference.",
    ]
    assert report["drift_warnings"] == [
        {
            "kind": "selection_baseline_mismatch",
            "message": "probe selected content differs from load_entries baseline",
            "selected_count": 1,
            "baseline_count": 2,
            "missing_from_probe": ["Daily-only approved preference."],
            "extra_in_probe": [],
        }
    ]
    assert load_entries("user") == report["load_entries_baseline"]


def test_memory_selection_probe_cli_reads_env_db_and_emits_json(tmp_path, monkeypatch):
    db = tmp_path / "pcltm.sqlite3"
    _create_db(db)
    monkeypatch.setenv("HERMES_PCLTM_DB", str(db))

    env = os.environ.copy()
    env["PYTHONPATH"] = "packages:adapters"
    result = subprocess.run(
        [
            sys.executable,
            "-m",
            "pcltm.memory_selection_probe",
            "--target",
            "user",
            "--mode",
            "work",
            "--budget",
            "1.0",
            "--json",
        ],
        check=True,
        capture_output=True,
        env=env,
        text=True,
    )
    report = json.loads(result.stdout)

    assert report["db_path"] == str(db)
    assert report["load_entries_baseline"] == [
        "Pinned identity for teacher.",
        "Daily-only approved preference.",
    ]
    assert report["drift_warnings"][0]["kind"] == "selection_baseline_mismatch"
