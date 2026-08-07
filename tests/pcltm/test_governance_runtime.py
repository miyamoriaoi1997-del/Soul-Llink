from __future__ import annotations

import json
import sqlite3

from pcltm.governance_runtime import run_governance, scan_scope_collisions
from pcltm.memfs_store import MemFSStore
from pcltm.store import EventStore
from pcltm.cli import main


def _add_record(store: EventStore, *, candidate_id: str, canonical_key: str, scope_key: str, status: str = "approved") -> None:
    store.add_memory_record(
        candidate_id=candidate_id,
        kind="project",
        target_file="MEMORY.md",
        content=f"Memory for {candidate_id}",
        confidence=0.8,
        sensitivity="normal",
        source_event_ids=[],
        source_node_ids=[],
        status=status,
        metadata={
            "canonical_key": canonical_key,
            "scope_key": scope_key,
            "object_type": "project",
            "scope": "project",
        },
    )


def test_scan_scope_collisions_flags_unscoped_duplicate_canonical_keys(tmp_path) -> None:
    db = tmp_path / "pcltm.db"
    store = EventStore(db)
    try:
        _add_record(store, candidate_id="a", canonical_key="project/index-doctor", scope_key="project:a")
        _add_record(store, candidate_id="b", canonical_key="project/index-doctor", scope_key="project:b")
        _add_record(store, candidate_id="c", canonical_key="project:c/project/index-doctor", scope_key="project:c")
    finally:
        store.close()

    report = scan_scope_collisions(db_path=db)

    assert report["ok"] is False
    assert report["collision_count"] == 1
    assert report["collisions"][0]["code"] == "scope_canonical_key_collision"
    assert report["collisions"][0]["canonical_key"] == "project/index-doctor"
    assert sorted(report["collisions"][0]["scope_keys"]) == ["project:a", "project:b"]
    assert {row["candidate_id"] for row in report["collisions"][0]["records"]} == {"a", "b"}


def test_run_governance_aggregates_index_governance_scope_and_selection(tmp_path, monkeypatch) -> None:
    db = tmp_path / "pcltm.db"
    memfs_root = tmp_path / "memfs"
    store = EventStore(db)
    try:
        event_id = store.append_event(
            session_id="s1",
            conversation_id="c1",
            platform="test",
            role="user",
            source="pytest",
            content="需要治理总入口",
            persona_mode="work",
        )
        store.add_memory_record(
            candidate_id="pending-1",
            kind="preference",
            target_file="USER.md",
            content="Pending preference needs review.",
            confidence=0.7,
            sensitivity="normal",
            source_event_ids=[event_id],
            source_node_ids=[],
            status="pending",
            metadata={"canonical_key": "pref.pending", "scope_key": "profile:default"},
        )
        _add_record(store, candidate_id="a", canonical_key="project/index-doctor", scope_key="project:a")
        _add_record(store, candidate_id="b", canonical_key="project/index-doctor", scope_key="project:b")
    finally:
        store.close()

    memfs = MemFSStore(memfs_root)
    memfs.init()
    monkeypatch.setenv("HERMES_PCLTM_DB", str(db))

    report = run_governance(
        db_path=db,
        memfs_root=memfs_root,
        selection_target="user",
        mode="work",
        dry_run=True,
    )

    assert report["authority_boundary"] == "read_only_governance_runtime"
    assert report["dry_run"] is True
    assert report["ok"] is False
    assert report["index"]["ok"] is True
    assert report["memory_governance"]["pending_candidates"] == 1
    assert report["scope_collisions"]["collision_count"] == 1
    assert report["selection_probe"]["target"] == "user"
    assert report["selection_probe"] == {
        "status": "retired",
        "bodyless": True,
        "reason": "legacy_memory_selection_probe_not_runtime_authority",
        "target": "user",
        "mode": "work",
    }
    assert report["summary"]["error_count"] == 1
    assert "scope_canonical_key_collision" in report["summary"]["issue_codes"]


def test_governance_run_cli_emits_json(tmp_path, monkeypatch, capsys) -> None:
    db = tmp_path / "pcltm.db"
    memfs_root = tmp_path / "memfs"
    EventStore(db).close()
    MemFSStore(memfs_root).init()
    monkeypatch.setenv("HERMES_PCLTM_DB", str(db))

    exit_code = main([
        "governance",
        "run",
        "--db",
        str(db),
        "--memfs",
        str(memfs_root),
        "--selection-target",
        "user",
        "--mode",
        "work",
        "--json",
    ])

    assert exit_code == 0
    output = json.loads(capsys.readouterr().out)
    assert output["authority_boundary"] == "read_only_governance_runtime"
    assert output["selection_probe"]["target"] == "user"
    assert output["selection_probe"]["status"] == "retired"
    assert output["selection_probe"]["bodyless"] is True
    assert "selected" not in output["selection_probe"]
    assert "load_entries_baseline" not in output["selection_probe"]
    assert output["ok"] is True
