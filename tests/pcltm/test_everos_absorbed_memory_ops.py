from __future__ import annotations

import json
import sqlite3
from pathlib import Path

from pcltm.cli import index_doctor, index_stats, main
from pcltm.memfs_store import MemFSStore
from pcltm.memory_object import MemoryObjectScope
from pcltm.scope import MemoryScope, build_scope_key, scoped_canonical_key
from pcltm.store import EventStore


def test_storage_scope_contract_documents_truth_and_derived_layers() -> None:
    contract = Path("docs/pcltm-storage-scope-contract.md")

    text = contract.read_text(encoding="utf-8")

    assert "source of truth" in text
    assert "derived" in text
    assert "MemFS" in text
    assert "SQLite" in text
    assert "scope_key" in text
    assert "app_id" in text
    assert "project_id" in text
    assert "persona_id" in text


def test_scope_key_is_stable_sanitized_and_included_in_canonical_key() -> None:
    scope = MemoryScope(
        profile_id="default",
        app_id="hermes desktop",
        project_id="SoulLink/PCLTM",
        persona_id="Example Persona",
        user_id="example-user",
        mode_scope=("work", "cron"),
    )

    assert scope.key == "profile:default/app:hermes-desktop/project:soullink-pcltm/persona:example-persona/user:example-user/modes:cron+work"
    assert build_scope_key(project_id="SoulLink/PCLTM", mode_scope=("work",)) == "project:soullink-pcltm/modes:work"
    assert scoped_canonical_key(scope, MemoryObjectScope.PROJECT, "Index Doctor") == (
        "profile:default/app:hermes-desktop/project:soullink-pcltm/persona:example-persona/user:example-user/modes:cron+work"
        "/project/index-doctor"
    )


def test_index_stats_reports_sqlite_fts_and_memfs_counts(tmp_path) -> None:
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
            content="用户喜欢严谨验证",
            persona_mode="work",
        )
        store.add_memory_record(
            candidate_id="cand-1",
            kind="user_preference",
            target_file="USER.md",
            content="User prefers rigorous verification.",
            confidence=0.9,
            sensitivity="normal",
            source_event_ids=[event_id],
            source_node_ids=[],
            status="approved",
            metadata={"scope_key": "profile:default"},
        )
    finally:
        store.close()

    memfs = MemFSStore(memfs_root)
    memfs.init()
    (memfs_root / "pinned" / "preference.md").write_text(
        "---\n"
        "description: Verification preference\n"
        "authority: pinned\n"
        "mode_scope: [work]\n"
        "buckets: [user_preference]\n"
        "memory_type: UserPreference\n"
        "lifecycle_state: active\n"
        "---\n\n"
        "User prefers rigorous verification.\n",
        encoding="utf-8",
    )

    report = index_stats(db_path=db, memfs_root=memfs_root)

    assert report["ok"] is True
    assert report["sqlite"]["events"] == 1
    assert report["sqlite"]["event_fts"] == 1
    assert report["sqlite"]["memory_records"] == 1
    assert report["memfs"]["files"] == 1
    assert report["memfs"]["by_layer"]["pinned"] == 1
    assert report["semantic_index"]["records"] == 1


def test_index_doctor_detects_and_rebuilds_fts_mismatch(tmp_path) -> None:
    db = tmp_path / "pcltm.db"
    memfs_root = tmp_path / "memfs"
    store = EventStore(db)
    try:
        store.append_event(
            session_id="s1",
            conversation_id="c1",
            platform="test",
            role="user",
            source="pytest",
            content="需要可重建索引",
            persona_mode="work",
        )
    finally:
        store.close()

    with sqlite3.connect(db) as con:
        con.execute("DELETE FROM event_fts")

    broken = index_doctor(db_path=db, memfs_root=memfs_root)
    assert broken["ok"] is False
    assert any(issue["code"] == "event_fts_mismatch" for issue in broken["issues"])

    fixed = index_doctor(db_path=db, memfs_root=memfs_root, rebuild=True)
    assert fixed["ok"] is True
    assert fixed["rebuild"]["event_rows"] == 1


def test_index_doctor_runs_semantic_query_smoke(tmp_path) -> None:
    db = tmp_path / "pcltm.db"
    memfs_root = tmp_path / "memfs"
    store = EventStore(db)
    try:
        store.add_memory_record(
            candidate_id="doctor-query", kind="project_fact", target_file="MEMORY.md",
            content="index doctor semantic smoke token", confidence=0.9,
            sensitivity="normal", source_event_ids=[], source_node_ids=[],
            status="approved", metadata={},
        )
    finally:
        store.close()

    report = index_doctor(db_path=db, memfs_root=memfs_root)

    assert report["semantic_index"]["query_smoke"]["ok"] is True
    assert report["semantic_index"]["query_smoke"]["result_count"] >= 1


def test_index_doctor_reports_semantic_query_error(monkeypatch, tmp_path) -> None:
    db = tmp_path / "pcltm.db"
    memfs_root = tmp_path / "memfs"
    EventStore(db).close()

    def broken_query(self, *args, **kwargs):
        raise RuntimeError("semantic query failed")

    monkeypatch.setattr("pcltm.index_observability.SemanticIndex.query", broken_query)
    report = index_doctor(db_path=db, memfs_root=memfs_root)

    assert report["ok"] is False
    assert report["semantic_index"]["query_smoke"]["ok"] is False
    issue = next(issue for issue in report["issues"] if issue["code"] == "semantic_query_smoke_failed")
    assert "semantic query failed" in issue["message"]


def test_cli_index_stats_json_outputs_machine_readable_report(tmp_path, capsys) -> None:
    db = tmp_path / "pcltm.db"
    memfs_root = tmp_path / "memfs"
    EventStore(db).close()
    MemFSStore(memfs_root).init()

    exit_code = main(["index", "stats", "--db", str(db), "--memfs", str(memfs_root), "--json"])

    assert exit_code == 0
    output = json.loads(capsys.readouterr().out)
    assert output["ok"] is True
    assert output["authority_boundary"] == "read_only_index_observability"
