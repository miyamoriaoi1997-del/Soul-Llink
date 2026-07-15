from __future__ import annotations

import sqlite3
from pathlib import Path

from pcltm.cli import init_runtime
from pcltm.monitoring.private_data import (
    collect_emotion_state,
    collect_injection_preview,
    collect_memory_bodies,
    collect_runtime_turn_capture,
    collect_soul_content,
)


def test_emotion_state_reads_frontmatter_values(tmp_path: Path) -> None:
    state = tmp_path / "STATE.md"
    state.write_text("---\nemotion_state:\n  affection: 105\n  trust: 60\n  emotion_score: 2.5\n  last_update: '2026-07-13T23:00:00'\n---\nbody\n", encoding="utf-8")
    report = collect_emotion_state(state)
    assert report["source"] == "runtime_state_file"
    assert report["axes"]["affection"] == 105
    assert report["emotion_score"] == 2.5


def test_memory_body_collector_returns_bounded_records(tmp_path: Path) -> None:
    db = tmp_path / "runtime.db"; memfs = tmp_path / "memfs"
    init_runtime(db_path=db, memfs_root=memfs)
    with sqlite3.connect(db) as con:
        con.execute("INSERT INTO memory_records(candidate_id,kind,target_file,content,confidence,sensitivity,source_event_ids,source_node_ids,status,metadata) VALUES(?,?,?,?,?,?,?,?,?,?)", ("c1","preference","USER.md","private memory body",1.0,"normal","[]","[]","approved",'{"scope_key":"user:example-user"}'))
    report = collect_memory_bodies(db, limit=10)
    assert report["records"][0]["content"] == "private memory body"
    assert report["records"][0]["scope_key"] == "user:example-user"


def test_injection_preview_is_labeled_and_does_not_update_retrieval_stats(tmp_path: Path) -> None:
    db = tmp_path / "runtime.db"; memfs = tmp_path / "memfs"
    init_runtime(db_path=db, memfs_root=memfs)
    with sqlite3.connect(db) as con:
        con.execute("INSERT INTO memory_records(candidate_id,kind,target_file,content,confidence,sensitivity,source_event_ids,source_node_ids,status,metadata) VALUES(?,?,?,?,?,?,?,?,?,?)", ("c1","preference","USER.md","preview body",1.0,"normal","[]","[]","approved",'{}'))
    before = db.read_bytes()
    report = collect_injection_preview(db, mode="work", query=None)
    after = db.read_bytes()
    assert report["source"] == "sidecar_reconstruction_preview"
    assert report["is_exact_host_capture"] is False
    assert report["state_machine_mode"] == "work"
    assert report["pcltm_mode"] == "work"
    assert report["mode_sync"] == "consistent"
    assert "preview body" in report["rendered"]
    assert report["selected_record_ids"]
    assert before == after


def test_runtime_turn_capture_returns_exact_emotion_and_state_machine_blocks(tmp_path: Path) -> None:
    capture = tmp_path / "latest-turn.json"
    capture.write_text(
        '{"source":"exact_host_capture","captured_at":"2026-07-14T00:00:00+00:00",'
        '"emotion_modifier":"<emotion_modifier>live</emotion_modifier>",'
        '"state_machine":{"mode":"work","selected_layers":["core","work"]},'
        '"mode_sync":{"state_machine_mode":"work","pcltm_mode":"work","status":"consistent"},'
        '"soul_mode_layer":{"source":"runtime_template","content":"# Work"},'
        '"turn_injection":"<soullink_turn_state>live</soullink_turn_state>"}',
        encoding="utf-8",
    )

    report = collect_runtime_turn_capture(capture)

    assert report["source"] == "exact_host_capture"
    assert report["emotion_modifier"].startswith("<emotion_modifier>")
    assert report["state_machine"]["mode"] == "work"
    assert report["mode_sync"]["status"] == "consistent"
    assert report["soul_mode_layer"]["content"] == "# Work"


def test_soul_content_reads_active_anchor_and_all_mode_layers(tmp_path: Path) -> None:
    active = tmp_path / "SOUL.md"
    layers = tmp_path / "layers"
    layers.mkdir()
    active.write_text("# active soul", encoding="utf-8")
    for name in ("core", "daily", "work", "sex"):
        (layers / f"SOUL.{name}.template.md").write_text(f"# {name}", encoding="utf-8")

    report = collect_soul_content(active, layers)

    assert report["source"] == "runtime_soul_files"
    assert report["active"]["content"] == "# active soul"
    assert report["layers"]["work"]["content"] == "# work"
