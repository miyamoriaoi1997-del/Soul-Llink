from __future__ import annotations

from pathlib import Path
import sqlite3

import pytest

from pcltm import memory_adapter
from pcltm.memfs_types import PromptMemoryView


def write_memfs_file(
    root: Path,
    rel: str,
    *,
    description: str,
    authority: str,
    mode_scope: list[str],
    buckets: list[str],
    body: str,
    memory_type: str = "UserPreference",
    lifecycle_state: str = "active",
) -> None:
    path = root / rel
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "---\n"
        f"description: {description!r}\n"
        f"authority: {authority!r}\n"
        f"mode_scope: {mode_scope!r}\n"
        f"buckets: {buckets!r}\n"
        f"memory_type: {memory_type!r}\n"
        f"lifecycle_state: {lifecycle_state!r}\n"
        "source: test\n"
        "---\n"
        f"{body}\n",
        encoding="utf-8",
    )


@pytest.fixture
def memfs_root(tmp_path, monkeypatch):
    root = tmp_path / "memfs"
    for layer in ("system", "pinned", "episodic", "transient"):
        (root / layer).mkdir(parents=True, exist_ok=True)
    monkeypatch.setattr(memory_adapter, "MEMFS_ROOT", root)
    return root


def populate_memfs(root: Path) -> None:
    write_memfs_file(
        root,
        "system/core.md",
        description="Core system contract",
        authority="system",
        mode_scope=["daily", "work", "sex"],
        buckets=["identity"],
        body="system contract",
        memory_type="RuntimeInvariant",
    )
    write_memfs_file(
        root,
        "pinned/runtime.md",
        description="Runtime boundary",
        authority="pinned",
        mode_scope=["work"],
        buckets=["runtime_boundary"],
        body="runtime production boundary",
        memory_type="RuntimeInvariant",
    )
    write_memfs_file(
        root,
        "pinned/project.md",
        description="Project path",
        authority="pinned",
        mode_scope=["work"],
        buckets=["project_path"],
        body="project workdir /example/soul-link",
        memory_type="ProjectPath",
    )
    write_memfs_file(
        root,
        "pinned/relationship.md",
        description="Relationship preference",
        authority="pinned",
        mode_scope=["daily", "sex"],
        buckets=["relationship"],
        body="daily relationship memory",
        memory_type="RelationshipAnchor",
    )
    write_memfs_file(
        root,
        "episodic/2026/05/work.md",
        description="Work episode",
        authority="episodic",
        mode_scope=["work"],
        buckets=["current_task"],
        body="recent work episode",
        memory_type="TemporaryTaskState",
    )
    write_memfs_file(
        root,
        "transient/current.md",
        description="Transient note",
        authority="transient",
        mode_scope=["work"],
        buckets=["current_task"],
        body="current transient context",
        memory_type="TemporaryTaskState",
    )


def test_returns_prompt_memory_view(memfs_root):
    populate_memfs(memfs_root)

    view = memory_adapter.load_layered_prompt_context(mode="work", query="runtime")

    assert isinstance(view, PromptMemoryView)
    summary = view.summary()
    assert summary["layers"][0]["rendered_preview"].startswith("[system]")


def test_system_layer_included_before_pinned(memfs_root):
    populate_memfs(memfs_root)

    text = memory_adapter.load_layered_prompt_context(mode="work", query="runtime").render()

    assert text.index("[system]") < text.index("[pinned]")
    assert "system contract" in text


def test_compression_layer_is_reference_only(memfs_root):
    populate_memfs(memfs_root)

    view = memory_adapter.load_layered_prompt_context(mode="work", query="runtime")

    assert view.compression.is_reference_only is True
    assert "compression" in view.context_summary()["reference_only_layers"]


def test_work_mode_selects_appropriate_buckets(memfs_root):
    populate_memfs(memfs_root)

    text = memory_adapter.load_layered_prompt_context(mode="work", query="runtime project").render()

    assert "runtime production boundary" in text
    assert "project workdir" in text
    assert "daily relationship memory" not in text


def test_daily_mode_excludes_runtime(memfs_root):
    populate_memfs(memfs_root)

    text = memory_adapter.load_layered_prompt_context(mode="daily", query="relationship runtime").render()

    assert "daily relationship memory" in text
    assert "runtime production boundary" not in text


def test_fallback_when_no_memfs_repo(tmp_path, monkeypatch):
    monkeypatch.setattr(memory_adapter, "MEMFS_ROOT", tmp_path / "missing")

    view = memory_adapter.load_layered_prompt_context(mode="work", query="runtime")

    assert isinstance(view, PromptMemoryView)
    assert view.system.layer == "system"
    assert view.compression.is_reference_only


def test_empty_query_graceful(memfs_root):
    populate_memfs(memfs_root)

    view = memory_adapter.load_layered_prompt_context(mode="work", query="")

    assert isinstance(view, PromptMemoryView)
    assert view.total_items >= 1


def test_custom_budgets_respected(memfs_root):
    populate_memfs(memfs_root)
    write_memfs_file(
        memfs_root,
        "pinned/huge.md",
        description="Huge pinned item",
        authority="pinned",
        mode_scope=["work"],
        buckets=["runtime_boundary"],
        body="x" * 500,
        memory_type="RuntimeInvariant",
    )

    view = memory_adapter.load_layered_prompt_context(
        mode="work",
        query="runtime",
        budgets={"pinned": 80},
    )

    assert view.pinned.used_chars <= 80
    assert view.pinned.omitted_count >= 1


def test_selected_layers_bound_active_memfs_view(memfs_root):
    populate_memfs(memfs_root)

    view = memory_adapter.load_layered_prompt_context(
        mode="daily",
        query="relationship runtime current transient",
        layers=["system", "pinned"],
        buckets=["relationship", "user_preference"],
    )

    assert view.system.items
    assert view.pinned.items
    assert not view.episodic.items
    assert not view.transient.items
    assert "relationship" in view.selected_buckets


def test_selected_buckets_filter_same_mode_memfs_items(memfs_root):
    populate_memfs(memfs_root)

    view = memory_adapter.load_layered_prompt_context(
        mode="work",
        query="runtime project current",
        buckets=["runtime_boundary"],
    )
    text = view.render()

    assert "runtime production boundary" in text
    assert "project workdir" not in text
    assert "recent work episode" not in text
    assert not view.episodic.items
    assert not view.transient.items


def test_active_layers_keep_archival_loaded_but_reference_only(memfs_root):
    populate_memfs(memfs_root)

    view = memory_adapter.load_layered_prompt_context(
        mode="work",
        query="runtime project current",
        layers=["system", "pinned", "episodic", "transient"],
        active_layers=["system", "pinned", "transient"],
        buckets=["runtime_boundary", "project_path", "current_task"],
    )

    assert view.episodic.items
    active = view.render_active_frame()
    full = view.render()
    assert "recent work episode" not in active
    assert "recent work episode" in full
    summary = view.context_summary()
    assert "episodic" in summary["reference_only_layers"]
    assert "compression" in summary["reference_only_layers"]


def test_select_context_snapshot_public_api_is_host_neutral(memfs_root):
    populate_memfs(memfs_root)

    snapshot = memory_adapter.select_context_snapshot(
        mode="work",
        query="runtime project current",
        layers=["system", "pinned", "episodic", "transient"],
        active_layers=["system", "pinned", "transient"],
        buckets=["runtime_boundary", "project_path", "current_task"],
        root=memfs_root,
    )
    payload = snapshot.to_dict()

    assert payload["object_type"] == "pcltm_context_selection_snapshot"
    assert payload["selection_source"] == "memfs"
    assert payload["active_layers"] == ["system", "pinned", "transient"]
    assert payload["selected_buckets"] == ["current_task", "project_path", "runtime_boundary"]
    assert "episodic" in payload["reference_only_layers"]
    assert "compression" in payload["reference_only_layers"]
    assert payload["total_selected_items"] >= 3
    assert all("selected_items" in layer for layer in payload["layers"])


def test_explicit_memfs_root_does_not_merge_unrelated_default_db(memfs_root, tmp_path, monkeypatch):
    populate_memfs(memfs_root)
    live_db = tmp_path / "unrelated-live.db"
    con = sqlite3.connect(live_db)
    con.execute(
        "CREATE TABLE memory_records (record_id INTEGER PRIMARY KEY, candidate_id TEXT, "
        "kind TEXT, target_file TEXT, content TEXT, confidence REAL, sensitivity TEXT, "
        "source_event_ids TEXT, source_node_ids TEXT, status TEXT, metadata TEXT, created_at TEXT)"
    )
    con.execute(
        "INSERT INTO memory_records VALUES (1, 'live-1', 'fact', 'MEMORY.md', "
        "'unrelated production episode', 1, 'normal', '[]', '[]', 'approved', '{}', NULL)"
    )
    con.commit()
    con.close()
    monkeypatch.setenv("HERMES_PCLTM_DB", str(live_db))

    view = memory_adapter.load_layered_prompt_context(
        mode="work", query="runtime", buckets=["runtime_boundary"], root=memfs_root
    )

    assert view.selection_source == "memfs"
    assert "unrelated production episode" not in view.render()


def test_configured_test_memfs_root_does_not_merge_live_db_from_another_data_boundary(memfs_root, tmp_path, monkeypatch):
    """An implicit module-configured fixture root must not inherit cwd/live DB rows."""
    populate_memfs(memfs_root)
    live_dir = tmp_path / "production-var"
    live_dir.mkdir()
    live_db = live_dir / "pcltm-prod.db"
    con = sqlite3.connect(live_db)
    con.execute(
        "CREATE TABLE memory_records (record_id INTEGER PRIMARY KEY, candidate_id TEXT, "
        "kind TEXT, target_file TEXT, content TEXT, confidence REAL, sensitivity TEXT, "
        "source_event_ids TEXT, source_node_ids TEXT, status TEXT, metadata TEXT, created_at TEXT)"
    )
    con.execute(
        "INSERT INTO memory_records VALUES (1, 'live-1', 'fact', 'MEMORY.md', "
        "'live db must stay outside fixture view', 1, 'normal', '[]', '[]', "
        "'approved', '{\"buckets\":[\"runtime_boundary\"]}', NULL)"
    )
    con.commit()
    con.close()
    monkeypatch.setenv("HERMES_PCLTM_DB", str(live_db))

    view = memory_adapter.load_layered_prompt_context(
        mode="work", query="runtime", buckets=["runtime_boundary"]
    )

    assert view.selection_source == "memfs"
    assert "live db must stay outside fixture view" not in view.render()


def test_active_task_bucket_expands_to_continuity_capsule(memfs_root):
    populate_memfs(memfs_root)
    write_memfs_file(
        memfs_root,
        "episodic/2026/06/continuity.md",
        description="Previous conversation continuity capsule",
        authority="episodic",
        mode_scope=["work"],
        buckets=["continuity_capsule"],
        body="<previous_conversation_state>resume PCLTM layered memory施工</previous_conversation_state>",
        memory_type="TemporaryTaskState",
    )

    view = memory_adapter.load_layered_prompt_context(
        mode="work",
        query="继续施工",
        layers=["system", "pinned", "episodic", "transient"],
        active_layers=["system", "pinned", "transient"],
        buckets=["active_task"],
        root=memfs_root,
    )

    assert "continuity_capsule" in view.selected_buckets
    assert view.episodic.items
    assert "resume PCLTM layered memory施工" not in view.render_active_frame()
    assert "resume PCLTM layered memory施工" in view.render()
    assert "previous_conversation_state" in view.render()
    frame = view.active_frame()
    assert "episodic" in frame.reference_only_layers
    assert any(
        layer["layer"] == "episodic" and layer["item_count"] >= 1
        for layer in frame.reference_summary["layers"]
    )


def test_render_labels_typed_memory_without_exposing_paths(memfs_root):
    populate_memfs(memfs_root)

    active = memory_adapter.load_layered_prompt_context(
        mode="work",
        query="current transient",
        buckets=["current_task"],
        root=memfs_root,
    ).render_active_frame()

    assert "[type=TemporaryTaskState; state=active; buckets=current_task] current transient context" in active
    assert "transient/current.md" not in active
