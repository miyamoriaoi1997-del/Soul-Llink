from __future__ import annotations

import inspect
from pathlib import Path

import pytest

import pcltm
from pcltm import memory_adapter
from pcltm.memfs_store import MemFSStore
from pcltm.memfs_types import MemoryFileFrontmatter
from soul_link.hermes_plugin.memory_provider import SoulLinkMemoryProvider


def test_package_root_does_not_export_retired_legacy_memory_runtime_apis() -> None:
    assert not hasattr(pcltm, "load_entries")
    assert not hasattr(pcltm, "load_prompt_context")
    assert not hasattr(pcltm, "sync_memory_tool_write")


def test_retired_direct_db_prompt_path_is_fail_closed() -> None:
    assert memory_adapter.load_prompt_context(mode="work", query="legacy needle") == ""
    observation = memory_adapter.last_memory_selection_observation()
    assert observation == {
        "status": "retired",
        "authority": "pcltm.memory_current",
        "reason": "legacy_direct_db_prompt_retired",
        "selected_count": 0,
        "selected_records": [],
    }


def test_retired_memory_tool_sync_cannot_request_caller_fallback() -> None:
    with pytest.raises(RuntimeError, match="legacy_memory_tool_sync_retired"):
        memory_adapter.sync_memory_tool_write("memory", "add", content="never persisted")


def test_legacy_db_archival_ids_are_not_openable_on_runtime_surface() -> None:
    with pytest.raises(ValueError, match="legacy_db_memory_id_retired"):
        memory_adapter.open_archival_memory("db/MEMORY.md/1")


def test_quota_selection_has_no_ignore_quota_escape_hatch() -> None:
    source = inspect.getsource(memory_adapter._select_entry_rows)
    assert "ignore_quota" not in source
    assert "if not selected" not in source


def test_production_provider_does_not_call_retired_memory_adapter_paths() -> None:
    source = inspect.getsource(SoulLinkMemoryProvider._load_memory_context)
    assert "search_governed_memories" in source
    assert "build_governed_memory_context" in source
    assert "memory_adapter" not in source
    assert "load_prompt_context" not in source
    assert "load_layered_prompt_context" not in source


def test_retired_legacy_implementations_are_removed_not_left_as_private_backdoors() -> None:
    assert not hasattr(memory_adapter, "_fallback_layered_prompt_context")
    assert not hasattr(memory_adapter, "_retired_load_prompt_context_implementation")
    assert not hasattr(memory_adapter, "_retired_sync_memory_tool_write_implementation")
    assert not hasattr(memory_adapter, "_materialize_memfs_record")
    assert not hasattr(memory_adapter, "_merge_memfs_with_db_authority")
    assert not hasattr(memory_adapter, "_paths_share_data_boundary")
    source = inspect.getsource(memory_adapter.search_archival_memories)
    assert "memory_records" not in source
    assert "db/" not in source


def test_retirement_registry_names_active_architecture_classes() -> None:
    from pcltm.memory_architecture_status import ARCHITECTURE_SURFACES

    assert set(ARCHITECTURE_SURFACES) == {"canonical", "legacy", "retired"}
    assert "pcltm.memory_current" in ARCHITECTURE_SURFACES["canonical"]
    assert "pcltm.memory_records" in ARCHITECTURE_SURFACES["legacy"]
    assert "legacy_direct_db_prompt" in ARCHITECTURE_SURFACES["retired"]
    assert all(
        "authority" not in surface
        for surfaces in ARCHITECTURE_SURFACES.values()
        for surface in surfaces
    )


def test_public_memfs_search_and_open_cannot_disclose_ungoverned_bodies(tmp_path: Path) -> None:
    store = MemFSStore(tmp_path / "arbitrary-root")
    store.write_file(
        "episodic/ungoverned.md",
        MemoryFileFrontmatter(
            description="ungoverned fixture", authority="episodic",
            mode_scope=("work",),
        ),
        "UNGOVERNED_MEMFS_SENTINEL",
    )

    assert store.search("UNGOVERNED_MEMFS_SENTINEL", mode="work") == []
    with pytest.raises(RuntimeError, match="legacy_memfs_open_retired"):
        store.open_memory("episodic/ungoverned.md")
