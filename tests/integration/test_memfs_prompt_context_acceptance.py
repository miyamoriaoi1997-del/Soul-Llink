from __future__ import annotations

from pathlib import Path

import pytest

import pcltm.memory_adapter as ma
from pcltm.defrag import MemFSDefragger
from pcltm.memfs_store import MemFSStore
from pcltm.memfs_types import (
    MemoryFileFrontmatter,
    MemoryLayerItem,
    MemoryLayerView,
    PromptMemoryView,
)
from pcltm.memory_adapter import MEMFS_ROOT, load_layered_prompt_context
from pcltm.reflection import ReflectionCandidateBuilder, ReflectionWriter
from persona_engine.persona_orchestrator.memory_selector import MemorySelector
from persona_engine.persona_orchestrator.prompt_composer import PromptComposer
from persona_engine.persona_orchestrator.types import MemorySelection


def write_memfs_file(
    root: Path,
    rel: str,
    *,
    description: str,
    authority: str,
    mode_scope: list[str],
    buckets: list[str],
    body: str,
) -> None:
    path = root / rel
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "---\n"
        f"description: {description!r}\n"
        f"authority: {authority!r}\n"
        f"mode_scope: {mode_scope!r}\n"
        f"buckets: {buckets!r}\n"
        "source: test\n"
        "---\n"
        f"{body}\n",
        encoding="utf-8",
    )


@pytest.fixture
def memfs_repo(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    root = tmp_path / "memfs"
    for layer in ("system", "pinned", "episodic", "transient"):
        (root / layer).mkdir(parents=True, exist_ok=True)
    monkeypatch.setattr(ma, "MEMFS_ROOT", root)
    return root


def populate_repo(root: Path) -> None:
    write_memfs_file(
        root,
        "system/identity.md",
        description="System identity contract",
        authority="system",
        mode_scope=["daily", "work", "sex"],
        buckets=["identity"],
        body="system identity and authority",
    )
    write_memfs_file(
        root,
        "pinned/runtime-ops.md",
        description="Runtime boundary facts",
        authority="pinned",
        mode_scope=["work"],
        buckets=["runtime_boundary", "project_path"],
        body="runtime ops for /example/soul-link",
    )
    write_memfs_file(
        root,
        "pinned/relationship.md",
        description="Relationship facts",
        authority="pinned",
        mode_scope=["daily", "sex"],
        buckets=["relationship"],
        body="relationship memory for daily mode",
    )
    write_memfs_file(
        root,
        "episodic/some-event.md",
        description="Mixed event",
        authority="episodic",
        mode_scope=["work", "daily"],
        buckets=["current_task"],
        body="recent event evidence",
    )


def test_work_request_loads_runtime_not_relationship(memfs_repo: Path) -> None:
    populate_repo(memfs_repo)

    view = load_layered_prompt_context(mode="work")

    assert view.system.items
    assert any(item.path == "pinned/runtime-ops.md" for item in view.pinned.items)
    assert all(item.path != "pinned/relationship.md" for item in view.pinned.items)


def test_daily_request_loads_relationship_facts(memfs_repo: Path) -> None:
    populate_repo(memfs_repo)

    view = load_layered_prompt_context(mode="daily")

    assert any(item.path == "pinned/relationship.md" for item in view.pinned.items)
    assert view.transient.items == []


def test_compression_sidecar_is_not_active_render_layer() -> None:
    view = PromptMemoryView(
        system=MemoryLayerView(
            layer="system",
            items=[MemoryLayerItem(path="system/identity.md", body="system identity", authority="system")],
        ),
        compression=MemoryLayerView(
            layer="compression",
            items=[MemoryLayerItem(path="compression/summary.md", body="compressed reference", authority="compression")],
            is_reference_only=True,
        ),
    )

    rendered = view.render()

    assert "[system]" in rendered
    assert "[compression]" not in rendered
    assert "compressed reference" not in rendered


def test_old_compression_cannot_enter_active_memory_view() -> None:
    view = PromptMemoryView(
        compression=MemoryLayerView(
            layer="compression",
            items=[MemoryLayerItem(path="compression/old-task.md", body="Current Active Task: fix old bug")],
            is_reference_only=True,
        )
    )

    rendered = view.render()

    assert rendered == ""
    assert "Current Active Task: fix old bug" not in rendered


def test_emotion_state_not_from_memory_prose() -> None:
    frontmatter = MemoryFileFrontmatter(
        description="System state note",
        authority="system",
        extra={"emotion_score": 0.87, "current_emotion": 0.42},
    )
    system_layer = MemoryLayerView(layer="system")

    assert frontmatter.extra["emotion_score"] == 0.87
    assert frontmatter.extra["current_emotion"] == 0.42
    assert not hasattr(system_layer, "emotion_score")


def test_memfs_fallback_when_no_repo(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    missing_root = tmp_path / "missing-memfs"
    monkeypatch.setattr(ma, "MEMFS_ROOT", missing_root)

    view = load_layered_prompt_context(mode="work")

    assert isinstance(view, PromptMemoryView)
    assert view.system.layer == "system"
    assert view.system.items == []


def test_full_prompt_composer_omits_managed_memory_view_when_host_context_owns_active_surface() -> None:
    composer = PromptComposer(Path(__file__).resolve().parents[2] / "packages" / "persona_engine")
    memory_view = PromptMemoryView(
        system=MemoryLayerView(
            layer="system",
            items=[MemoryLayerItem(path="system/identity.md", body="system identity")],
        ),
        pinned=MemoryLayerView(
            layer="pinned",
            items=[MemoryLayerItem(path="pinned/runtime-ops.md", body="runtime boundary")],
        ),
    )

    composition = composer.compose_with_memory_view(
        selected_layers=["daily"],
        memory_view_text=memory_view.render(),
    )

    assert "<pcltm_memory_view>" not in composition.prompt_text
    assert "</pcltm_memory_view>" not in composition.prompt_text
    assert "system identity" not in composition.prompt_text
    assert "runtime boundary" not in composition.prompt_text


def test_memory_selector_work_includes_all_layers() -> None:
    selector = MemorySelector()

    selection = selector.select(mode="work")

    assert isinstance(selection, MemorySelection)
    assert selection.layers == ["system", "pinned", "episodic", "transient"]
    assert "runtime_boundary" in selection.buckets
    assert "project_path" in selection.buckets
