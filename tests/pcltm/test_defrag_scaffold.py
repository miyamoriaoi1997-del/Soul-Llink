from __future__ import annotations

from pathlib import Path

import pytest

from pcltm.defrag import DefragAction, DefragPlan, MemFSDefragger
from pcltm.memfs_store import MemFSStore


def make_store(tmp_path: Path) -> MemFSStore:
    store = MemFSStore(tmp_path / "memfs")
    store.init()
    return store


def write_memory(store: MemFSStore, relative_path: str, body: str, *, description: str = "Test memory") -> Path:
    path = store.root / relative_path
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "---\n"
        f"description: {description}\n"
        "authority: pinned\n"
        "mode_scope: [daily, work, sex]\n"
        "buckets: [test]\n"
        "source: test\n"
        "last_reviewed: ''\n"
        "---\n"
        f"{body}",
        encoding="utf-8",
    )
    return path


def test_analyze_empty_store(tmp_path: Path) -> None:
    store = make_store(tmp_path)

    plan = MemFSDefragger(store).analyze()

    assert plan.actions == []
    assert plan.total_files_analyzed == 0
    assert plan.duplicate_count == 0
    assert plan.oversized_count == 0


def test_analyze_detects_duplicates(tmp_path: Path) -> None:
    store = make_store(tmp_path)
    write_memory(store, "pinned/a.md", "same body")
    write_memory(store, "pinned/b.md", "same body")

    plan = MemFSDefragger(store).analyze()

    assert plan.duplicate_count == 1
    assert any(
        action.action_type == "merge"
        and action.source_path == "pinned/a.md"
        and action.target_path == "pinned/b.md"
        for action in plan.actions
    )


def test_analyze_detects_oversized(tmp_path: Path) -> None:
    store = make_store(tmp_path)
    write_memory(store, "pinned/large.md", "x" * 11)

    plan = MemFSDefragger(store, size_threshold=10).analyze()

    assert plan.oversized_count == 1
    assert any(
        action.action_type == "split"
        and action.source_path == "pinned/large.md"
        and action.target_path == "pinned/large.part1.md"
        for action in plan.actions
    )


def test_analyze_no_action_for_normal_files(tmp_path: Path) -> None:
    store = make_store(tmp_path)
    write_memory(store, "pinned/a.md", "short unique body")
    write_memory(store, "episodic/2026/05/b.md", "another normal body")

    plan = MemFSDefragger(store, size_threshold=100).analyze()

    assert plan.actions == []
    assert plan.duplicate_count == 0
    assert plan.oversized_count == 0


def test_analyze_returns_counts(tmp_path: Path) -> None:
    store = make_store(tmp_path)
    write_memory(store, "pinned/a.md", "duplicate")
    write_memory(store, "pinned/b.md", "duplicate")
    write_memory(store, "pinned/large.md", "x" * 12)
    write_memory(store, "pinned/normal.md", "normal")

    plan = MemFSDefragger(store, size_threshold=10).analyze()

    assert plan.total_files_analyzed == 4
    assert plan.duplicate_count == 1
    assert plan.oversized_count == 1
    assert len(plan.actions) == 2


def test_apply_merges_duplicates(tmp_path: Path) -> None:
    store = make_store(tmp_path)
    source = write_memory(store, "pinned/a.md", "same body")
    duplicate = write_memory(store, "pinned/b.md", "same body")
    plan = DefragPlan(
        actions=[DefragAction(action_type="merge", source_path="pinned/a.md", target_path="pinned/b.md")]
    )

    MemFSDefragger(store).apply(plan)

    assert source.exists()
    assert not duplicate.exists()


def test_apply_splits_oversized(tmp_path: Path) -> None:
    store = make_store(tmp_path)
    write_memory(store, "pinned/large.md", "abcdefghijkl")
    plan = DefragPlan(
        actions=[
            DefragAction(
                action_type="split",
                source_path="pinned/large.md",
                target_path="pinned/large.part1.md",
            )
        ]
    )

    MemFSDefragger(store, size_threshold=5).apply(plan)

    assert (store.root / "pinned/large.part1.md").exists()
    _frontmatter, target_body = store.read_file("pinned/large.part1.md")
    _frontmatter, source_body = store.read_file("pinned/large.md")
    assert target_body == "abcde"
    assert source_body == "fghijkl"


def test_apply_empty_plan_noop(tmp_path: Path) -> None:
    store = make_store(tmp_path)

    MemFSDefragger(store).apply(DefragPlan())

    assert store.root.exists()


def test_defragger_size_threshold_customizable(tmp_path: Path) -> None:
    store = make_store(tmp_path)
    write_memory(store, "pinned/memory.md", "123456")

    default_plan = MemFSDefragger(store).analyze()
    custom_plan = MemFSDefragger(store, size_threshold=5).analyze()

    assert default_plan.actions == []
    assert custom_plan.oversized_count == 1
    assert any(action.action_type == "split" for action in custom_plan.actions)


def test_apply_continues_on_error(tmp_path: Path) -> None:
    store = make_store(tmp_path)
    duplicate = write_memory(store, "pinned/duplicate.md", "same body")
    plan = DefragPlan(
        actions=[
            DefragAction(
                action_type="split",
                source_path="../unsafe.md",
                target_path="pinned/unsafe.part1.md",
            ),
            DefragAction(
                action_type="delete",
                source_path="pinned/duplicate.md",
            ),
        ]
    )

    with pytest.warns(RuntimeWarning, match="unsafe MemFS relative path"):
        MemFSDefragger(store, size_threshold=5).apply(plan)

    assert not duplicate.exists()
