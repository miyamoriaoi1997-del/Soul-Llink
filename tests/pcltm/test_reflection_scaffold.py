from __future__ import annotations

from dataclasses import FrozenInstanceError
from pathlib import Path

import pytest

from pcltm.memfs_store import MemFSStore
from pcltm.memfs_types import MemoryFileFrontmatter
from pcltm.reflection import MemoryDraft, ReflectionCandidateBuilder, ReflectionWriter


def make_frontmatter(*, authority: str = "episodic") -> MemoryFileFrontmatter:
    return MemoryFileFrontmatter(
        description="Reflection draft",
        authority=authority,
        mode_scope=("work",),
        buckets=("reflection_candidate",),
        source="reflection",
        last_reviewed="2026-05-23",
    )


def make_draft(
    *,
    target_layer: str = "episodic",
    relative_path: str = "episodic/2026/2026-05-task-review.md",
    authority: str | None = None,
    body: str = "Draft body.",
) -> MemoryDraft:
    return MemoryDraft(
        target_layer=target_layer,
        relative_path=relative_path,
        frontmatter=make_frontmatter(authority=authority or target_layer),
        body=body,
        provenance="test",
    )


def test_memory_draft_valid_frontmatter() -> None:
    draft = make_draft()

    assert draft.frontmatter.description == "Reflection draft"
    assert draft.frontmatter.authority == "episodic"


def test_memory_draft_rejects_invalid_frontmatter() -> None:
    with pytest.raises(ValueError):
        make_frontmatter(authority="invalid")

    with pytest.raises(ValueError):
        MemoryDraft(
            target_layer="invalid",
            relative_path="invalid/test.md",
            frontmatter=make_frontmatter(),
            body="body",
        )

    draft = make_draft()
    with pytest.raises(FrozenInstanceError):
        draft.frontmatter.authority = "invalid"  # type: ignore[misc]


def test_build_from_events_creates_drafts() -> None:
    events = [
        {
            "type": "preference_confirmed",
            "user_message": "user likes dark mode",
            "mode": "daily",
            "timestamp": "2026-05-23T10:00:00",
        }
    ]

    drafts = ReflectionCandidateBuilder().build_from_events(events)

    assert len(drafts) == 1
    draft = drafts[0]
    assert draft.target_layer == "episodic"
    assert draft.relative_path.startswith("episodic/2026/2026-05-23-preference-confirmed-")
    assert draft.frontmatter.authority == "episodic"
    assert draft.frontmatter.mode_scope == ("daily",)
    assert "preference_confirmed" in draft.frontmatter.buckets
    assert "user likes dark mode" in draft.body
    assert draft.source == "reflection"
    assert draft.provenance == "event[0]:preference_confirmed"


def test_build_from_events_empty_input() -> None:
    assert ReflectionCandidateBuilder().build_from_events([]) == []


def test_writer_rejects_system_without_flag(tmp_path: Path) -> None:
    store = MemFSStore(tmp_path / "memfs")
    writer = ReflectionWriter(store)
    draft = make_draft(
        target_layer="system",
        relative_path="system/reflection.md",
        authority="system",
    )

    assert writer.write_draft(draft) is False
    assert not (tmp_path / "memfs" / "system" / "reflection.md").exists()


def test_writer_allows_system_with_flag(tmp_path: Path) -> None:
    store = MemFSStore(tmp_path / "memfs")
    writer = ReflectionWriter(store)
    draft = make_draft(
        target_layer="system",
        relative_path="system/reflection.md",
        authority="system",
    )

    assert writer.write_draft(draft, allow_system=True) is True
    frontmatter, body = store.read_file("system/reflection.md")
    assert frontmatter.authority == "system"
    assert "Draft body" in body


def test_writer_writes_episodic_draft(tmp_path: Path) -> None:
    store = MemFSStore(tmp_path / "memfs")
    writer = ReflectionWriter(store)
    draft = make_draft(
        target_layer="episodic",
        relative_path="episodic/2026/2026-05-task-review.md",
        authority="episodic",
        body="Episodic draft body.",
    )

    assert writer.write_draft(draft) is True
    frontmatter, body = store.read_file("episodic/2026/2026-05-task-review.md")
    assert frontmatter.authority == "episodic"
    assert frontmatter.source == "reflection"
    assert frontmatter.extra["provenance"] == "test"
    assert "Episodic draft body." in body


def test_writer_writes_transient_draft(tmp_path: Path) -> None:
    store = MemFSStore(tmp_path / "memfs")
    writer = ReflectionWriter(store)
    draft = make_draft(
        target_layer="transient",
        relative_path="transient/current/reflection.md",
        authority="transient",
        body="Transient draft body.",
    )

    assert writer.write_draft(draft) is True
    frontmatter, body = store.read_file("transient/current/reflection.md")
    assert frontmatter.authority == "transient"
    assert "Transient draft body." in body


def test_write_all_returns_stats(tmp_path: Path) -> None:
    store = MemFSStore(tmp_path / "memfs")
    writer = ReflectionWriter(store)
    drafts = [
        make_draft(relative_path="episodic/2026/a.md"),
        make_draft(target_layer="system", relative_path="system/a.md", authority="system"),
        make_draft(target_layer="transient", relative_path="episodic/bad.md", authority="transient"),
    ]

    stats = writer.write_all(drafts)

    assert stats == {"written": 1, "rejected": 1, "errors": 1}
    assert (tmp_path / "memfs" / "episodic" / "2026" / "a.md").is_file()
    assert not (tmp_path / "memfs" / "system" / "a.md").exists()


def test_writer_respects_store_root(tmp_path: Path) -> None:
    store_root = tmp_path / "memfs-root"
    store = MemFSStore(store_root)
    writer = ReflectionWriter(store)
    draft = make_draft(relative_path="episodic/2026/root-check.md")

    assert writer.write_draft(draft) is True
    written_path = store_root / "episodic" / "2026" / "root-check.md"
    assert written_path.is_file()
    assert written_path.resolve().is_relative_to(store_root.resolve())
    assert not (tmp_path / "episodic" / "2026" / "root-check.md").exists()
