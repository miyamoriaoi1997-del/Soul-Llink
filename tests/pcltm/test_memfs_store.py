from __future__ import annotations

import os
from pathlib import Path
from unittest.mock import patch

import pytest

from pcltm.memfs_store import MemFSStore
from pcltm.memfs_types import (
    MemoryAuthority,
    MemoryFileFrontmatter,
    MemoryLayerItem,
    MemoryLayerView,
    PromptMemoryView,
)


EXPECTED_DIRS = {"system", "pinned", "episodic", "transient", "skills"}


def write_memory_file(
    root: Path,
    relative_path: str,
    *,
    description: str = "Short description",
    authority: str | None = None,
    mode_scope: list[str] | None = None,
    buckets: list[str] | None = None,
    body: str = "Body content here.",
) -> Path:
    layer = relative_path.split("/", 1)[0]
    authority = authority or layer
    mode_scope = mode_scope or ["daily", "work", "sex"]
    buckets = buckets or []
    path = root / relative_path
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "---\n"
        f'description: "{description}"\n'
        f"authority: {authority}\n"
        f"mode_scope: [{', '.join(mode_scope)}]\n"
        f"buckets: [{', '.join(buckets)}]\n"
        "source: pcltm\n"
        'last_reviewed: "2026-05-23"\n'
        "memory_type: UserPreference\n"
        "lifecycle_state: active\n"
        "---\n\n"
        f"{body}\n",
        encoding="utf-8",
    )
    return path


def test_imports_include_required_types() -> None:
    assert MemoryAuthority.SYSTEM.value == "system"
    assert PromptMemoryView().total_items == 0


def test_init_creates_directories(tmp_path: Path) -> None:
    store = MemFSStore(tmp_path / "memfs")

    store.init()

    for dirname in EXPECTED_DIRS:
        assert (tmp_path / "memfs" / dirname).is_dir()


def test_init_idempotent(tmp_path: Path) -> None:
    store = MemFSStore(tmp_path)
    store.init()
    keep = tmp_path / "pinned" / "keep.md"
    keep.write_text("keep me", encoding="utf-8")

    store.init()

    assert keep.read_text(encoding="utf-8") == "keep me"


def test_read_file_parses_frontmatter(tmp_path: Path) -> None:
    write_memory_file(
        tmp_path,
        "pinned/user.md",
        description="User preferences",
        authority="pinned",
        mode_scope=["work"],
        buckets=["user_preference"],
        body="Teacher prefers exact evidence.",
    )
    store = MemFSStore(tmp_path)

    frontmatter, body = store.read_file("pinned/user.md")

    assert isinstance(frontmatter, MemoryFileFrontmatter)
    assert frontmatter.description == "User preferences"
    assert frontmatter.authority == "pinned"
    assert frontmatter.mode_scope == ("work",)
    assert frontmatter.buckets == ("user_preference",)
    assert frontmatter.memory_type == "UserPreference"
    assert frontmatter.lifecycle_state == "active"
    assert "exact evidence" in body


def test_read_file_rejects_path_traversal(tmp_path: Path) -> None:
    store = MemFSStore(tmp_path)

    with pytest.raises(ValueError):
        store.read_file("../../etc/passwd")


@pytest.mark.parametrize("relative_path", ["/etc/passwd"])
def test_read_file_rejects_absolute_path(tmp_path: Path, relative_path: str) -> None:
    store = MemFSStore(tmp_path)

    with pytest.raises(ValueError):
        store.read_file(relative_path)


def test_read_file_skips_no_frontmatter(tmp_path: Path) -> None:
    path = tmp_path / "system" / "bad.md"
    path.parent.mkdir(parents=True)
    path.write_text("No frontmatter\nFull body", encoding="utf-8")
    store = MemFSStore(tmp_path)

    with pytest.raises(ValueError):
        store.read_file("system/bad.md")


def test_list_tree_returns_progressive_disclosure(tmp_path: Path) -> None:
    write_memory_file(
        tmp_path,
        "system/runtime.md",
        description="Runtime boundary",
        authority="system",
        body="SECRET FULL BODY SHOULD NOT APPEAR",
    )
    store = MemFSStore(tmp_path)

    items = store.list_tree()

    assert len(items) == 1
    assert isinstance(items[0], MemoryLayerItem)
    assert items[0].path == "system/runtime.md"
    assert items[0].description == "Runtime boundary"
    assert items[0].body == ""
    assert items[0].id == "system/runtime.md"
    assert items[0].memory_type == "UserPreference"
    assert items[0].lifecycle_state == "active"
    assert "SECRET" not in repr(items[0])


def test_list_tree_empty(tmp_path: Path) -> None:
    store = MemFSStore(tmp_path)

    assert store.list_tree() == []


def test_memfs_search_and_open_redact_secret_values(tmp_path: Path) -> None:
    fake_secret = "PASSWORD=hunter2"
    write_memory_file(
        tmp_path,
        "episodic/dirty.md",
        description="Dirty legacy memory",
        authority="episodic",
        body=f"Legacy body contains {fake_secret} for search/open testing.",
    )
    store = MemFSStore(tmp_path)

    view = store.load_layer("episodic", mode="work", query="legacy", budget_chars=1000)
    assert fake_secret not in str(view)
    assert "[REDACTED_SECRET]" in str(view)

    hits = store.search("legacy", layers=["episodic"], mode="work", limit=5)
    assert hits
    assert fake_secret not in str(hits)
    assert "[REDACTED_SECRET]" in str(hits)

    opened = store.open_memory("episodic/dirty.md")
    assert fake_secret not in str(opened)
    assert "[REDACTED_SECRET]" in str(opened)


def test_load_layer_system_returns_all(tmp_path: Path) -> None:
    write_memory_file(tmp_path, "system/a.md", description="A", authority="system", body="A body")
    write_memory_file(tmp_path, "system/b.md", description="B", authority="system", body="B body")
    store = MemFSStore(tmp_path)

    view = store.load_layer("system", mode="sex", query="unrelated", budget_chars=100)

    assert [item.path for item in view.items] == ["system/a.md", "system/b.md"]
    assert [item.body.strip() for item in view.items] == ["A body", "B body"]


def test_load_layer_pinned_filters_mode_scope(tmp_path: Path) -> None:
    write_memory_file(
        tmp_path,
        "pinned/work.md",
        description="Work item",
        authority="pinned",
        mode_scope=["work"],
    )
    write_memory_file(
        tmp_path,
        "pinned/daily.md",
        description="Daily item",
        authority="pinned",
        mode_scope=["daily"],
    )
    store = MemFSStore(tmp_path)

    view = store.load_layer("pinned", mode="work")

    assert [item.path for item in view.items] == ["pinned/work.md"]


def test_load_layer_pinned_filters_query(tmp_path: Path) -> None:
    write_memory_file(
        tmp_path,
        "pinned/z-runtime.md",
        description="Runtime operations",
        authority="pinned",
        buckets=["runtime_boundary"],
        body="Operational facts.",
    )
    write_memory_file(
        tmp_path,
        "pinned/a-other.md",
        description="Other item",
        authority="pinned",
        buckets=["relationship"],
        body="Other facts.",
    )
    store = MemFSStore(tmp_path)

    view = store.load_layer("pinned", mode="work", query="runtime")

    assert [item.path for item in view.items] == ["pinned/z-runtime.md", "pinned/a-other.md"]
    assert view.items[0].score > view.items[1].score


def test_load_layer_episodic_sorts_by_recency(tmp_path: Path) -> None:
    write_memory_file(
        tmp_path,
        "episodic/2025/12/old.md",
        description="Old",
        authority="episodic",
        body="old event",
    )
    write_memory_file(
        tmp_path,
        "episodic/2026/05/new.md",
        description="New",
        authority="episodic",
        body="new event",
    )
    store = MemFSStore(tmp_path)

    view = store.load_layer("episodic", mode="work")

    assert [item.path for item in view.items] == [
        "episodic/2026/05/new.md",
        "episodic/2025/12/old.md",
    ]


def test_load_layer_episodic_query_hit_first(tmp_path: Path) -> None:
    write_memory_file(
        tmp_path,
        "episodic/2026/05/new.md",
        description="New unrelated",
        authority="episodic",
        body="ordinary event",
    )
    write_memory_file(
        tmp_path,
        "episodic/2025/12/old.md",
        description="Old matching",
        authority="episodic",
        body="contains rollback evidence",
    )
    store = MemFSStore(tmp_path)

    view = store.load_layer("episodic", mode="work", query="rollback")

    assert [item.path for item in view.items] == [
        "episodic/2025/12/old.md",
        "episodic/2026/05/new.md",
    ]


def test_load_layer_transient_returns_current(tmp_path: Path) -> None:
    write_memory_file(
        tmp_path,
        "transient/current-session.md",
        description="Current session",
        authority="transient",
        mode_scope=["daily"],
        body="Current task ledger.",
    )
    store = MemFSStore(tmp_path)

    view = store.load_layer("transient", mode="work", query="nothing")

    assert [item.path for item in view.items] == ["transient/current-session.md"]
    assert "Current task ledger" in view.items[0].body


def test_load_layer_respects_budget(tmp_path: Path) -> None:
    write_memory_file(tmp_path, "system/a.md", description="A", authority="system", body="12345")
    write_memory_file(tmp_path, "system/b.md", description="B", authority="system", body="12345")
    store = MemFSStore(tmp_path)

    view = store.load_layer("system", budget_chars=7)

    assert [item.path for item in view.items] == ["system/a.md"]
    assert view.omitted_count == 1
    assert view.used_chars == view.items[0].char_count


def test_load_layer_omits_item_larger_than_budget(tmp_path: Path) -> None:
    write_memory_file(tmp_path, "system/large.md", description="Large", authority="system", body="x" * 20)
    store = MemFSStore(tmp_path)

    view = store.load_layer("system", budget_chars=10)

    assert view.items == []
    assert view.omitted_count == 1


def test_load_layer_invalid_layer_raises(tmp_path: Path) -> None:
    store = MemFSStore(tmp_path)

    with pytest.raises(ValueError):
        store.load_layer("skills")


def test_load_layer_returns_memory_layer_view(tmp_path: Path) -> None:
    write_memory_file(tmp_path, "pinned/item.md", authority="pinned")
    store = MemFSStore(tmp_path)

    view = store.load_layer("pinned", mode="work", budget_chars=2000)

    assert isinstance(view, MemoryLayerView)
    assert view.layer == "pinned"
    assert view.items[0].authority == "pinned"


def test_write_file_atomically_preserves_previous_content_on_replace_failure(tmp_path: Path) -> None:
    store = MemFSStore(tmp_path)
    path = tmp_path / "pinned" / "item.md"
    frontmatter = MemoryFileFrontmatter(description="Atomic write")
    store.write_file("pinned/item.md", frontmatter, "old body")
    previous = path.read_bytes()

    with patch.object(Path, "replace", autospec=True, side_effect=OSError("replace failed")):
        with pytest.raises(OSError, match="replace failed"):
            store.write_file("pinned/item.md", frontmatter, "new body")

    assert path.read_bytes() == previous
    assert not list(path.parent.glob(f".{path.name}.*.tmp"))
