from __future__ import annotations

from pathlib import Path

from pcltm.memory_adapter import load_layered_prompt_context


def _write_memfs(root: Path, body: str) -> None:
    path = root / "pinned" / "legacy.md"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "---\ndescription: legacy integration fixture\nauthority: pinned\n"
        "mode_scope: [work, daily]\nbuckets: [runtime_boundary]\n---\n"
        f"{body}\n",
        encoding="utf-8",
    )


def test_memfs_prompt_context_surface_is_retired_in_work_mode(tmp_path: Path) -> None:
    _write_memfs(tmp_path, "WORK_LEGACY_SENTINEL")
    view = load_layered_prompt_context(mode="work", root=tmp_path)
    assert view.selection_source == "retired_legacy_memfs_prompt"
    assert view.render() == ""
    assert view.total_items == 0


def test_memfs_prompt_context_surface_is_retired_in_daily_mode(tmp_path: Path) -> None:
    _write_memfs(tmp_path, "DAILY_LEGACY_SENTINEL")
    view = load_layered_prompt_context(mode="daily", root=tmp_path)
    assert view.selection_source == "retired_legacy_memfs_prompt"
    assert view.render() == ""
    assert view.total_items == 0
