from __future__ import annotations

from pathlib import Path

from pcltm import memory_adapter
from pcltm.memfs_types import PromptMemoryView


def _write_legacy_memfs(root: Path, body: str = "LEGACY_BODY_SENTINEL") -> None:
    path = root / "pinned" / "legacy.md"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "---\n"
        "description: ungoverned legacy body\n"
        "authority: pinned\n"
        "mode_scope: [work]\n"
        "buckets: [runtime_boundary]\n"
        "---\n"
        f"{body}\n",
        encoding="utf-8",
    )


def test_layered_prompt_context_is_fail_closed_for_arbitrary_memfs(tmp_path: Path) -> None:
    _write_legacy_memfs(tmp_path)

    view = memory_adapter.load_layered_prompt_context(
        mode="work", query="legacy", root=tmp_path,
        layers=["system", "pinned", "episodic", "transient"],
        active_layers=["system", "pinned", "transient"],
        buckets=["runtime_boundary"],
    )

    assert isinstance(view, PromptMemoryView)
    assert view.selection_source == "retired_legacy_memfs_prompt"
    assert view.render() == ""
    assert view.total_items == 0
    assert "LEGACY_BODY_SENTINEL" not in view.render()
    assert view.compression.is_reference_only is True


def test_context_snapshot_reports_retired_empty_surface(tmp_path: Path) -> None:
    _write_legacy_memfs(tmp_path, "SNAPSHOT_BODY_SENTINEL")

    payload = memory_adapter.select_context_snapshot(
        mode="work", query="snapshot", root=tmp_path,
        active_layers=["system", "pinned"],
    ).to_dict()

    assert payload["object_type"] == "pcltm_context_selection_snapshot"
    assert payload["selection_source"] == "retired_legacy_memfs_prompt"
    assert payload["total_selected_items"] == 0
    assert all(layer["selected_count"] == 0 for layer in payload["layers"])
    assert "SNAPSHOT_BODY_SENTINEL" not in str(payload)



def test_layered_prompt_context_never_reads_configured_legacy_db(tmp_path: Path, monkeypatch) -> None:
    db = tmp_path / "legacy.db"
    db.write_bytes(b"not even a sqlite database")
    root = tmp_path / "memfs"
    _write_legacy_memfs(root, "DB_AND_MEMFS_SENTINEL")
    monkeypatch.setenv("HERMES_PCLTM_DB", str(db))

    view = memory_adapter.load_layered_prompt_context(mode="daily", root=root)

    assert view.render() == ""
    assert view.selection_source == "retired_legacy_memfs_prompt"
