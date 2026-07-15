from __future__ import annotations

from pathlib import Path

from pcltm.memfs_store import MemFSStore
from pcltm.defrag import MemFSDefragger


def write_memory(
    store: MemFSStore,
    relative_path: str,
    body: str,
    *,
    description: str = "Test memory",
    memory_type: str = "UserPreference",
    ttl: str = "none",
    authority: str = "pinned",
    mode_scope: str = "[daily, work, sex]",
) -> Path:
    path = store.root / relative_path
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "---\n"
        f"description: {description}\n"
        f"authority: {authority}\n"
        f"mode_scope: {mode_scope}\n"
        "buckets: [test]\n"
        "source: test\n"
        "last_reviewed: ''\n"
        "metadata:\n"
        f"  memory_type: {memory_type}\n"
        f"  ttl: {ttl}\n"
        "---\n"
        f"{body}",
        encoding="utf-8",
    )
    return path


def test_defrag_dry_run_reports_stale_task_archival_and_duplicate_merge(tmp_path: Path) -> None:
    store = MemFSStore(tmp_path / "memfs")
    store.init()
    write_memory(store, "transient/old-task.md", "旧临时任务状态", memory_type="TemporaryTaskState", ttl="short")
    write_memory(store, "pinned/a.md", "same stable preference")
    write_memory(store, "pinned/b.md", "same stable preference")

    report = MemFSDefragger(store).analyze_report()

    assert report.schema_version == 1
    assert report.dry_run is True
    assert report.authority_boundary == "read_only_defrag_governance"
    assert report.merged_count == 1
    assert report.archived_count == 1
    assert report.stale_task_count == 1
    assert report.high_risk_changes == 0
    assert report.needs_review is True
    assert any(action.action_type == "archive" and action.source_path == "transient/old-task.md" for action in report.actions)
    serialized = report.to_dict()
    assert serialized["merged_count"] == 1
    assert serialized["actions"][0]["risk_level"] in {"low", "medium"}


def test_defrag_report_marks_system_layer_changes_high_risk(tmp_path: Path) -> None:
    store = MemFSStore(tmp_path / "memfs")
    store.init()
    write_memory(store, "system/a.md", "same system rule", authority="system", memory_type="RuntimeInvariant")
    write_memory(store, "system/b.md", "same system rule", authority="system", memory_type="RuntimeInvariant")

    report = MemFSDefragger(store).analyze_report()

    assert report.high_risk_changes == 1
    assert report.needs_review is True
    high = [action for action in report.actions if action.risk_level == "high"]
    assert high
    assert high[0].requires_human_review is True
    assert high[0].authority_boundary == "system_memory"


def test_defrag_report_is_read_only(tmp_path: Path) -> None:
    store = MemFSStore(tmp_path / "memfs")
    store.init()
    duplicate = write_memory(store, "pinned/b.md", "same body")
    write_memory(store, "pinned/a.md", "same body")

    report = MemFSDefragger(store).analyze_report()

    assert report.dry_run is True
    assert duplicate.exists()
    assert report.to_dict()["dry_run"] is True
