from __future__ import annotations

from pathlib import Path

from pcltm.governance import (
    MemoryGovernanceOrchestrator,
    MemoryLifecycleLedger,
    MemoryLifecycleTransition,
)
from pcltm.memfs_store import MemFSStore


def write_memory(store: MemFSStore, relative_path: str, body: str, *, description: str = "Test memory") -> None:
    path = store.root / relative_path
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "---\n"
        f"description: {description}\n"
        "authority: pinned\n"
        "mode_scope: [work]\n"
        "buckets: [test]\n"
        "source: test\n"
        "last_reviewed: ''\n"
        "---\n"
        f"{body}",
        encoding="utf-8",
    )


class FakeEventStore:
    def __init__(self, records=None):
        self._records = records or []

    def list_memory_records(self, *, status=None, target_file=None):
        rows = list(self._records)
        if status is not None:
            rows = [row for row in rows if row.get("status") == status]
        if target_file is not None:
            rows = [row for row in rows if row.get("target_file") == target_file]
        return rows


def test_governance_report_is_dry_run_and_combines_pending_and_defrag(tmp_path: Path) -> None:
    from pcltm import MemoryGovernanceOrchestrator as ExportedOrchestrator

    assert ExportedOrchestrator is MemoryGovernanceOrchestrator

    store = MemFSStore(tmp_path / "memfs")
    store.init()
    write_memory(store, "pinned/a.md", "duplicate body")
    write_memory(store, "pinned/b.md", "duplicate body")

    event_store = FakeEventStore(
        [
            {
                "record_id": 1,
                "candidate_id": "cand-1",
                "kind": "reflection",
                "target_file": "MEMORY.md",
                "content": "candidate content",
                "status": "pending",
                "metadata": {"source": "reflection"},
            }
        ]
    )

    report = MemoryGovernanceOrchestrator(memfs_store=store, event_store=event_store).analyze()

    assert report.dry_run is True
    assert report.pending_candidates == 1
    assert report.defrag_plan.duplicate_count == 1
    assert report.actions[0].kind == "review_memory_candidate"
    assert report.actions[0].requires_human_review is True
    assert isinstance(report.lifecycle_ledger, MemoryLifecycleLedger)
    assert report.lifecycle_ledger.transition_count == len(report.actions)
    assert report.lifecycle_ledger.risk_level == "medium"
    first_transition = report.lifecycle_ledger.transitions[0]
    assert isinstance(first_transition, MemoryLifecycleTransition)
    assert first_transition.transition_id.startswith("mlt_")
    assert first_transition.object_id == "cand-1"
    assert first_transition.object_kind == "memory_candidate"
    assert first_transition.current_state == "pending"
    assert first_transition.proposed_state == "review_required"
    assert first_transition.authority_boundary == "candidate_memory"
    assert first_transition.risk_level == "medium"
    assert first_transition.evidence_refs == (
        {"type": "memory_record", "id": 1},
        {"type": "candidate", "id": "cand-1"},
    )
    serialized_ledger = report.lifecycle_ledger.to_dict()
    serialized_report = report.to_dict()
    assert serialized_report["dry_run"] is True
    assert serialized_report["pending_candidates"] == 1
    assert serialized_report["defrag_plan"]["duplicate_count"] == 1
    assert serialized_report["actions"][0]["kind"] == "review_memory_candidate"
    assert serialized_report["lifecycle_ledger"] == serialized_ledger
    assert serialized_ledger["schema_version"] == 1
    assert serialized_ledger["transition_count"] == len(report.actions)
    assert serialized_ledger["transitions"][0]["transition_id"] == first_transition.transition_id
    assert any(action.kind == "defrag_merge" for action in report.actions)


def test_governance_report_does_not_apply_defrag_by_default(tmp_path: Path) -> None:
    store = MemFSStore(tmp_path / "memfs")
    store.init()
    write_memory(store, "pinned/a.md", "duplicate body")
    write_memory(store, "pinned/b.md", "duplicate body")

    report = MemoryGovernanceOrchestrator(memfs_store=store, event_store=FakeEventStore()).analyze()

    assert report.defrag_plan.duplicate_count == 1
    assert (store.root / "pinned" / "a.md").exists()
    assert (store.root / "pinned" / "b.md").exists()


def test_governance_execute_refuses_system_defrag_without_flag(tmp_path: Path) -> None:
    store = MemFSStore(tmp_path / "memfs")
    store.init()
    write_memory(store, "system/a.md", "duplicate body")
    write_memory(store, "system/b.md", "duplicate body")

    result = MemoryGovernanceOrchestrator(memfs_store=store, event_store=FakeEventStore()).execute(
        apply_defrag=True,
    )

    assert result.dry_run is False
    assert (store.root / "system" / "a.md").exists()
    assert (store.root / "system" / "b.md").exists()
    assert any(action.kind == "blocked_system_defrag" for action in result.actions)
    blocked_transition = next(
        transition
        for transition in result.lifecycle_ledger.transitions
        if transition.trigger == "blocked_system_defrag"
    )
    assert blocked_transition.authority_boundary == "system_memory"
    assert blocked_transition.risk_level == "high"
    assert blocked_transition.current_state == "blocked"
    assert blocked_transition.proposed_state == "blocked_until_explicit_authority"
    assert result.lifecycle_ledger.blocked_count == 1


def test_governance_execute_can_apply_explicitly(tmp_path: Path) -> None:
    class ReviewableEventStore(FakeEventStore):
        def __init__(self, records=None):
            super().__init__(records)
            self.review_calls = []

        def review_candidate(self, record_id, *, decision, reviewer, decision_reason):
            self.review_calls.append((record_id, decision, reviewer, decision_reason))
            return {"record_id": record_id, "status": decision}

    store = MemFSStore(tmp_path / "memfs")
    store.init()
    write_memory(store, "pinned/a.md", "duplicate body")
    write_memory(store, "pinned/b.md", "duplicate body")

    event_store = ReviewableEventStore(
        [
            {
                "record_id": 7,
                "candidate_id": "cand-7",
                "kind": "reflection",
                "target_file": "MEMORY.md",
                "content": "candidate content",
                "status": "pending",
                "metadata": {"source": "reflection"},
            }
        ]
    )

    orchestrator = MemoryGovernanceOrchestrator(memfs_store=store, event_store=event_store)
    result = orchestrator.execute(
        reflection_events=[
            {
                "type": "preference_confirmed",
                "user_message": "用户不要再使用上下文压缩",
                "mode": "work",
                "timestamp": "2026-05-24T00:00:00Z",
            }
        ],
        approve_pending=True,
        write_reflection_drafts=True,
        apply_defrag=True,
        reviewer="tester",
    )

    assert result.dry_run is False
    assert event_store.review_calls == [(7, "approved", "tester", "approved by governance execute")]
    assert any((store.root / "episodic").rglob("*.md"))
    assert not (store.root / "pinned" / "b.md").exists()


def test_governance_report_can_include_reflection_drafts_without_writing(tmp_path: Path) -> None:
    store = MemFSStore(tmp_path / "memfs")
    store.init()
    events = [
        {
            "type": "preference_confirmed",
            "user_message": "用户不要再使用上下文压缩",
            "mode": "work",
            "timestamp": "2026-05-24T00:00:00Z",
        }
    ]

    report = MemoryGovernanceOrchestrator(memfs_store=store, event_store=FakeEventStore()).analyze(
        reflection_events=events
    )

    assert report.reflection_drafts == 1
    reflection_action = next(action for action in report.actions if action.kind == "write_reflection_draft")
    assert reflection_action.requires_human_review is True
    assert not any((store.root / "episodic").rglob("*.md"))
