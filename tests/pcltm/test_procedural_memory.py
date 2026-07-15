from datetime import UTC, datetime

from pcltm.procedural import (
    ExistingSkillStatus,
    ProceduralMemory,
    SkillCandidateExtractor,
    SkillUpdateAction,
    TaskTrace,
    merge_procedural_memory,
    operation_for_skill_manage,
)
from pcltm.procedural.skill_exporter import SkillExporter


def test_extracts_reusable_skill_candidate_from_complex_task_trace() -> None:
    trace = TaskTrace(
        title="Debug pytest failure workflow",
        messages=(
            "Trigger: when a Python test failure needs reusable debugging discipline",
            "Procedure: reproduce the failure with the narrowest test command",
            "Procedure: inspect the failing code and nearby tests before editing",
            "Procedure: patch the root cause instead of only changing assertions",
            "Verification: rerun the narrow test and then the relevant test file",
            "Pitfall: do not report success until real tool output verifies it",
        ),
        tool_calls=("read_file", "search_files", "terminal", "patch", "terminal"),
        outcomes=("Verification: pytest tests/pcltm/test_example.py -q passed",),
        source_session="session-debug-1",
        category="software-development",
    )

    candidate = SkillCandidateExtractor().extract(trace)

    assert candidate.accepted is True
    assert candidate.memory is not None
    assert candidate.memory.skill_name == "debug-pytest-failure-workflow"
    assert candidate.memory.category == "software-development"
    assert candidate.memory.source_sessions == ("session-debug-1",)
    assert len(candidate.memory.procedure) >= 3
    assert candidate.memory.verification_steps
    assert candidate.memory.pitfalls
    assert candidate.memory.is_exportable is True


def test_rejects_one_off_task_status_even_when_tool_count_is_high() -> None:
    trace = TaskTrace(
        title="Phase 5 done",
        messages=(
            "阶段 5 完工，提交 commit abcdef1234567890",
            "今天已完成 PR #42",
            "当前进度已同步",
        ),
        tool_calls=("read_file", "terminal", "patch", "terminal", "terminal", "git"),
        outcomes=("completed",),
        source_session="session-status-only",
    )

    candidate = SkillCandidateExtractor().extract(trace)

    assert candidate.accepted is False
    assert candidate.memory is None
    assert "one-off task status" in candidate.reasons[0]


def test_procedural_memory_rejects_log_shaped_export() -> None:
    memory = ProceduralMemory(
        skill_name="phase-5-completion-log",
        trigger_conditions=("Phase 5 completed",),
        procedure=("submitted PR #42", "commit abcdef1234567890"),
        verification_steps=("done",),
        confidence=0.95,
    )

    assert memory.looks_like_task_log() is True
    assert memory.is_exportable is False


def test_exporter_creates_skill_manage_payload_for_missing_skill() -> None:
    memory = ProceduralMemory(
        skill_name="pytest-debugging-workflow",
        trigger_conditions=("when Python tests fail and need evidence-first debugging",),
        procedure=(
            "Reproduce the failure with the narrowest command",
            "Inspect code and tests before editing",
            "Patch the root cause",
        ),
        verification_steps=("Rerun the narrow test and relevant file",),
        pitfalls=("Do not turn task status into skill content",),
        source_sessions=("session-1",),
        confidence=0.9,
        category="software-development",
    )

    plan = SkillExporter().plan(memory)
    payload = operation_for_skill_manage(plan)

    assert plan.action == SkillUpdateAction.CREATE
    assert payload is not None
    assert payload["action"] == "create"
    assert payload["name"] == "pytest-debugging-workflow"
    assert payload["category"] == "software-development"
    assert "PCLTM procedural memory records" in str(payload["content"])
    assert "Do not add one-off task progress" in str(payload["content"])


def test_exporter_patches_wrong_or_outdated_existing_skill() -> None:
    memory = ProceduralMemory(
        skill_name="pytest-debugging-workflow",
        trigger_conditions=("when Python tests fail",),
        procedure=("Reproduce", "Inspect", "Patch"),
        verification_steps=("Rerun pytest",),
        confidence=0.8,
    )
    existing = "---\nname: pytest-debugging-workflow\n---\n\n# Old\n\n## Relationship to PCLTM\nOld body"

    plan = SkillExporter().plan(
        memory,
        existing_skill=existing,
        existing_status=ExistingSkillStatus.WRONG,
    )
    payload = operation_for_skill_manage(plan)

    assert plan.action == SkillUpdateAction.PATCH
    assert payload is not None
    assert payload["action"] == "patch"
    assert payload["old_string"] == "## Relationship to PCLTM\nOld body"
    assert "## Procedure" in str(payload["new_string"])
    assert "Do not store task progress" in str(payload["new_string"])


def test_exporter_skips_current_skill_and_non_exportable_memory() -> None:
    good = ProceduralMemory(
        skill_name="safe-workflow",
        trigger_conditions=("when reusable workflow exists",),
        procedure=("Step one", "Step two"),
        verification_steps=("Verify result",),
        confidence=0.8,
    )
    current_plan = SkillExporter().plan(
        good,
        existing_skill="current",
        existing_status=ExistingSkillStatus.CURRENT,
    )
    assert current_plan.action == SkillUpdateAction.SKIP
    assert operation_for_skill_manage(current_plan) is None

    weak = ProceduralMemory(
        skill_name="weak-workflow",
        trigger_conditions=("when weak",),
        procedure=("Only one step",),
        verification_steps=(),
        confidence=0.2,
    )
    weak_plan = SkillExporter().plan(weak)
    assert weak_plan.action == SkillUpdateAction.SKIP
    assert operation_for_skill_manage(weak_plan) is None


def test_merge_procedural_memory_combines_reusable_parts_without_duplicates() -> None:
    old = ProceduralMemory(
        skill_name="evidence-first-debugging",
        trigger_conditions=("when debugging failures",),
        procedure=("Reproduce failure", "Inspect evidence"),
        verification_steps=("Run focused test",),
        pitfalls=("Do not guess",),
        source_sessions=("s1",),
        last_updated=datetime(2026, 1, 1, tzinfo=UTC),
        confidence=0.7,
    )
    new = ProceduralMemory(
        skill_name="evidence-first-debugging",
        trigger_conditions=("when debugging failures", "when fixing regressions"),
        procedure=("Inspect evidence", "Patch root cause"),
        verification_steps=("Run focused test", "Run regression file"),
        pitfalls=("Do not guess", "Do not save progress as skill"),
        source_sessions=("s2",),
        last_updated=datetime(2026, 2, 1, tzinfo=UTC),
        confidence=0.9,
    )

    merged = merge_procedural_memory(old, new)

    assert merged.trigger_conditions == ("when debugging failures", "when fixing regressions")
    assert merged.procedure == ("Reproduce failure", "Inspect evidence", "Patch root cause")
    assert merged.verification_steps == ("Run focused test", "Run regression file")
    assert merged.pitfalls == ("Do not guess", "Do not save progress as skill")
    assert merged.source_sessions == ("s1", "s2")
    assert merged.confidence == 0.9
    assert merged.last_updated == datetime(2026, 2, 1, tzinfo=UTC)
