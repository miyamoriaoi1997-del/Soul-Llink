from __future__ import annotations

import json

from pcltm.live_context_governor import (
    ContextBudgetPolicy,
    ContinuationCapsule,
    RecallIntent,
    ToolEvidenceCapsule,
    classify_recall_intent,
    govern_prompt_context,
)


def test_govern_prompt_context_preserves_task_and_caps_total_budget() -> None:
    policy = ContextBudgetPolicy(total_chars=420, continuation_chars=220, evidence_chars=160, memory_chars=180)
    capsule = ContinuationCapsule(
        conversation_goal="吸收成熟 memory runtime 的优点并增强 PCLTM live context budget governance",
        current_task="实现 Live Context Budget Governor v0，严格控制预算但保留任务目标",
        completed=("已提交 governance observability", "已验证 303 passed"),
        open_threads=("不要推远端，先本地观察", "后续稳定再 fetch/rebase/test/push"),
        constraints=("不启用 Hermes built-in compression", "不自动删除长期记忆"),
        latest_verified_state={"pytest": "303 passed", "governance_smoke": "ok=True error_count=0"},
    )
    evidence = ToolEvidenceCapsule.from_tool_output(
        command="python -m pytest -q",
        exit_code=0,
        output="." * 2000 + "\n303 passed in 8.66s\n",
        affected_files=("packages/pcltm/live_context_governor.py",),
    )
    raw_context = "<pcltm_context>\n" + ("低优先级历史记录\n" * 200) + "</pcltm_context>"

    governed = govern_prompt_context(
        raw_context,
        policy=policy,
        continuation_capsule=capsule,
        tool_evidence=(evidence,),
    )

    assert len(governed.rendered) <= policy.total_chars
    assert "current_task" in governed.rendered
    assert "Live Context Budget Governor v0" in governed.rendered
    assert "303 passed" in governed.rendered
    assert "omitted" in governed.telemetry["actions"]
    assert governed.telemetry["within_budget"] is True
    assert governed.telemetry["capsules"]["continuation"] == 1
    assert governed.telemetry["capsules"]["tool_evidence"] == 1


def test_govern_prompt_context_can_render_compatible_pcltm_context_envelope() -> None:
    policy = ContextBudgetPolicy(total_chars=260, continuation_chars=0, evidence_chars=0, memory_chars=120)

    governed = govern_prompt_context(
        "<pcltm_context>\n原始记忆\n</pcltm_context>",
        policy=policy,
        outer_tag="pcltm_context",
    )

    assert len(governed.rendered) <= policy.total_chars
    assert governed.rendered.startswith("<pcltm_context>")
    assert governed.rendered.endswith("</pcltm_context>")
    assert governed.rendered.count("<pcltm_context>") == 1
    assert governed.rendered.count("</pcltm_context>") == 1
    assert "【governed_memory_view】" in governed.rendered


def test_tool_evidence_capsule_is_short_hashed_and_secret_safe() -> None:
    capsule = ToolEvidenceCapsule.from_tool_output(
        command="pytest tests/pcltm -q",
        exit_code=0,
        output="TOKEN=super-secret-value\n" + "x" * 1000 + "\n260 passed in 5.23s",
        affected_files=("tests/pcltm/test_live_context_budget_governor.py",),
    )

    rendered = capsule.render(max_chars=220)

    assert len(rendered) <= 220
    assert "super-secret-value" not in rendered
    assert "[REDACTED_SECRET]" in rendered
    assert capsule.evidence_hash.startswith("sha256:")
    assert "260 passed" in rendered


def test_recall_intent_gates_memory_targets_and_buckets() -> None:
    context_intent = classify_recall_intent("把当前上下文链路和PCLTM预算治理整理出来")
    git_intent = classify_recall_intent("稳定后 fetch rebase test push 到远端")
    relationship_intent = classify_recall_intent("我想抱抱你，今天有点累")

    assert context_intent.intent is RecallIntent.CONTEXT_DIAGNOSTICS
    assert "runtime_boundary" in context_intent.allowed_buckets
    assert "relationship" not in context_intent.allowed_buckets
    assert context_intent.allow_user_preferences is False

    assert git_intent.intent is RecallIntent.GIT_WORKFLOW
    assert "workflow" in git_intent.allowed_buckets
    assert git_intent.allow_user_preferences is True

    assert relationship_intent.intent is RecallIntent.RELATIONSHIP
    assert "relationship" in relationship_intent.allowed_buckets
    assert "runtime_boundary" not in relationship_intent.allowed_buckets


def test_continuation_capsule_round_trips_json_and_never_expands_unbounded() -> None:
    capsule = ContinuationCapsule(
        conversation_goal="goal " * 100,
        current_task="task " * 100,
        completed=tuple(f"done {i}" for i in range(20)),
        open_threads=tuple(f"thread {i}" for i in range(20)),
        constraints=tuple(f"constraint {i}" for i in range(20)),
        latest_verified_state={"pytest": "303 passed", "cli": "ok=True"},
    )

    data = capsule.to_dict()
    restored = ContinuationCapsule.from_dict(json.loads(json.dumps(data, ensure_ascii=False)))
    rendered = restored.render(max_chars=360)

    assert restored.current_task
    assert len(rendered) <= 360
    assert "continuation_capsule" in rendered
    assert "pytest" in rendered
