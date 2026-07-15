from pcltm.context_budget import (
    ContextBudgetBucket,
    estimate_context_budget,
    estimate_tokens,
)
from pcltm.memory_object import InjectionPolicy, MemoryObject, MemoryObjectType
from pcltm.memory_selection import SelectionDecision, PriorityClass


def _memory(key, content, *, injection_policy=InjectionPolicy.SELECTIVE):
    return MemoryObject(
        canonical_key=key,
        object_type=MemoryObjectType.PREFERENCE,
        content=content,
        injection_policy=injection_policy,
    )


def test_estimate_context_budget_reports_typed_buckets_without_mutation():
    pinned = _memory("core.identity", "identity memory", injection_policy=InjectionPolicy.PINNED)
    selected = _memory("pref.response", "selected preference memory")
    ignored = _memory("pref.ignored", "this should not count")
    decisions = (
        SelectionDecision(
            selected=True,
            reason="selected",
            canonical_key="pref.response",
            priority_class=PriorityClass.SELECTIVE,
            budget_weight=1.0,
            matched_mode=None,
            matched_emotion_axes=(),
            rejected_reason=None,
        ),
    )

    report = estimate_context_budget(
        context_window=1000,
        active_frame="active frame text",
        pinned_memories=(pinned,),
        selection_decisions=decisions,
        memory_objects=(selected, ignored),
        transcript_messages=(
            {"role": "user", "content": "latest request"},
            {"role": "assistant", "content": "current response"},
        ),
        tool_evidence=({"role": "tool", "tool_call_id": "call_1", "content": "evidence"},),
        reference_items=("reference handoff",),
    )

    assert report.within_budget
    assert report.line_for(ContextBudgetBucket.ACTIVE_FRAME).token_count == estimate_tokens("active frame text")
    assert report.line_for(ContextBudgetBucket.PINNED_MEMORY).token_count == estimate_tokens("identity memory")
    assert report.line_for(ContextBudgetBucket.SELECTED_MEMORY).token_count == estimate_tokens("selected preference memory")
    assert report.line_for(ContextBudgetBucket.SELECTED_MEMORY).token_count != estimate_tokens("this should not count")
    assert report.line_for(ContextBudgetBucket.TOOL_EVIDENCE).token_count > 0
    assert report.to_dict()["lines"][0]["bucket"] == "active_frame"


def test_estimate_context_budget_marks_bucket_truncation_and_overflow():
    report = estimate_context_budget(
        context_window=20,
        active_frame="x" * 400,
        transcript_messages=({"role": "user", "content": "y" * 400},),
    )

    assert not report.within_budget
    assert report.over_budget_tokens > 0
    assert report.line_for(ContextBudgetBucket.ACTIVE_FRAME).truncated
    assert report.line_for(ContextBudgetBucket.TRANSCRIPT).truncated


def test_estimate_context_budget_rejects_invalid_window():
    try:
        estimate_context_budget(context_window=0)
    except ValueError as exc:
        assert "context_window" in str(exc)
    else:
        raise AssertionError("expected invalid context_window to fail")
