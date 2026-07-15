from pcltm.memory_object import (
    InjectionPolicy,
    MemoryObject,
    MemoryObjectStatus,
    MemoryObjectType,
    StateAffinity,
)
from pcltm.memory_selection import PriorityClass, explain_memory_selection


def test_pinned_identity_is_selected_with_highest_priority():
    obj = MemoryObject(
        canonical_key="identity-anchor",
        object_type=MemoryObjectType.IDENTITY,
        content="Stable identity anchor.",
        status=MemoryObjectStatus.APPROVED,
        injection_policy=InjectionPolicy.PINNED,
    )

    decision = explain_memory_selection(obj, mode="work", emotion_axes=set(), budget_available=1.0)

    assert decision.selected is True
    assert decision.reason == "pinned identity"
    assert decision.rejected_reason is None
    assert decision.priority_class is PriorityClass.PINNED_IDENTITY
    assert decision.budget_weight == 1.0


def test_approved_selective_memory_is_selected_when_state_matches():
    obj = MemoryObject(
        canonical_key="work-pref",
        object_type=MemoryObjectType.PREFERENCE,
        content="Work mode response preference.",
        status=MemoryObjectStatus.APPROVED,
        injection_policy=InjectionPolicy.SELECTIVE,
        state_affinity=StateAffinity(modes=("work",), emotion_axes=("trust",)),
        budget_weight=0.75,
    )

    decision = explain_memory_selection(obj, mode="work", emotion_axes={"trust"}, budget_available=1.0)

    assert decision.selected is True
    assert decision.reason == "approved selective memory matched state"
    assert decision.matched_mode == "work"
    assert decision.matched_emotion_axes == ("trust",)
    assert decision.priority_class is PriorityClass.SELECTIVE
    assert decision.budget_weight == 0.75


def test_pending_memory_is_rejected_before_state_or_budget_checks():
    obj = MemoryObject(
        canonical_key="pending-pref",
        object_type=MemoryObjectType.PREFERENCE,
        content="Pending preference.",
        status=MemoryObjectStatus.PENDING,
        injection_policy=InjectionPolicy.SELECTIVE,
        state_affinity=StateAffinity(modes=("work",)),
    )

    decision = explain_memory_selection(obj, mode="work", emotion_axes=set(), budget_available=1.0)

    assert decision.selected is False
    assert decision.reason == "not eligible for injection"
    assert decision.rejected_reason == "status=pending"
    assert decision.priority_class is PriorityClass.REJECTED


def test_quarantined_memory_is_rejected_as_evidence_only():
    obj = MemoryObject(
        canonical_key="quarantined-identity",
        object_type=MemoryObjectType.IDENTITY,
        content="Conflicted identity claim.",
        status=MemoryObjectStatus.QUARANTINED,
        injection_policy=InjectionPolicy.EVIDENCE_ONLY,
    )

    decision = explain_memory_selection(obj, mode="work", emotion_axes=set(), budget_available=1.0)

    assert decision.selected is False
    assert decision.reason == "not eligible for injection"
    assert decision.rejected_reason == "status=quarantined"
    assert decision.priority_class is PriorityClass.EVIDENCE_ONLY


def test_retired_memory_is_rejected_even_with_budget():
    obj = MemoryObject(
        canonical_key="retired-pref",
        object_type=MemoryObjectType.RETIRED,
        content="Old preference.",
        status=MemoryObjectStatus.RETIRED,
        injection_policy=InjectionPolicy.NEVER,
    )

    decision = explain_memory_selection(obj, mode="work", emotion_axes=set(), budget_available=10.0)

    assert decision.selected is False
    assert decision.reason == "not eligible for injection"
    assert decision.rejected_reason == "status=retired"
    assert decision.priority_class is PriorityClass.REJECTED


def test_mode_mismatch_rejects_approved_selective_memory():
    obj = MemoryObject(
        canonical_key="daily-pref",
        object_type=MemoryObjectType.PREFERENCE,
        content="Daily mode preference.",
        status=MemoryObjectStatus.APPROVED,
        injection_policy=InjectionPolicy.SELECTIVE,
        state_affinity=StateAffinity(modes=("daily",)),
    )

    decision = explain_memory_selection(obj, mode="work", emotion_axes=set(), budget_available=1.0)

    assert decision.selected is False
    assert decision.reason == "state affinity mismatch"
    assert decision.rejected_reason == "mode_mismatch"
    assert decision.matched_mode is None
    assert decision.priority_class is PriorityClass.REJECTED


def test_budget_shortfall_rejects_selective_memory_after_state_match():
    obj = MemoryObject(
        canonical_key="large-project-memory",
        object_type=MemoryObjectType.PROJECT,
        content="Large project memory.",
        status=MemoryObjectStatus.APPROVED,
        injection_policy=InjectionPolicy.SELECTIVE,
        budget_weight=2.0,
    )

    decision = explain_memory_selection(obj, mode="work", emotion_axes=set(), budget_available=1.0)

    assert decision.selected is False
    assert decision.reason == "insufficient budget"
    assert decision.rejected_reason == "budget_weight_exceeds_available"
    assert decision.priority_class is PriorityClass.REJECTED


def test_evidence_only_approved_memory_is_not_selected_for_prompt_injection():
    obj = MemoryObject(
        canonical_key="tool-output",
        object_type=MemoryObjectType.TOOL_EVIDENCE,
        content="Terminal output capsule.",
        status=MemoryObjectStatus.APPROVED,
        injection_policy=InjectionPolicy.EVIDENCE_ONLY,
    )

    decision = explain_memory_selection(obj, mode="work", emotion_axes=set(), budget_available=1.0)

    assert decision.selected is False
    assert decision.reason == "evidence-only memory"
    assert decision.rejected_reason == "injection_policy=evidence_only"
    assert decision.priority_class is PriorityClass.EVIDENCE_ONLY
