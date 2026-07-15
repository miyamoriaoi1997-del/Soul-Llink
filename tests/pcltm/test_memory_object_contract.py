import pytest

from pcltm.memory_object import (
    InjectionPolicy,
    MemoryObject,
    MemoryObjectScope,
    MemoryObjectStatus,
    MemoryObjectType,
    StateAffinity,
)


def test_memory_object_round_trips_typed_contract():
    obj = MemoryObject(
        canonical_key="user-pref-response-style",
        object_type=MemoryObjectType.PREFERENCE,
        content="User prefers compact but evidenced closeouts.",
        scope=MemoryObjectScope.USER,
        status=MemoryObjectStatus.APPROVED,
        injection_policy=InjectionPolicy.SELECTIVE,
        source="telegram",
        confidence=0.9,
        stability=0.8,
        emotional_weight=0.2,
        budget_weight=1.5,
        state_affinity=StateAffinity(modes=("work",), emotion_axes=("trust",)),
        tags=("preference", "closeout"),
        conflict_keys=("legacy-response-style",),
        metadata={"record_id": "mem-1"},
    )

    restored = MemoryObject.from_dict(obj.to_dict())

    assert restored == obj
    assert restored.injectable is True
    assert restored.matches_state("work", {"trust"}) is True
    assert restored.matches_state("daily", {"trust"}) is False
    assert restored.matches_state("work", {"stress"}) is False


def test_identity_memory_must_be_pinned():
    with pytest.raises(ValueError, match="active identity memories must use pinned injection"):
        MemoryObject(
            canonical_key="identity-anchor",
            object_type=MemoryObjectType.IDENTITY,
            content="Assistant identity anchor.",
            injection_policy=InjectionPolicy.SELECTIVE,
        )


def test_retired_memory_must_never_inject():
    with pytest.raises(ValueError, match="retired memories must not be injectable"):
        MemoryObject(
            canonical_key="old-memory",
            object_type=MemoryObjectType.RETIRED,
            content="Superseded memory.",
            status=MemoryObjectStatus.RETIRED,
            injection_policy=InjectionPolicy.EVIDENCE_ONLY,
        )


def test_quarantined_memory_cannot_be_pinned():
    with pytest.raises(ValueError, match="quarantined memories cannot be pinned"):
        MemoryObject(
            canonical_key="conflicted-memory",
            object_type=MemoryObjectType.CONFLICT,
            content="Conflicting memory awaiting governance.",
            status=MemoryObjectStatus.QUARANTINED,
            injection_policy=InjectionPolicy.PINNED,
        )


def test_invalid_scores_are_rejected():
    with pytest.raises(ValueError, match="confidence must be between 0 and 1"):
        MemoryObject(
            canonical_key="bad-confidence",
            object_type=MemoryObjectType.EPISODIC,
            content="Invalid confidence.",
            confidence=1.2,
        )

    with pytest.raises(ValueError, match="budget_weight must be positive"):
        MemoryObject(
            canonical_key="bad-budget",
            object_type=MemoryObjectType.EPISODIC,
            content="Invalid budget.",
            budget_weight=0,
        )


def test_neutral_state_affinity_matches_any_state():
    obj = MemoryObject(
        canonical_key="project-fact",
        object_type=MemoryObjectType.PROJECT,
        content="Project fact with no mode affinity.",
        status=MemoryObjectStatus.APPROVED,
    )

    assert obj.matches_state("work", set()) is True
    assert obj.matches_state("daily", {"affection"}) is True
