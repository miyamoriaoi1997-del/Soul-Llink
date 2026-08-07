from __future__ import annotations

import importlib
from pathlib import Path

import pytest


def _contracts():
    module_path = Path(__file__).parents[2] / "packages" / "pcltm" / "memory_contracts.py"
    assert module_path.is_file(), "memory contracts are not implemented"
    return importlib.import_module("pcltm.memory_contracts")


def _policy():
    module_path = Path(__file__).parents[2] / "packages" / "pcltm" / "memory_policy.py"
    assert module_path.is_file(), "memory policy is not implemented"
    return importlib.import_module("pcltm.memory_policy")


def _snapshot(sensitivity, lifecycle):
    c = _contracts()
    ref = c.AuthorityRef("event", "event-1", 1, "a" * 64)
    return c.AuthoritySnapshot("memory_claim", "claim-1", 1, "b" * 64, 1, lifecycle,
                               sensitivity, lifecycle, (ref,), None)


def test_memory_contracts_are_frozen_and_validate_nested_authority_refs() -> None:
    contracts = _contracts()
    ref = contracts.AuthorityRef("event", "event-7", 1, "a" * 64)
    snapshot = contracts.AuthoritySnapshot(
        "memory_claim", "claim-1", 1, "b" * 64, 3,
        contracts.LifecycleState.ACTIVE, contracts.Sensitivity.NORMAL,
        contracts.LifecycleState.ACTIVE, (ref,), None,
    )
    assert snapshot.source_refs == (ref,)
    with pytest.raises((AttributeError, TypeError)):
        snapshot.object_id = "changed"
    with pytest.raises(ValueError):
        contracts.AuthorityRef("event", "", 1, "a" * 64)
    with pytest.raises(TypeError):
        contracts.AuthorityRef("event", "event-7", True, "a" * 64)


def test_sensitivity_inheritance_and_write_lineage_are_fail_closed() -> None:
    contracts = _contracts()
    policy = _policy()
    private = _snapshot(contracts.Sensitivity.PRIVATE, contracts.LifecycleState.ACTIVE)
    inherited = policy.inherit_sensitivity((private,), contracts.Sensitivity.NORMAL)
    assert inherited.effective == contracts.Sensitivity.PRIVATE
    assert inherited.allowed is False
    empty = contracts.MemoryWriteCommand(
        contracts.LineageKind.EVENT_DERIVED, contracts.Sensitivity.NORMAL, (), (),
    )
    assert policy.admit_write(empty).allowed is False


def test_access_rejects_inactive_secret_and_restricted_normal_modes() -> None:
    c = _contracts()
    policy = _policy()
    for snapshot, request, reason in (
        (_snapshot(c.Sensitivity.NORMAL, c.LifecycleState.RETIRED),
         c.MemoryAccessRequest(c.MemoryMode.SEARCH), policy.REASON_INACTIVE),
        (_snapshot(c.Sensitivity.SECRET, c.LifecycleState.ACTIVE),
         c.MemoryAccessRequest(c.MemoryMode.OPEN, sensitivity_ceiling=c.Sensitivity.SECRET),
         policy.REASON_SECRET_ACCESS),
        (_snapshot(c.Sensitivity.RESTRICTED, c.LifecycleState.ACTIVE),
         c.MemoryAccessRequest(c.MemoryMode.SEARCH), policy.REASON_RESTRICTED_ACCESS),
    ):
        decision = policy.admit_access(snapshot, request)
        assert decision.allowed is False
        assert decision.reason_code == reason


def test_legal_transition_is_deterministic_and_invalid_transition_is_rejected() -> None:
    c = _contracts()
    policy = _policy()
    activate = c.MemoryTransitionCommand("activate", c.LifecycleState.ACTIVE)
    first = policy.resolve_transition(c.LifecycleState.PENDING_REVIEW, activate)
    second = policy.resolve_transition(c.LifecycleState.PENDING_REVIEW, activate)
    assert first == second
    assert first.allowed is True
    invalid = policy.resolve_transition(c.LifecycleState.RETIRED, activate)
    assert invalid.allowed is False
    assert invalid.reason_code == policy.REASON_INVALID_TRANSITION


def test_requested_sensitivity_is_retained_when_it_is_stricter_than_sources() -> None:
    c = _contracts()
    policy = _policy()
    normal = _snapshot(c.Sensitivity.NORMAL, c.LifecycleState.ACTIVE)
    decision = policy.inherit_sensitivity((normal,), c.Sensitivity.RESTRICTED)
    assert decision.effective is c.Sensitivity.RESTRICTED
    assert decision.allowed is True


def test_access_enforces_ceiling_before_surface_rules() -> None:
    c = _contracts()
    policy = _policy()
    restricted = _snapshot(c.Sensitivity.RESTRICTED, c.LifecycleState.ACTIVE)
    request = c.MemoryAccessRequest(c.AccessSurface.OPEN, sensitivity_ceiling=c.Sensitivity.NORMAL)
    decision = policy.admit_access(restricted, request)
    assert decision.allowed is False
    assert decision.reason_code == policy.REASON_SENSITIVITY_CEILING_EXCEEDED


def test_only_auditable_restricted_access_is_allowed_and_open_is_not_audit() -> None:
    c = _contracts()
    policy = _policy()
    restricted = c.AuthoritySnapshot(
        "memory_claim", "claim-1", 1, "b" * 64, 1,
        c.LifecycleState.ACTIVE, c.Sensitivity.RESTRICTED, c.LifecycleState.ACTIVE,
        (), None, (c.PersonaMode.DEFAULT,), "allow",
    )
    for surface in (c.AccessSurface.SEARCH, c.AccessSurface.OPEN, c.AccessSurface.EXACT, c.AccessSurface.INJECT):
        decision = policy.admit_access(
            restricted, c.MemoryAccessRequest(surface, sensitivity_ceiling=c.Sensitivity.RESTRICTED)
        )
        assert decision.allowed is False
        assert decision.reason_code == policy.REASON_RESTRICTED_ACCESS
    audit = policy.admit_access(
        restricted, c.MemoryAccessRequest(c.AccessSurface.AUDIT, sensitivity_ceiling=c.Sensitivity.RESTRICTED)
    )
    assert audit.allowed is True
    assert audit.reason_code == policy.REASON_ACCESS_ALLOWED


def test_access_and_write_modes_are_strictly_separate_and_snapshot_bound() -> None:
    c = _contracts()
    policy = _policy()
    assert c.MemoryWriteCommand(
        c.LineageKind.SYSTEM_GOVERNED_INVARIANT, c.Sensitivity.NORMAL,
        (c.PersonaMode.DAILY,), (),
    ).mode_scope == (c.PersonaMode.DAILY,)
    snapshot = c.AuthoritySnapshot(
        "memory_claim", "claim-1", 1, "b" * 64, 1,
        c.LifecycleState.ACTIVE, c.Sensitivity.NORMAL, c.LifecycleState.ACTIVE,
        (), None, (c.PersonaMode.DAILY,), "allow",
    )
    allowed = policy.admit_access(
        snapshot, c.MemoryAccessRequest(c.AccessSurface.INJECT, c.PersonaMode.DAILY)
    )
    assert allowed.allowed is True
    denied = policy.admit_access(
        snapshot, c.MemoryAccessRequest(c.AccessSurface.INJECT, c.PersonaMode.WORK)
    )
    assert denied.allowed is False
    assert denied.reason_code == policy.REASON_MODE_DENIED


def test_write_requires_bound_source_snapshots() -> None:
    c = _contracts()
    policy = _policy()
    ref = c.AuthorityRef("event", "event-1", 1, "a" * 64)
    command = c.MemoryWriteCommand(c.LineageKind.EVENT_DERIVED, c.Sensitivity.NORMAL, (), (ref,))
    missing = policy.admit_write(command, ())
    assert missing.allowed is False
    assert missing.reason_code == policy.REASON_SOURCE_SNAPSHOT_MISSING
    other = _snapshot(c.Sensitivity.NORMAL, c.LifecycleState.ACTIVE)
    mismatch = policy.admit_write(command, (other,))
    assert mismatch.allowed is False
    assert mismatch.reason_code == policy.REASON_SOURCE_SNAPSHOT_MISMATCH


def test_transition_action_must_match_state_transition() -> None:
    c = _contracts()
    policy = _policy()
    command = c.MemoryTransitionCommand("retire", c.LifecycleState.ACTIVE)
    decision = policy.resolve_transition(c.LifecycleState.PENDING_REVIEW, command)
    assert decision.allowed is False
    assert decision.reason_code == policy.REASON_INVALID_TRANSITION


def test_normal_access_obeys_snapshot_mode_scope_without_private_mode_ban() -> None:
    c = _contracts()
    policy = _policy()
    snapshot = c.AuthoritySnapshot(
        "memory_claim", "claim-1", 1, "b" * 64, 1,
        c.LifecycleState.ACTIVE, c.Sensitivity.NORMAL, c.LifecycleState.ACTIVE,
        (), None, (c.PersonaMode.WORK, c.PersonaMode.CRON, c.PersonaMode.DEFAULT), "allow",
    )
    for mode in (c.PersonaMode.WORK, c.PersonaMode.CRON, c.PersonaMode.DEFAULT):
        decision = policy.admit_access(snapshot, c.MemoryAccessRequest(c.AccessSurface.OPEN, mode))
        assert decision.allowed is True
        assert decision.reason_code == policy.REASON_ACCESS_ALLOWED
    denied = policy.admit_access(
        snapshot, c.MemoryAccessRequest(c.AccessSurface.OPEN, c.PersonaMode.DAILY)
    )
    assert denied.allowed is False
    assert denied.reason_code == policy.REASON_MODE_DENIED


def test_private_access_is_limited_to_scoped_daily_or_sex_modes() -> None:
    c = _contracts()
    policy = _policy()
    snapshot = c.AuthoritySnapshot(
        "memory_claim", "claim-1", 1, "b" * 64, 1,
        c.LifecycleState.ACTIVE, c.Sensitivity.PRIVATE, c.LifecycleState.ACTIVE,
        (), None, tuple(c.PersonaMode), "allow",
    )
    for mode in (c.PersonaMode.DAILY, c.PersonaMode.SEX):
        assert policy.admit_access(
            snapshot, c.MemoryAccessRequest(c.AccessSurface.OPEN, mode)
        ).allowed is True
    for mode in (c.PersonaMode.WORK, c.PersonaMode.CRON, c.PersonaMode.DEFAULT):
        decision = policy.admit_access(snapshot, c.MemoryAccessRequest(c.AccessSurface.OPEN, mode))
        assert decision.allowed is False
        assert decision.reason_code == policy.REASON_MODE_DENIED


def test_injection_policy_is_enforced_by_the_pure_access_policy() -> None:
    c = _contracts()
    policy = _policy()
    snapshot = c.AuthoritySnapshot(
        "memory_claim", "claim-1", 1, "b" * 64, 1,
        c.LifecycleState.ACTIVE, c.Sensitivity.NORMAL, c.LifecycleState.ACTIVE,
        (), None, (c.PersonaMode.DAILY,), "deny",
    )
    decision = policy.admit_access(
        snapshot, c.MemoryAccessRequest(c.AccessSurface.INJECT, c.PersonaMode.DAILY)
    )
    assert decision.allowed is False
    assert decision.reason_code == policy.REASON_INJECTION_POLICY_DENIED


def test_source_snapshot_binding_is_order_independent() -> None:
    c = _contracts()
    policy = _policy()
    ref1 = c.AuthorityRef("event", "event-1", 1, "a" * 64)
    ref2 = c.AuthorityRef("event", "event-2", 1, "b" * 64)
    snapshot1 = c.AuthoritySnapshot(
        "event", "event-1", 1, "a" * 64, 1,
        c.LifecycleState.ACTIVE, c.Sensitivity.NORMAL, c.LifecycleState.ACTIVE,
        (), None,
    )
    snapshot2 = c.AuthoritySnapshot(
        "event", "event-2", 1, "b" * 64, 1,
        c.LifecycleState.ACTIVE, c.Sensitivity.NORMAL, c.LifecycleState.ACTIVE,
        (), None,
    )
    command = c.MemoryWriteCommand(
        c.LineageKind.EVENT_DERIVED, c.Sensitivity.NORMAL, (), (ref1, ref2)
    )
    decision = policy.admit_write(command, (snapshot2, snapshot1))
    assert decision.allowed is True
    assert decision.reason_code == policy.REASON_WRITE_ALLOWED


def test_access_ceiling_precedes_secret_and_surface_rules() -> None:
    c = _contracts()
    policy = _policy()
    secret = _snapshot(c.Sensitivity.SECRET, c.LifecycleState.ACTIVE)
    below_secret = policy.admit_access(
        secret,
        c.MemoryAccessRequest(
            c.AccessSurface.AUDIT,
            c.PersonaMode.DAILY,
            c.Sensitivity.RESTRICTED,
        ),
    )
    assert below_secret.allowed is False
    assert below_secret.reason_code == policy.REASON_SENSITIVITY_CEILING_EXCEEDED

    at_secret = policy.admit_access(
        secret,
        c.MemoryAccessRequest(
            c.AccessSurface.AUDIT,
            c.PersonaMode.DAILY,
            c.Sensitivity.SECRET,
        ),
    )
    assert at_secret.allowed is False
    assert at_secret.reason_code == policy.REASON_SECRET_ACCESS


def test_audit_obeys_private_mode_boundary_and_snapshot_mode_scope() -> None:
    c = _contracts()
    policy = _policy()
    private = c.AuthoritySnapshot(
        "memory_claim", "private-claim", 1, "b" * 64, 1,
        c.LifecycleState.ACTIVE, c.Sensitivity.PRIVATE, c.LifecycleState.ACTIVE,
        (), None, tuple(c.PersonaMode), "allow",
    )
    denied_private = policy.admit_access(
        private,
        c.MemoryAccessRequest(c.AccessSurface.AUDIT, c.PersonaMode.WORK),
    )
    assert denied_private.allowed is False
    assert denied_private.reason_code == policy.REASON_MODE_DENIED

    restricted = c.AuthoritySnapshot(
        "memory_claim", "restricted-claim", 1, "c" * 64, 1,
        c.LifecycleState.ACTIVE, c.Sensitivity.RESTRICTED, c.LifecycleState.ACTIVE,
        (), None, (c.PersonaMode.DAILY,), "allow",
    )
    denied_scope = policy.admit_access(
        restricted,
        c.MemoryAccessRequest(
            c.AccessSurface.AUDIT,
            c.PersonaMode.WORK,
            c.Sensitivity.RESTRICTED,
        ),
    )
    assert denied_scope.allowed is False
    assert denied_scope.reason_code == policy.REASON_MODE_DENIED

    allowed_scope = policy.admit_access(
        restricted,
        c.MemoryAccessRequest(
            c.AccessSurface.AUDIT,
            c.PersonaMode.DAILY,
            c.Sensitivity.RESTRICTED,
        ),
    )
    assert allowed_scope.allowed is True
    assert allowed_scope.reason_code == policy.REASON_ACCESS_ALLOWED
