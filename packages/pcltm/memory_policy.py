"""Deterministic, I/O-free policy kernel for memory authority decisions."""

from __future__ import annotations

from dataclasses import dataclass

from .memory_contracts import (
    AdmissionDecision,
    AuthoritySnapshot,
    LifecycleState,
    LineageKind,
    MemoryAccessRequest,
    AccessSurface,
    PersonaMode,
    MemoryTransitionCommand,
    MemoryWriteCommand,
    Sensitivity,
)

POLICY_VERSION = "memory-policy-v1"

REASON_WRITE_ALLOWED = "write_allowed"
REASON_EMPTY_LINEAGE = "empty_lineage"
REASON_LEGACY_LINEAGE = "legacy_lineage_forbidden"
REASON_SECRET_WRITE = "secret_write_forbidden"
REASON_SENSITIVITY_DOWNGRADE = "sensitivity_downgrade"
REASON_INACTIVE = "lifecycle_inactive"
REASON_SECRET_ACCESS = "secret_access_denied"
REASON_RESTRICTED_ACCESS = "restricted_access_denied"
REASON_MODE_DENIED = "mode_denied"
REASON_INVALID_TRANSITION = "invalid_transition"
REASON_ACCESS_ALLOWED = "access_allowed"
REASON_TRANSITION_ALLOWED = "transition_allowed"
REASON_SENSITIVITY_CEILING_EXCEEDED = "sensitivity_ceiling_exceeded"
REASON_SOURCE_SNAPSHOT_MISSING = "source_snapshot_missing"
REASON_SOURCE_SNAPSHOT_MISMATCH = "source_snapshot_mismatch"
REASON_INJECTION_POLICY_DENIED = "injection_policy_denied"

_SENSITIVITY_RANK = {
    Sensitivity.NORMAL: 0,
    Sensitivity.PRIVATE: 1,
    Sensitivity.RESTRICTED: 2,
    Sensitivity.SECRET: 3,
}

_ALLOWED_TRANSITIONS = {
    LifecycleState.PENDING_REVIEW: {LifecycleState.ACTIVE, LifecycleState.REJECTED, LifecycleState.QUARANTINED},
    LifecycleState.ACTIVE: {LifecycleState.SUPERSEDED, LifecycleState.RETIRED, LifecycleState.EXPIRED},
    LifecycleState.SUPERSEDED: set(),
    LifecycleState.RETIRED: set(),
    LifecycleState.EXPIRED: set(),
    LifecycleState.REJECTED: set(),
    LifecycleState.QUARANTINED: set(),
}


@dataclass(frozen=True, slots=True)
class SensitivityDecision:
    effective: Sensitivity
    allowed: bool
    reason_code: str
    policy_version: str = POLICY_VERSION


@dataclass(frozen=True, slots=True)
class TransitionDecision:
    allowed: bool
    previous_state: LifecycleState
    new_state: LifecycleState
    reason_code: str
    policy_version: str = POLICY_VERSION


def _decision(allowed: bool, action: str, reason_code: str, *, snapshot: AuthoritySnapshot | None = None,
              sensitivity: Sensitivity | None = None, modes: tuple[PersonaMode, ...] = ()) -> AdmissionDecision:
    return AdmissionDecision(allowed, action, reason_code, POLICY_VERSION, snapshot, sensitivity, modes)


def inherit_sensitivity(source_snapshots: tuple[AuthoritySnapshot, ...], requested: Sensitivity) -> SensitivityDecision:
    if type(requested) is not Sensitivity or type(source_snapshots) is not tuple:
        raise TypeError("invalid sensitivity input")
    source_effective = max((snapshot.sensitivity for snapshot in source_snapshots), default=Sensitivity.NORMAL,
                           key=lambda value: _SENSITIVITY_RANK[value])
    effective = max(source_effective, requested, key=lambda value: _SENSITIVITY_RANK[value])
    if _SENSITIVITY_RANK[requested] < _SENSITIVITY_RANK[source_effective]:
        return SensitivityDecision(effective, False, REASON_SENSITIVITY_DOWNGRADE)
    return SensitivityDecision(effective, True, REASON_WRITE_ALLOWED)


def admit_write(command: MemoryWriteCommand, source_snapshots: tuple[AuthoritySnapshot, ...] = ()) -> AdmissionDecision:
    if type(command) is not MemoryWriteCommand or type(source_snapshots) is not tuple:
        raise TypeError("invalid write input")
    if command.sensitivity is Sensitivity.SECRET:
        return _decision(False, "reject", REASON_SECRET_WRITE)
    if command.lineage_kind is LineageKind.LEGACY_GOVERNED:
        return _decision(False, "reject", REASON_LEGACY_LINEAGE)
    needs_source = command.lineage_kind in {
        LineageKind.EVENT_DERIVED, LineageKind.EXPLICIT_USER_ASSERTION,
    }
    if needs_source and not command.source_refs:
        return _decision(False, "reject", REASON_SOURCE_SNAPSHOT_MISSING)
    if needs_source and not source_snapshots:
        return _decision(False, "reject", REASON_SOURCE_SNAPSHOT_MISSING)
    if needs_source:
        ref_keys = {
            (ref.authority_kind, ref.object_id, ref.object_version, ref.payload_sha256)
            for ref in command.source_refs
        }
        snapshot_keys = {
            (snapshot.authority_kind, snapshot.object_id, snapshot.object_version, snapshot.payload_sha256)
            for snapshot in source_snapshots
        }
        if (len(ref_keys) != len(command.source_refs)
                or len(snapshot_keys) != len(source_snapshots)
                or ref_keys != snapshot_keys):
            return _decision(False, "reject", REASON_SOURCE_SNAPSHOT_MISMATCH)
    inherited = inherit_sensitivity(source_snapshots, command.sensitivity)
    if not inherited.allowed:
        return _decision(False, "reject", inherited.reason_code, sensitivity=inherited.effective)
    if command.requested_lifecycle is LifecycleState.ACTIVE and not command.source_refs:
        return _decision(False, "reject", REASON_EMPTY_LINEAGE, sensitivity=inherited.effective)
    return _decision(True, "allow", REASON_WRITE_ALLOWED, sensitivity=inherited.effective, modes=command.mode_scope)


def admit_access(snapshot: AuthoritySnapshot, request: MemoryAccessRequest) -> AdmissionDecision:
    if type(snapshot) is not AuthoritySnapshot or type(request) is not MemoryAccessRequest:
        raise TypeError("invalid access input")
    if snapshot.lifecycle_state is not LifecycleState.ACTIVE:
        return _decision(False, "reject", REASON_INACTIVE, snapshot=snapshot, sensitivity=snapshot.sensitivity)
    if _SENSITIVITY_RANK[snapshot.sensitivity] > _SENSITIVITY_RANK[request.sensitivity_ceiling]:
        return _decision(False, "reject", REASON_SENSITIVITY_CEILING_EXCEEDED, snapshot=snapshot, sensitivity=snapshot.sensitivity)
    if snapshot.sensitivity is Sensitivity.SECRET:
        return _decision(False, "reject", REASON_SECRET_ACCESS, snapshot=snapshot, sensitivity=snapshot.sensitivity)
    if snapshot.sensitivity is Sensitivity.RESTRICTED and request.surface is not AccessSurface.AUDIT:
        return _decision(False, "reject", REASON_RESTRICTED_ACCESS, snapshot=snapshot, sensitivity=snapshot.sensitivity)
    if snapshot.sensitivity is Sensitivity.PRIVATE and request.persona_mode not in {
        PersonaMode.DAILY, PersonaMode.SEX,
    }:
        return _decision(False, "reject", REASON_MODE_DENIED, snapshot=snapshot, sensitivity=snapshot.sensitivity)
    if request.persona_mode not in snapshot.mode_scope:
        return _decision(False, "reject", REASON_MODE_DENIED, snapshot=snapshot, sensitivity=snapshot.sensitivity)
    if request.surface is AccessSurface.INJECT and snapshot.injection_policy != "allow":
        return _decision(False, "reject", REASON_INJECTION_POLICY_DENIED,
                         snapshot=snapshot, sensitivity=snapshot.sensitivity)
    return _decision(True, "allow", REASON_ACCESS_ALLOWED, snapshot=snapshot, sensitivity=snapshot.sensitivity,
                     modes=(request.persona_mode,))


def admit_injection(snapshot: AuthoritySnapshot, request: MemoryAccessRequest, candidate: object = None) -> AdmissionDecision:
    del candidate
    if request.surface is not AccessSurface.INJECT:
        return _decision(False, "reject", REASON_MODE_DENIED, snapshot=snapshot, sensitivity=snapshot.sensitivity)
    return admit_access(snapshot, request)


def resolve_transition(current_state: LifecycleState, command: MemoryTransitionCommand) -> TransitionDecision:
    if type(current_state) is not LifecycleState or type(command) is not MemoryTransitionCommand:
        raise TypeError("invalid transition input")
    action_targets = {"activate": LifecycleState.ACTIVE, "supersede": LifecycleState.SUPERSEDED,
                      "retire": LifecycleState.RETIRED, "expire": LifecycleState.EXPIRED,
                      "reject": LifecycleState.REJECTED, "quarantine": LifecycleState.QUARANTINED}
    allowed = action_targets.get(command.action) is command.target_state and command.target_state in _ALLOWED_TRANSITIONS[current_state]
    return TransitionDecision(allowed, current_state, command.target_state,
                              REASON_TRANSITION_ALLOWED if allowed else REASON_INVALID_TRANSITION)
