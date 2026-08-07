"""Frozen, side-effect-free contracts shared by memory policy consumers."""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Any


class _ValueEnum(str, Enum):
    def __str__(self) -> str:
        return self.value


class Sensitivity(_ValueEnum):
    NORMAL = "normal"
    PRIVATE = "private"
    RESTRICTED = "restricted"
    SECRET = "secret"


class LifecycleState(_ValueEnum):
    PENDING_REVIEW = "pending_review"
    ACTIVE = "active"
    SUPERSEDED = "superseded"
    RETIRED = "retired"
    EXPIRED = "expired"
    REJECTED = "rejected"
    QUARANTINED = "quarantined"


class LineageKind(_ValueEnum):
    EVENT_DERIVED = "event_derived"
    EXPLICIT_USER_ASSERTION = "explicit_user_assertion"
    SYSTEM_GOVERNED_INVARIANT = "system_governed_invariant"
    LEGACY_GOVERNED = "legacy_governed"
    TRANSIENT_TASK_STATE = "transient_task_state"


class AccessSurface(_ValueEnum):
    SEARCH = "search"
    OPEN = "open"
    EXACT = "exact"
    INJECT = "inject"
    AUDIT = "audit"


class PersonaMode(_ValueEnum):
    DAILY = "daily"
    WORK = "work"
    SEX = "sex"
    CRON = "cron"
    DEFAULT = "default"


# Compatibility name for the pre-split access-only contract.  It is an alias
# of AccessSurface, never a union of access and persona values.
MemoryMode = AccessSurface


@dataclass(frozen=True, slots=True)
class AuthorityRef:
    authority_kind: str
    object_id: str
    object_version: int
    payload_sha256: str

    def __post_init__(self) -> None:
        if type(self.object_version) is not int or self.object_version <= 0:
            raise TypeError("object_version must be a positive int")
        if not self.authority_kind or not self.object_id:
            raise ValueError("authority reference identifiers are required")
        if (type(self.payload_sha256) is not str or len(self.payload_sha256) != 64
                or any(char not in "0123456789abcdef" for char in self.payload_sha256)):
            raise ValueError("payload_sha256 must be 64 lowercase hexadecimal characters")


@dataclass(frozen=True, slots=True)
class AuthoritySnapshot:
    authority_kind: str
    object_id: str
    object_version: int
    payload_sha256: str
    governance_id: int
    governance_state: LifecycleState
    sensitivity: Sensitivity
    lifecycle_state: LifecycleState
    source_refs: tuple[AuthorityRef, ...]
    projection_generation: int | None
    mode_scope: tuple[PersonaMode, ...] = ()
    injection_policy: str = "allow"

    def __post_init__(self) -> None:
        if type(self.object_version) is not int or self.object_version <= 0:
            raise TypeError("object_version must be a positive int")
        if type(self.governance_id) is not int or self.governance_id <= 0:
            raise TypeError("governance_id must be a positive int")
        if type(self.projection_generation) not in (int, type(None)):
            raise TypeError("projection_generation must be an int or None")
        if self.projection_generation is not None and self.projection_generation <= 0:
            raise ValueError("projection_generation must be positive")
        if type(self.governance_state) is not LifecycleState or type(self.lifecycle_state) is not LifecycleState:
            raise TypeError("lifecycle states must be LifecycleState members")
        if type(self.sensitivity) is not Sensitivity:
            raise TypeError("sensitivity must be a Sensitivity member")
        if type(self.source_refs) is not tuple or not all(type(ref) is AuthorityRef for ref in self.source_refs):
            raise TypeError("source_refs must be a tuple of AuthorityRef")
        if type(self.mode_scope) is not tuple or not all(type(mode) is PersonaMode for mode in self.mode_scope):
            raise TypeError("mode_scope must be a tuple of PersonaMode")
        if type(self.injection_policy) is not str or not self.injection_policy.strip():
            raise ValueError("injection_policy is required")
        if self.governance_state is not self.lifecycle_state:
            raise ValueError("governance_state and lifecycle_state must agree")
        AuthorityRef(self.authority_kind, self.object_id, self.object_version, self.payload_sha256)


@dataclass(frozen=True, slots=True)
class AdmissionDecision:
    allowed: bool
    action: str
    reason_code: str
    policy_version: str
    snapshot: AuthoritySnapshot | None = None
    effective_sensitivity: Sensitivity | None = None
    injectable_modes: tuple[PersonaMode, ...] = ()

    def __post_init__(self) -> None:
        if type(self.allowed) is not bool:
            raise TypeError("allowed must be bool")
        if not self.action or not self.reason_code or not self.policy_version:
            raise ValueError("decision identifiers are required")
        if type(self.snapshot) not in (AuthoritySnapshot, type(None)):
            raise TypeError("snapshot must be AuthoritySnapshot or None")
        if type(self.effective_sensitivity) not in (Sensitivity, type(None)):
            raise TypeError("effective_sensitivity must be Sensitivity or None")
        if type(self.injectable_modes) is not tuple or not all(type(mode) is PersonaMode for mode in self.injectable_modes):
            raise TypeError("injectable_modes must be a tuple of PersonaMode")


@dataclass(frozen=True, slots=True)
class MemoryWriteReceipt:
    success: bool
    status: str
    claim_id: int | None
    claim_version: int | None
    governance_id: int | None
    persisted: bool
    projection_status: str
    recall_ready: bool
    reason_code: str

    def __post_init__(self) -> None:
        if type(self.success) is not bool or type(self.persisted) is not bool or type(self.recall_ready) is not bool:
            raise TypeError("receipt boolean fields must be bool")
        for name in ("claim_id", "claim_version", "governance_id"):
            value = getattr(self, name)
            if value is not None and (type(value) is not int or value <= 0):
                raise TypeError(f"{name} must be a positive int or None")
        if not self.status or not self.projection_status or not self.reason_code:
            raise ValueError("receipt status and reason are required")


@dataclass(frozen=True, slots=True)
class MemoryWriteCommand:
    lineage_kind: LineageKind
    sensitivity: Sensitivity
    mode_scope: tuple[PersonaMode, ...]
    source_refs: tuple[AuthorityRef, ...]
    requested_lifecycle: LifecycleState = LifecycleState.PENDING_REVIEW

    def __post_init__(self) -> None:
        if type(self.lineage_kind) is not LineageKind or type(self.sensitivity) is not Sensitivity:
            raise TypeError("command enum fields are invalid")
        if type(self.mode_scope) is not tuple or not all(type(mode) is PersonaMode for mode in self.mode_scope):
            raise TypeError("mode_scope must be a tuple of PersonaMode")
        if type(self.source_refs) is not tuple or not all(type(ref) is AuthorityRef for ref in self.source_refs):
            raise TypeError("source_refs must be a tuple of AuthorityRef")
        if type(self.requested_lifecycle) is not LifecycleState:
            raise TypeError("requested_lifecycle must be LifecycleState")


@dataclass(frozen=True, slots=True)
class MemoryAccessRequest:
    surface: AccessSurface
    persona_mode: PersonaMode = PersonaMode.DEFAULT
    sensitivity_ceiling: Sensitivity = Sensitivity.RESTRICTED

    def __post_init__(self) -> None:
        if type(self.surface) is not AccessSurface or type(self.persona_mode) is not PersonaMode or type(self.sensitivity_ceiling) is not Sensitivity:
            raise TypeError("access request enum fields are invalid")

    @property
    def mode(self) -> AccessSurface:
        return self.surface


@dataclass(frozen=True, slots=True)
class MemoryTransitionCommand:
    action: str
    target_state: LifecycleState

    def __post_init__(self) -> None:
        if not self.action or type(self.target_state) is not LifecycleState:
            raise TypeError("transition command is invalid")
