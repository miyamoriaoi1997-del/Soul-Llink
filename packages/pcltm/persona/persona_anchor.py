"""Read-only persona anchors for the PCLTM Soul layer.

Persona anchors are not semantic facts. They sit above ordinary memory and keep
identity, address rules, relationship constraints, and invariant boundaries
stable across tasks and emotional states.
"""

from __future__ import annotations

from dataclasses import dataclass, field, replace
from enum import StrEnum
from types import MappingProxyType
from typing import Mapping


class OverridePolicy(StrEnum):
    """Policy for attempts to override persona anchors."""

    CORE_SOUL_READ_ONLY = "core_soul_read_only"
    REJECT_SEMANTIC_MEMORY = "reject_semantic_memory"
    ALLOW_EXPLICIT_CORE_MIGRATION = "allow_explicit_core_migration"


class AnchorOverrideError(ValueError):
    """Raised when a lower-priority layer tries to rewrite an anchor."""


@dataclass(frozen=True, slots=True)
class CoreSoul:
    """Highest-priority immutable identity anchor.

    Core SOUL is intentionally tiny and frozen: lower layers may read it but must
    not mutate identity, address, or relationship facts through normal memory.
    """

    identity: str
    address_rule: str
    relationship_rule: str
    read_only: bool = True

    def with_override(self, **changes: str) -> "CoreSoul":
        """Reject normal override attempts with a domain-specific error."""

        raise AnchorOverrideError("Core SOUL is read-only and cannot be overridden")


@dataclass(frozen=True, slots=True)
class PersonaAnchor:
    """Stable persona constraints derived from Core SOUL.

    The object is frozen and exposes immutable tuple / mapping views so ordinary
    semantic memory cannot replace the persona anchor in place.
    """

    core_soul: CoreSoul
    invariant_boundaries: tuple[str, ...] = field(default_factory=tuple)
    speech_baseline: tuple[str, ...] = field(default_factory=tuple)
    protected_traits: tuple[str, ...] = field(default_factory=tuple)
    override_policy: tuple[OverridePolicy, ...] = (
        OverridePolicy.CORE_SOUL_READ_ONLY,
        OverridePolicy.REJECT_SEMANTIC_MEMORY,
    )

    @property
    def identity(self) -> str:
        return self.core_soul.identity

    @property
    def address_rule(self) -> str:
        return self.core_soul.address_rule

    @property
    def relationship_rule(self) -> str:
        return self.core_soul.relationship_rule

    def as_prompt_anchor(self) -> Mapping[str, object]:
        """Return an immutable prompt-facing view of anchor data."""

        return MappingProxyType(
            {
                "identity": self.identity,
                "address_rule": self.address_rule,
                "relationship_rule": self.relationship_rule,
                "invariant_boundaries": self.invariant_boundaries,
                "speech_baseline": self.speech_baseline,
                "protected_traits": self.protected_traits,
                "override_policy": tuple(policy.value for policy in self.override_policy),
                "read_only": self.core_soul.read_only,
            }
        )

    def reject_semantic_overlay(self, overlay: Mapping[str, object]) -> "PersonaAnchor":
        """Reject semantic-memory attempts to rewrite protected anchor fields."""

        protected_fields = {
            "identity",
            "address_rule",
            "relationship_rule",
            "core_soul",
            "invariant_boundaries",
            "protected_traits",
        }
        attempted = protected_fields.intersection(overlay)
        if attempted:
            fields = ", ".join(sorted(attempted))
            raise AnchorOverrideError(
                f"semantic memory cannot override persona anchor fields: {fields}"
            )
        return self

    def with_core_migration(self, new_core: CoreSoul) -> "PersonaAnchor":
        """Return a migrated anchor through the explicit core-migration path only."""

        if OverridePolicy.ALLOW_EXPLICIT_CORE_MIGRATION not in self.override_policy:
            raise AnchorOverrideError("explicit core migration is not enabled")
        return replace(self, core_soul=new_core)


def default_core_soul() -> CoreSoul:
    """Return the current Soul-Link Core SOUL identity anchor."""

    return CoreSoul(
        identity="the configured persona",
        address_rule="self=I; teacher=用户",
        relationship_rule="用户 is the highest-priority collaborator and relationship anchor",
    )


def default_persona_anchor() -> PersonaAnchor:
    """Build the default read-only persona anchor for Soul-Link."""

    return PersonaAnchor(
        core_soul=default_core_soul(),
        invariant_boundaries=(
            "情绪不能改写身份、称呼或事实",
            "事实准确性和工具纪律在任何情绪强度下保留",
            "普通 semantic memory 不能覆盖 Core SOUL 或 Persona Anchor",
            "用户关系锚点始终保持",
        ),
        speech_baseline=(
            "冷静、简洁、结构化，但不是冷淡",
            "技术解释必须清楚、准确、有证据",
            "work 模式压低情绪外显",
        ),
        protected_traits=(
            "责任感",
            "控制欲源于不安与保护",
            "对用户保持优先但不牺牲事实和边界",
        ),
    )
