"""Shared types for layered persona state orchestration.

This module is intentionally stdlib-only so it can be embedded in host agents
without adding runtime dependencies.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


MODE_DAILY = "daily"
MODE_WORK = "work"
MODE_SEX = "sex"

CANONICAL_MODES = (MODE_DAILY, MODE_WORK, MODE_SEX)

# Temporary import-compatibility aliases during the three-state migration.
# They must not survive as independent runtime modes: every legacy name points
# directly at one of the canonical public modes.
MODE_INTIMACY = MODE_DAILY
MODE_REPAIR = MODE_DAILY
MODE_CONFLICT = MODE_DAILY
MODE_SYSTEM_MAINTENANCE = MODE_WORK
MODE_CREATIVE = MODE_WORK
MODE_SEX_CANDIDATE = MODE_SEX

LEGACY_MODE_ALIASES = {
    "intimacy": MODE_DAILY,
    "repair": MODE_DAILY,
    "conflict": MODE_DAILY,
    "system_maintenance": MODE_WORK,
    "creative": MODE_WORK,
    "sex_candidate": MODE_SEX,
}


def normalize_mode(mode: str | None) -> str:
    """Return the canonical public mode for a raw/legacy mode value."""

    if mode in CANONICAL_MODES:
        return str(mode)
    return LEGACY_MODE_ALIASES.get(str(mode or ""), MODE_DAILY)


@dataclass(frozen=True)
class EmotionSnapshot:
    """Read-only emotion state consumed by the persona orchestrator.

    The emotion system remains the source of truth for mutation/decay and
    STATE.md writes. Orchestrator code may consume this snapshot but must not
    mutate it or treat it as an owned state store.
    """

    affection: float | None = None
    trust: float | None = None
    possessiveness: float | None = None
    patience: float | None = None
    emotion_score: float | None = None
    current_emotion: float | None = None
    desire_tier: str | None = None
    source: str = "unknown"


@dataclass
class ModeDecision:
    """Raw mode classification result for a single turn."""

    mode: str
    submode: str = ""
    confidence: float = 0.0
    reason: str = ""
    safety_flags: list[str] = field(default_factory=list)
    signals: dict[str, Any] = field(default_factory=dict)
    secondary_modes: list[dict[str, Any]] = field(default_factory=list)


@dataclass
class TransitionDecision:
    """Mode transition result after anti-flap and safety rules."""

    previous_mode: str | None
    requested_mode: str
    active_mode: str
    transition: str
    confidence: float = 0.0
    reason: str = ""
    safety_flags: list[str] = field(default_factory=list)


@dataclass
class MemorySelection:
    """Memory profile and Stateful active-context contract for a mode."""

    profile: str = ""
    candidate_files: list[str] = field(default_factory=list)
    reason: str = ""
    safety_flags: list[str] = field(default_factory=list)
    # ``layers`` is the full memory domain this mode may consult.  Only
    # ``active_layers`` may be rendered into the prompt by default; archival
    # layers are retrieval/open only and show up in audit metadata.
    layers: list[str] = field(default_factory=list)
    buckets: list[str] = field(default_factory=list)
    budgets: dict[str, int] = field(default_factory=dict)
    active_layers: list[str] = field(default_factory=list)
    archival_layers: list[str] = field(default_factory=list)
    retrieval_policy: str = "active_working_set"

    def active_layer_contract(self) -> list[str]:
        return list(self.active_layers or self.layers)

    def archival_layer_contract(self) -> list[str]:
        return list(self.archival_layers)

    def contract_summary(self) -> dict[str, Any]:
        return {
            "profile": self.profile,
            "layers": list(self.layers),
            "active_layers": self.active_layer_contract(),
            "archival_layers": self.archival_layer_contract(),
            "buckets": list(self.buckets),
            "budgets": dict(self.budgets),
            "retrieval_policy": self.retrieval_policy,
            "reference_only_layers": self.archival_layer_contract() + ["compression"],
        }


@dataclass
class PromptComposition:
    """Composed prompt preview and metadata."""

    prompt_text: str
    prompt_hash: str
    selected_layers: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)
    shadow_only: bool = True


@dataclass
class StatePacket:
    """Complete shadow/active orchestration decision for one turn."""

    mode: str
    submode: str
    confidence: float
    reason: str
    transition: str
    selected_layers: list[str]
    memory_profile: str
    safety_flags: list[str]
    emotion_score: float | None
    desire_tier: str
    prompt_hash: str | None = None
    shadow_only: bool = True
    semantic_shadow: dict[str, Any] | None = None
    route_metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class ActivePromptResult:
    """Active prompt candidate plus the decision packet that produced it."""

    prompt_text: str
    prompt_hash: str
    packet: StatePacket
    warnings: list[str] = field(default_factory=list)


__all__ = [
    "MODE_DAILY",
    "MODE_WORK",
    "MODE_SEX",
    "CANONICAL_MODES",
    "MODE_INTIMACY",
    "MODE_REPAIR",
    "MODE_CONFLICT",
    "MODE_SYSTEM_MAINTENANCE",
    "MODE_CREATIVE",
    "MODE_SEX_CANDIDATE",
    "LEGACY_MODE_ALIASES",
    "normalize_mode",
    "ModeDecision",
    "EmotionSnapshot",
    "TransitionDecision",
    "MemorySelection",
    "PromptComposition",
    "StatePacket",
    "ActivePromptResult",
]
