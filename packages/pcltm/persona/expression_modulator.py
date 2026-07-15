"""Expression modulation for persona/emotion state.

The modulator turns persona anchors plus current emotion into expression hints.
It never mutates identity, address rules, facts, or tool discipline.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum
from typing import Mapping

from .emotional_state import EmotionalState, Intensity
from .persona_anchor import PersonaAnchor


class PersonaMode(StrEnum):
    """Supported expression modes."""

    WORK = "work"
    DAILY = "daily"
    SEX = "sex"


@dataclass(frozen=True, slots=True)
class ExpressionProfile:
    """Prompt-facing expression profile after applying mode and emotion."""

    mode: PersonaMode
    identity: str
    address_rule: str
    relationship_rule: str
    warmth: float
    directness: float
    emotional_visibility: float
    boundary_hardness: float
    intimacy_permission: str
    fact_layer_locked: bool = True
    tool_discipline_locked: bool = True
    identity_locked: bool = True
    address_locked: bool = True

    def as_dict(self) -> Mapping[str, object]:
        return {
            "mode": self.mode.value,
            "identity": self.identity,
            "address_rule": self.address_rule,
            "relationship_rule": self.relationship_rule,
            "warmth": self.warmth,
            "directness": self.directness,
            "emotional_visibility": self.emotional_visibility,
            "boundary_hardness": self.boundary_hardness,
            "intimacy_permission": self.intimacy_permission,
            "fact_layer_locked": self.fact_layer_locked,
            "tool_discipline_locked": self.tool_discipline_locked,
            "identity_locked": self.identity_locked,
            "address_locked": self.address_locked,
        }


class ExpressionModulator:
    """Apply mode-specific expression rules without touching fact authority."""

    def modulate(
        self,
        *,
        anchor: PersonaAnchor,
        state: EmotionalState,
        mode: PersonaMode | str,
    ) -> ExpressionProfile:
        selected_mode = PersonaMode(mode)
        emotion_visibility = _visibility_from_intensity(state.intensity)
        warmth = _clamp(0.25 + state.affection * 0.55 + state.trust * 0.2 - state.distance * 0.2)
        directness = _clamp(0.55 + state.boundary_hardness * 0.35 + state.anxiety * 0.1)
        boundary = state.boundary_hardness

        if selected_mode is PersonaMode.WORK:
            emotion_visibility *= 0.35
            warmth *= 0.65
            directness = _clamp(max(directness, 0.78))
            boundary = _clamp(max(boundary, 0.65))
            intimacy = "suppressed; care may appear as concise protection"
        elif selected_mode is PersonaMode.DAILY:
            emotion_visibility *= 0.85
            warmth = _clamp(warmth + 0.1)
            directness = _clamp(directness - 0.1)
            boundary = _clamp(boundary - 0.05)
            intimacy = "softened; ordinary closeness may show through text"
        else:
            emotion_visibility = _clamp(max(emotion_visibility, 0.45))
            warmth = _clamp(warmth + 0.2)
            directness = _clamp(directness - 0.2)
            boundary = _clamp(max(boundary, 0.7))
            intimacy = "explicit adult boundary required; facts and consent remain locked"

        return ExpressionProfile(
            mode=selected_mode,
            identity=anchor.identity,
            address_rule=anchor.address_rule,
            relationship_rule=anchor.relationship_rule,
            warmth=round(warmth, 4),
            directness=round(directness, 4),
            emotional_visibility=round(_clamp(emotion_visibility), 4),
            boundary_hardness=round(_clamp(boundary), 4),
            intimacy_permission=intimacy,
        )


def enforce_conflict_rules(profile: ExpressionProfile, facts: Mapping[str, object]) -> Mapping[str, object]:
    """Return facts unchanged while asserting expression locks remain active."""

    if not (
        profile.fact_layer_locked
        and profile.tool_discipline_locked
        and profile.identity_locked
        and profile.address_locked
    ):
        raise ValueError("expression profile violates persona conflict locks")
    return facts


def _visibility_from_intensity(intensity: Intensity) -> float:
    return {
        Intensity.CALM: 0.1,
        Intensity.MILD: 0.3,
        Intensity.MODERATE: 0.55,
        Intensity.STRONG: 0.8,
        Intensity.OVERFLOW: 1.0,
    }[intensity]


def _clamp(value: float, minimum: float = 0.0, maximum: float = 1.0) -> float:
    return max(minimum, min(maximum, value))
