"""Decay rules for PCLTM emotional state."""

from __future__ import annotations

from dataclasses import replace
from datetime import UTC, datetime
import math

from .emotional_state import EmotionalState, Intensity, Valence


class EmotionalDecay:
    """Apply bounded decay without instant reset or permanent pollution."""

    def apply(self, state: EmotionalState, *, now: datetime | None = None) -> EmotionalState:
        """Return a decayed state at ``now``.

        Numeric dimensions move toward neutral gradually. Residues disappear after
        their own decay window. Identity and facts are not represented here and
        therefore cannot be changed by decay.
        """

        current_time = _aware(now or datetime.now(UTC))
        elapsed = max(0.0, (current_time - state.updated_at).total_seconds())
        half_life_seconds = state.decay_policy.half_life.total_seconds()
        retention = math.pow(0.5, elapsed / half_life_seconds)
        floor = state.decay_policy.floor
        factor = max(floor, retention) if elapsed > 0 else 1.0

        decayed_values = {
            "affection": _decay_positive(state.affection, factor, floor),
            "trust": _decay_positive(state.trust, factor, floor),
            "possessiveness": _decay_positive(state.possessiveness, factor, floor),
            "anxiety": _decay_positive(state.anxiety, factor, floor),
            "distance": _decay_toward(state.distance, target=0.35, factor=factor),
            "boundary_hardness": _decay_toward(state.boundary_hardness, target=0.55, factor=factor),
            "desire_level": _decay_positive(state.desire_level, factor, floor),
            "inertia": _decay_toward(state.inertia, target=0.2, factor=factor),
        }
        residues = tuple(residue for residue in state.residues if residue.decayed(now=current_time))
        intensity = _decayed_intensity(state.intensity, factor)
        valence = Valence.NEUTRAL if intensity is Intensity.CALM and not residues else state.valence
        secondary = state.secondary_emotion if intensity is not Intensity.CALM else None

        return replace(
            state,
            **decayed_values,
            intensity=intensity,
            valence=valence,
            secondary_emotion=secondary,
            residues=residues,
            updated_at=current_time,
        )


def _decay_positive(value: float, factor: float, floor: float) -> float:
    if value <= floor:
        return value
    return max(floor, value * factor)


def _decay_toward(value: float, *, target: float, factor: float) -> float:
    return target + (value - target) * factor


def _decayed_intensity(intensity: Intensity, factor: float) -> Intensity:
    if factor > 0.75:
        return intensity
    if intensity is Intensity.OVERFLOW:
        return Intensity.STRONG
    if intensity is Intensity.STRONG:
        return Intensity.MODERATE
    if intensity is Intensity.MODERATE:
        return Intensity.MILD
    if intensity is Intensity.MILD and factor <= 0.25:
        return Intensity.CALM
    return intensity


def _aware(value: datetime) -> datetime:
    if value.tzinfo is None:
        return value.replace(tzinfo=UTC)
    return value
