"""Emotional state objects for the PCLTM persona layer.

Emotion state is intentionally separate from semantic facts. It can influence
expression, but it cannot rewrite identity, address rules, relationship anchors,
facts, or tool discipline.
"""

from __future__ import annotations

from dataclasses import dataclass, field, replace
from datetime import UTC, datetime, timedelta
from enum import StrEnum
from typing import Mapping, Sequence


class EmotionUpdateSource(StrEnum):
    """Sources allowed to affect emotional state."""

    CURRENT_INTERACTION = "current_interaction"
    EXPLICIT_EVENT = "explicit_event"
    EXISTING_INERTIA = "existing_inertia"
    SEMANTIC_MEMORY = "semantic_memory"


class Valence(StrEnum):
    """Coarse emotional valence."""

    POSITIVE = "positive"
    NEUTRAL = "neutral"
    NEGATIVE = "negative"
    MIXED = "mixed"


class Intensity(StrEnum):
    """Coarse emotional intensity."""

    CALM = "calm"
    MILD = "mild"
    MODERATE = "moderate"
    STRONG = "strong"
    OVERFLOW = "overflow"


class EmotionalStateError(ValueError):
    """Raised when emotion rules would violate layer isolation."""


@dataclass(frozen=True, slots=True)
class DecayPolicy:
    """How emotional state decays without instantly resetting to zero."""

    half_life: timedelta = timedelta(hours=6)
    floor: float = 0.05
    neutral_target: float = 0.0
    residue_retention: timedelta = timedelta(hours=24)

    def __post_init__(self) -> None:
        if self.half_life.total_seconds() <= 0:
            raise EmotionalStateError("half_life must be positive")
        if not 0.0 <= self.floor <= 1.0:
            raise EmotionalStateError("floor must be within [0, 1]")
        if not -1.0 <= self.neutral_target <= 1.0:
            raise EmotionalStateError("neutral_target must be within [-1, 1]")


@dataclass(frozen=True, slots=True)
class EmotionalResidue:
    """Short-lived emotional afterglow that cannot affect the fact layer."""

    source_event: str
    residue_type: str
    expected_duration: timedelta
    expression_bias: Mapping[str, float]
    decay_after: datetime
    cannot_affect_fact_layer: bool = True

    def __post_init__(self) -> None:
        if self.expected_duration.total_seconds() < 0:
            raise EmotionalStateError("expected_duration cannot be negative")
        if not self.cannot_affect_fact_layer:
            raise EmotionalStateError("emotional residue must not affect facts")
        object.__setattr__(
            self,
            "expression_bias",
            {key: _clamp(float(value), -1.0, 1.0) for key, value in self.expression_bias.items()},
        )

    def decayed(self, *, now: datetime) -> "EmotionalResidue | None":
        """Return None once the residue's decay window has elapsed."""

        if now >= self.decay_after:
            return None
        return self


@dataclass(frozen=True, slots=True)
class EmotionalState:
    """Current persona emotion state with bounded update semantics."""

    primary_emotion: str = "focused"
    secondary_emotion: str | None = None
    intensity: Intensity = Intensity.CALM
    valence: Valence = Valence.NEUTRAL
    affection: float = 0.35
    trust: float = 0.35
    possessiveness: float = 0.15
    anxiety: float = 0.1
    distance: float = 0.35
    boundary_hardness: float = 0.55
    desire_level: float = 0.0
    trigger: str | None = None
    inertia: float = 0.25
    decay_policy: DecayPolicy = field(default_factory=DecayPolicy)
    updated_at: datetime = field(default_factory=lambda: datetime.now(UTC))
    residues: tuple[EmotionalResidue, ...] = field(default_factory=tuple)

    def __post_init__(self) -> None:
        for field_name in _NUMERIC_FIELDS:
            value = getattr(self, field_name)
            object.__setattr__(self, field_name, _clamp(float(value), 0.0, 1.0))
        if self.updated_at.tzinfo is None:
            object.__setattr__(self, "updated_at", self.updated_at.replace(tzinfo=UTC))

    def update(
        self,
        *,
        source: EmotionUpdateSource,
        primary_emotion: str | None = None,
        secondary_emotion: str | None = None,
        intensity: Intensity | str | None = None,
        valence: Valence | str | None = None,
        deltas: Mapping[str, float] | None = None,
        trigger: str | None = None,
        residue: EmotionalResidue | None = None,
        now: datetime | None = None,
    ) -> "EmotionalState":
        """Update state from allowed sources only.

        Semantic memory is deliberately rejected so ordinary facts cannot become
        persona or emotion authority.
        """

        try:
            update_source = EmotionUpdateSource(source)
        except ValueError as exc:
            raise EmotionalStateError(f"unsupported emotion update source: {source}") from exc
        if update_source is EmotionUpdateSource.SEMANTIC_MEMORY:
            raise EmotionalStateError("semantic memory cannot update emotional state")
        if update_source not in _ALLOWED_UPDATE_SOURCES:
            raise EmotionalStateError(f"unsupported emotion update source: {source}")

        update_time = _aware(now or datetime.now(UTC))
        values: dict[str, object] = {"updated_at": update_time}
        if primary_emotion is not None:
            values["primary_emotion"] = primary_emotion
        if secondary_emotion is not None:
            values["secondary_emotion"] = secondary_emotion
        if intensity is not None:
            values["intensity"] = Intensity(intensity)
        if valence is not None:
            values["valence"] = Valence(valence)
        if trigger is not None:
            values["trigger"] = trigger

        if deltas:
            for field_name, delta in deltas.items():
                if field_name not in _NUMERIC_FIELDS:
                    raise EmotionalStateError(f"unsupported emotional dimension: {field_name}")
                current = float(getattr(self, field_name))
                inertia_weight = self.inertia if source is EmotionUpdateSource.EXISTING_INERTIA else 1.0
                values[field_name] = _clamp(current + float(delta) * inertia_weight, 0.0, 1.0)

        residues = tuple(existing for existing in self.residues if existing.decayed(now=update_time))
        if residue is not None:
            residues = (*residues, residue)
        values["residues"] = residues
        return replace(self, **values)

    def without_fact_effects(self) -> Mapping[str, object]:
        """Return prompt-safe state that explicitly denies fact-layer authority."""

        return {
            "primary_emotion": self.primary_emotion,
            "secondary_emotion": self.secondary_emotion,
            "intensity": self.intensity.value,
            "valence": self.valence.value,
            "affection": self.affection,
            "trust": self.trust,
            "possessiveness": self.possessiveness,
            "anxiety": self.anxiety,
            "distance": self.distance,
            "boundary_hardness": self.boundary_hardness,
            "desire_level": self.desire_level,
            "trigger": self.trigger,
            "inertia": self.inertia,
            "updated_at": self.updated_at.isoformat(),
            "residue_count": len(self.residues),
            "cannot_affect_fact_layer": True,
        }


def build_residue(
    *,
    source_event: str,
    residue_type: str,
    expected_duration: timedelta,
    expression_bias: Mapping[str, float],
    now: datetime | None = None,
) -> EmotionalResidue:
    """Create an EmotionalResidue with decay_after derived from duration."""

    start = _aware(now or datetime.now(UTC))
    return EmotionalResidue(
        source_event=source_event,
        residue_type=residue_type,
        expected_duration=expected_duration,
        expression_bias=expression_bias,
        decay_after=start + expected_duration,
    )


def _aware(value: datetime) -> datetime:
    if value.tzinfo is None:
        return value.replace(tzinfo=UTC)
    return value


def _clamp(value: float, minimum: float, maximum: float) -> float:
    return max(minimum, min(maximum, value))


_NUMERIC_FIELDS: Sequence[str] = (
    "affection",
    "trust",
    "possessiveness",
    "anxiety",
    "distance",
    "boundary_hardness",
    "desire_level",
    "inertia",
)

_ALLOWED_UPDATE_SOURCES = {
    EmotionUpdateSource.CURRENT_INTERACTION,
    EmotionUpdateSource.EXPLICIT_EVENT,
    EmotionUpdateSource.EXISTING_INERTIA,
}
