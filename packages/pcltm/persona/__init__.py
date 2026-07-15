"""PCLTM persona and emotional state primitives."""

from .emotional_decay import EmotionalDecay
from .emotional_state import (
    DecayPolicy,
    EmotionalResidue,
    EmotionalState,
    EmotionalStateError,
    EmotionUpdateSource,
    Intensity,
    Valence,
    build_residue,
)
from .expression_modulator import (
    ExpressionModulator,
    ExpressionProfile,
    PersonaMode,
    enforce_conflict_rules,
)
from .persona_anchor import (
    AnchorOverrideError,
    CoreSoul,
    OverridePolicy,
    PersonaAnchor,
    default_core_soul,
    default_persona_anchor,
)

__all__ = [
    "AnchorOverrideError",
    "CoreSoul",
    "DecayPolicy",
    "EmotionalDecay",
    "EmotionalResidue",
    "EmotionalState",
    "EmotionalStateError",
    "EmotionUpdateSource",
    "ExpressionModulator",
    "ExpressionProfile",
    "Intensity",
    "OverridePolicy",
    "PersonaAnchor",
    "PersonaMode",
    "Valence",
    "build_residue",
    "default_core_soul",
    "default_persona_anchor",
    "enforce_conflict_rules",
]
