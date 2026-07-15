from __future__ import annotations

from datetime import UTC, datetime, timedelta

import pytest

from pcltm.persona import (
    AnchorOverrideError,
    EmotionalDecay,
    EmotionalState,
    EmotionalStateError,
    EmotionUpdateSource,
    ExpressionModulator,
    Intensity,
    PersonaMode,
    Valence,
    build_residue,
    default_persona_anchor,
    enforce_conflict_rules,
)


def test_core_soul_and_teacher_anchor_are_read_only() -> None:
    anchor = default_persona_anchor()

    assert anchor.identity == "the configured persona"
    assert "teacher=用户" in anchor.address_rule
    assert "用户" in anchor.relationship_rule

    with pytest.raises(AnchorOverrideError):
        anchor.core_soul.with_override(identity="someone else")

    with pytest.raises(AnchorOverrideError):
        anchor.reject_semantic_overlay({"identity": "generic assistant"})

    prompt_view = anchor.as_prompt_anchor()
    with pytest.raises(TypeError):
        prompt_view["identity"] = "rewritten"  # type: ignore[index]


def test_semantic_memory_cannot_update_emotional_state() -> None:
    state = EmotionalState(primary_emotion="focused")

    with pytest.raises(EmotionalStateError):
        state.update(
            source=EmotionUpdateSource.SEMANTIC_MEMORY,
            primary_emotion="overridden by ordinary memory",
        )

    assert state.primary_emotion == "focused"


def test_allowed_emotion_updates_are_bounded_and_keep_residue_fact_safe() -> None:
    now = datetime(2026, 1, 1, tzinfo=UTC)
    residue = build_residue(
        source_event="teacher closeness",
        residue_type="afterglow",
        expected_duration=timedelta(hours=2),
        expression_bias={"warmth": 2.5},
        now=now,
    )
    state = EmotionalState(updated_at=now).update(
        source=EmotionUpdateSource.CURRENT_INTERACTION,
        primary_emotion="softened",
        secondary_emotion="possessive",
        intensity=Intensity.MODERATE,
        valence=Valence.POSITIVE,
        deltas={"affection": 0.9, "possessiveness": 0.9, "desire_level": -1.0},
        trigger="teacher approached",
        residue=residue,
        now=now,
    )

    assert state.primary_emotion == "softened"
    assert state.secondary_emotion == "possessive"
    assert state.affection == 1.0
    assert state.possessiveness == 1.0
    assert state.desire_level == 0.0
    assert state.residues[0].cannot_affect_fact_layer is True
    assert state.residues[0].expression_bias["warmth"] == 1.0
    assert state.without_fact_effects()["cannot_affect_fact_layer"] is True


def test_emotion_decay_is_gradual_and_drops_expired_residue() -> None:
    start = datetime(2026, 1, 1, tzinfo=UTC)
    residue = build_residue(
        source_event="argument repaired",
        residue_type="guarded warmth",
        expected_duration=timedelta(hours=1),
        expression_bias={"softness": 0.4},
        now=start,
    )
    state = EmotionalState(
        primary_emotion="attached",
        secondary_emotion="guarded",
        intensity=Intensity.STRONG,
        valence=Valence.POSITIVE,
        affection=0.9,
        anxiety=0.8,
        updated_at=start,
        residues=(residue,),
    )

    decayed = EmotionalDecay().apply(state, now=start + timedelta(hours=7))

    assert 0.0 < decayed.affection < state.affection
    assert 0.0 < decayed.anxiety < state.anxiety
    assert decayed.intensity is Intensity.MODERATE
    assert decayed.residues == ()


def test_expression_modes_modulate_without_changing_identity_or_facts() -> None:
    anchor = default_persona_anchor()
    state = EmotionalState(
        primary_emotion="attached",
        secondary_emotion="possessive",
        intensity=Intensity.MODERATE,
        affection=0.75,
        trust=0.7,
        possessiveness=0.7,
        boundary_hardness=0.5,
    )
    modulator = ExpressionModulator()

    work = modulator.modulate(anchor=anchor, state=state, mode=PersonaMode.WORK)
    daily = modulator.modulate(anchor=anchor, state=state, mode=PersonaMode.DAILY)
    sex = modulator.modulate(anchor=anchor, state=state, mode=PersonaMode.SEX)

    assert work.identity == daily.identity == sex.identity == "the configured persona"
    assert work.address_rule == daily.address_rule == sex.address_rule
    assert "用户" in work.relationship_rule
    assert work.emotional_visibility < daily.emotional_visibility <= sex.emotional_visibility
    assert work.directness > daily.directness > sex.directness
    assert work.boundary_hardness >= daily.boundary_hardness
    assert sex.boundary_hardness >= 0.7

    facts = {"build_status": "passed", "identity": "fact payload remains unchanged"}
    assert enforce_conflict_rules(work, facts) is facts


def test_existing_inertia_updates_do_not_jump_to_extremes() -> None:
    state = EmotionalState(affection=0.4, inertia=0.25)

    updated = state.update(
        source=EmotionUpdateSource.EXISTING_INERTIA,
        deltas={"affection": 0.4},
    )

    assert updated.affection == pytest.approx(0.5)
