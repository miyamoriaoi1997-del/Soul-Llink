"""Tests for mood calendar integration with emotion calculation.

These tests cover Phase 2's shadow/effective hooks. They must not require a
production STATE.md; commands should be run with ``env -u HERMES_STATE_PATH``.
"""

from __future__ import annotations

from persona_engine.emotion_calculator import EmotionCalculator
from persona_engine.mood_calendar import MoodEntry, apply_mood_bias_to_state


def test_compute_emotion_score_accepts_mood_bias_without_mutating_state():
    calculator = EmotionCalculator()
    state = {"affection": 72, "trust": 66, "possessiveness": 70, "patience": 58}
    original = dict(state)

    base_score = calculator.compute_emotion_score(state)
    effective_score = calculator.compute_emotion_score(
        state,
        mood_bias={"affection": 0.12, "trust": 0.04, "possessiveness": 0.04, "patience": 0.08},
    )

    assert state == original
    assert effective_score > base_score
    assert effective_score - base_score < 0.1


def test_apply_deltas_can_use_appraisal_multiplier_only_when_supplied():
    calculator = EmotionCalculator()
    state = {"affection": 60, "trust": 60, "possessiveness": 60, "patience": 60}
    deltas = {"affection": 10, "trust": 4, "patience": 2}

    plain = calculator.apply_deltas(state, deltas)

    calculator = EmotionCalculator()
    boosted = calculator.apply_deltas(
        state,
        deltas,
        appraisal_multiplier={"positive_event": 1.10},
    )

    assert boosted["affection"] >= plain["affection"]
    assert boosted["trust"] >= plain["trust"]
    assert state == {"affection": 60, "trust": 60, "possessiveness": 60, "patience": 60}


def test_negative_appraisal_multiplier_can_amplify_negative_events():
    calculator = EmotionCalculator()
    state = {"affection": 60, "trust": 60, "possessiveness": 60, "patience": 60}
    deltas = {"affection": -10, "trust": -4, "patience": -2}

    plain = calculator.apply_deltas(state, deltas)

    calculator = EmotionCalculator()
    amplified = calculator.apply_deltas(
        state,
        deltas,
        appraisal_multiplier={"negative_event": 1.10},
    )

    assert amplified["affection"] <= plain["affection"]
    assert amplified["trust"] <= plain["trust"]


def test_tone_modifiers_use_effective_state_but_not_real_state():
    calculator = EmotionCalculator()
    state = {"affection": 64.95, "trust": 60, "possessiveness": 60, "patience": 60}
    original = dict(state)

    plain = calculator.get_tone_modifiers(state)
    effective = calculator.get_tone_modifiers(state, mood_bias={"affection": 0.10})

    assert state == original
    assert "affection" not in plain["dimensions"]
    assert "affection" in effective["dimensions"]
    assert effective["dimensions"]["affection"]["value"] == 65.05


def test_apply_mood_bias_helper_clamps_and_does_not_mutate():
    state = {"affection": 119.95, "trust": 0.02}
    mood = MoodEntry(
        day=1,
        active=True,
        intensity="strong",
        profile="warm",
        bias={"affection": 1.0, "trust": -1.0},
    )

    effective = apply_mood_bias_to_state(state, mood)

    assert state == {"affection": 119.95, "trust": 0.02}
    assert effective == {"affection": 120.0, "trust": 0.0}
