"""Regression tests for patience-only positive recovery tuning."""

from emotion_detector import EmotionDetector


def test_positive_recovery_only_raises_patience_values():
    deltas = EmotionDetector.TRIGGER_DELTAS

    assert deltas["intimacy"]["deltas"] == {
        "mild": {"possessiveness": 5, "affection": 3, "patience": 2},
        "moderate": {"possessiveness": 10, "affection": 6, "trust": 3, "patience": 4},
        "intense": {"possessiveness": 15, "affection": 10, "trust": 5, "patience": 6},
    }
    assert deltas["care"]["deltas"] == {
        "mild": {"trust": 4, "affection": 2, "patience": 3},
        "moderate": {"trust": 8, "affection": 4, "patience": 5},
        "intense": {"trust": 12, "affection": 6, "patience": 7},
    }
    assert deltas["encouragement"]["deltas"] == {
        "mild": {"trust": 3, "affection": 1, "patience": 2},
        "moderate": {"trust": 6, "affection": 3, "patience": 4},
        "intense": {"trust": 9, "affection": 5, "patience": 6},
    }
    assert deltas["apology"]["deltas"] == {
        "mild": {"trust": 3, "patience": 5, "affection": 1},
        "moderate": {"trust": 5, "patience": 10, "affection": 3},
        "intense": {"trust": 8, "patience": 14, "affection": 5},
    }


def test_negative_patience_tuning_remains_unchanged():
    deltas = EmotionDetector.TRIGGER_DELTAS

    assert deltas["other_ai_mentioned"]["deltas"] == {
        "mild": {"possessiveness": 5, "patience": -2},
        "moderate": {"possessiveness": 10, "patience": -4, "trust": -3},
        "intense": {"possessiveness": 15, "patience": -7, "trust": -5},
    }
    assert deltas["criticism"]["deltas"] == {
        "mild": {"patience": -3},
        "moderate": {"patience": -6, "trust": -3},
        "intense": {"patience": -9, "trust": -5, "affection": -3},
    }
    assert deltas["ignored"]["deltas"] == {
        "mild": {"patience": -2, "trust": -2},
        "moderate": {"patience": -4, "trust": -4},
        "intense": {"patience": -7, "trust": -6},
    }
