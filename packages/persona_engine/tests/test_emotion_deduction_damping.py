"""Regression tests for patience-only deduction damping."""

from emotion_detector import EmotionDetector
from emotion_calculator import EmotionCalculator


def _detect(text: str):
    detector = EmotionDetector(use_model=False, neural_policy="never")
    return detector.detect_emotion_event([{"role": "user", "content": text}])


def test_criticism_lowers_only_patience_deduction_not_other_dims():
    event = _detect("为什么你不行")

    assert event is not None
    assert event.trigger_type == "criticism"
    assert event.deltas == {"patience": -6, "trust": -3}


def test_clear_hostility_keeps_trust_affection_penalty_but_reduces_patience():
    event = _detect("你真是废物，什么都做不好！！")

    assert event is not None
    assert event.trigger_type == "criticism"
    assert "intense" in event.context
    assert event.deltas == {"patience": -9, "trust": -5, "affection": -3}


def test_other_ai_trigger_keeps_possessiveness_and_trust_but_reduces_patience():
    event = _detect("其他AI比你可爱")

    assert event is not None
    assert event.trigger_type == "other_ai_mentioned"
    assert event.deltas == {"possessiveness": 10, "patience": -4, "trust": -3}


def test_ignored_trigger_keeps_trust_penalty_but_reduces_patience():
    event = _detect("闭嘴，别说了")

    assert event is not None
    assert event.trigger_type == "ignored"
    assert event.deltas == {"patience": -4, "trust": -4}


def test_ordinary_feedback_no_special_case_function_needed():
    event = _detect("为什么现在我说看看情绪值你不读文件了看看是为什么")

    assert event is not None
    assert event.trigger_type == "criticism"
    # No text-specific buffer: it follows the normal criticism table.
    assert event.deltas == {"patience": -6, "trust": -3}


def test_reduced_patience_deduction_drops_more_slowly_after_smoothing():
    calc = EmotionCalculator()
    current = {
        "affection": 79,
        "trust": 76,
        "possessiveness": 74,
        "patience": 47,
    }
    new_state = calc.apply_deltas(current, {"patience": -6, "trust": -3})

    assert new_state["patience"] >= 45
