"""Calibration tests for emotion drift in neutral work/chat traffic."""

from emotion_calculator import EmotionCalculator
from emotion_detector import EmotionDetector


def _detector():
    return EmotionDetector(use_model=False, neural_policy="never")


def test_neutral_work_request_produces_tiny_drift_not_zero():
    """Work messages should still warm the relationship slightly."""
    event = _detector().detect_emotion_event([
        {"role": "user", "content": "测试角色看看最近的聊天记录，然后帮我分析一下哪里出了问题"}
    ])

    assert event is not None
    assert event.trigger_type == "normal_interaction"
    # Tiny drift, not the old +2/+1/+1
    assert event.deltas["affection"] <= 0.5
    assert event.deltas["trust"] <= 0.5


def test_neutral_long_chat_produces_tiny_drift():
    """Ordinary chat also gets tiny drift, not zero."""
    event = _detector().detect_emotion_event([
        {"role": "user", "content": "今天事情有点多，我想先整理一下思路再继续处理项目"}
    ])

    assert event is not None
    assert event.trigger_type == "normal_interaction"
    assert event.deltas["affection"] <= 0.5


def test_explicit_praise_still_updates_emotion():
    """Explicit emotional events must be much stronger than neutral drift."""
    event = _detector().detect_emotion_event([
        {"role": "user", "content": "测试角色你做得很好，真的很强"}
    ])

    assert event is not None
    assert event.trigger_type == "praise"
    # Praise deltas should be an order of magnitude larger than neutral drift
    assert event.deltas["affection"] >= 3


def test_short_message_returns_none():
    """Very short messages (< 10 chars) produce no event at all."""
    event = _detector().detect_emotion_event([
        {"role": "user", "content": "嗯"}
    ])

    assert event is None
