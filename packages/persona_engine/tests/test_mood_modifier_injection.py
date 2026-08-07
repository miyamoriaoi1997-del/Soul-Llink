"""Tests for mood bottom-noise injection into emotion_modifier."""

from __future__ import annotations

from persona_engine.emotion_state_manager import EmotionStateManager
from persona_engine.mood_calendar import MoodEntry


class FixedMoodEmotionStateManager(EmotionStateManager):
    def __init__(self, *args, mood_entry: MoodEntry, **kwargs):
        super().__init__(*args, **kwargs)
        self._fixed_mood_entry = mood_entry

    def _get_today_mood_entry(self) -> MoodEntry:
        return self._fixed_mood_entry


def _write_state(tmp_path, affection=78.0, trust=78.0, possessiveness=78.0, patience=60.0):
    state_path = tmp_path / "STATE.md"
    state_path.write_text(
        "---\n"
        "emotion_state:\n"
        f"  affection: {affection}\n"
        f"  trust: {trust}\n"
        f"  possessiveness: {possessiveness}\n"
        f"  patience: {patience}\n"
        "  last_update: '2099-01-01T00:00:00'\n"
        "---\n",
        encoding="utf-8",
    )
    return state_path


def test_active_mood_injects_daily_bottom_noise_without_mutating_state(tmp_path):
    _write_state(tmp_path)
    mood = MoodEntry(
        day=16,
        active=True,
        intensity="noticeable",
        profile="clingy",
        bias={"affection": 0.12, "trust": 0.02, "possessiveness": 0.14, "patience": -0.04},
        appraisal_multiplier={"positive_event": 1.05, "negative_event": 1.04, "jealousy_event": 1.10},
        hint="今天不太想放手，对距离更敏感。表达可以更靠近，但不要越过用户边界。",
    )
    mgr = FixedMoodEmotionStateManager(hermes_home=tmp_path, mood_entry=mood)

    before = (tmp_path / "STATE.md").read_text(encoding="utf-8")
    block = mgr.get_tone_modifiers()
    after = (tmp_path / "STATE.md").read_text(encoding="utf-8")

    assert before == after
    assert "【日内心情底噪】clingy/noticeable" in block
    assert "不改真实STATE" in block
    assert "不单独触发sex" in block
    assert "不覆盖work或crisis边界" in block
    assert block.index("【强度】") < block.index("【日内心情底噪】") < block.index("【边界】")


def test_inactive_mood_does_not_inject_bottom_noise(tmp_path):
    _write_state(tmp_path)
    mgr = FixedMoodEmotionStateManager(
        hermes_home=tmp_path,
        mood_entry=MoodEntry(day=1, active=False, intensity="none", profile="none"),
    )

    block = mgr.get_tone_modifiers()

    assert "【日内心情底噪】" not in block


def test_active_mood_can_create_modifier_even_when_real_deviation_is_subthreshold(tmp_path):
    _write_state(tmp_path, affection=64.95, trust=60, possessiveness=60, patience=60)
    mood = MoodEntry(
        day=1,
        active=True,
        intensity="mild",
        profile="soft",
        bias={"affection": 0.10},
        hint="今天状态和平时差不多，只是稍微柔软一点。",
    )
    mgr = FixedMoodEmotionStateManager(hermes_home=tmp_path, mood_entry=mood)

    block = mgr.get_tone_modifiers()

    assert "<emotion_modifier>" in block
    assert "【日内心情底噪】soft/mild" in block
    assert "今天状态和平时差不多，只是稍微柔软一点" in block
    assert "【边界】" in block


def test_active_mood_with_no_dimension_lines_still_returns_mood_modifier(tmp_path):
    _write_state(tmp_path, affection=60, trust=60, possessiveness=60, patience=60)
    mood = MoodEntry(
        day=1,
        active=True,
        intensity="mild",
        profile="focused",
        bias={"affection": 0.01, "trust": 0.0, "possessiveness": 0.0, "patience": 0.01},
        hint="今天状态和平时差不多，只是稍微更集中一点。",
    )
    mgr = FixedMoodEmotionStateManager(hermes_home=tmp_path, mood_entry=mood)

    block = mgr.get_tone_modifiers()

    assert "<emotion_modifier>" in block
    assert "【日内心情底噪】focused/mild" in block
    assert "- 【维度】" not in block
