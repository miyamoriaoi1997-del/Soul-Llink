from emotion_calculator import EmotionCalculator
from emotion_state_manager import EmotionStateManager
from mood_calendar import MoodEntry


class FixedMoodEmotionStateManager(EmotionStateManager):
    def __init__(self, *args, mood_entry: MoodEntry, **kwargs):
        super().__init__(*args, **kwargs)
        self._fixed_mood_entry = mood_entry

    def _get_today_mood_entry(self) -> MoodEntry:
        return self._fixed_mood_entry


def _polarity_text(result):
    return "\n".join([
        result["framework"],
        *[d["instruction"] for d in result["dimensions"].values() if d["direction"] == result["overall_direction"]],
        result["footnote"],
    ])


def test_positive_overwhelming_injects_only_positive_framework_words():
    calc = EmotionCalculator()
    result = calc.get_tone_modifiers({
        "affection": 108,
        "trust": 108,
        "possessiveness": 108,
        "patience": 108,
    })
    text = _polarity_text(result)

    assert result["overall_intensity"] == "overwhelming"
    assert result["overall_direction"] == "positive"
    assert "【强度】overwhelming / positive" in text
    assert "喜欢、依赖、索取、示弱" in text
    assert "不包装" in text
    assert "negative" not in text
    assert "爆发式不满" not in text
    assert "礼貌冷淡" not in text


def test_negative_overwhelming_injects_only_negative_framework_words():
    calc = EmotionCalculator()
    result = calc.get_tone_modifiers({
        "affection": 12,
        "trust": 12,
        "possessiveness": 12,
        "patience": 12,
    })
    text = _polarity_text(result)

    assert result["overall_intensity"] == "overwhelming"
    assert result["overall_direction"] == "negative"
    assert "【强度】overwhelming / negative" in text
    assert "防备、质问、收缩、追问" in text
    assert "不把关系判死刑" in text
    assert "positive" not in text
    assert "喜欢、依赖、索取、示弱" not in text


def test_desire_gate_is_scoped_to_explicit_sex_not_ordinary_affection():
    calc = EmotionCalculator()

    restrained = calc._compute_desire_instruction(-2.0)
    ambivalent = calc._compute_desire_instruction(3.2)
    uninhibited = calc._compute_desire_instruction(4.5)

    assert "普通亲密不受限" in restrained
    assert "不主动推进性意味" in restrained
    assert "用户靠近或挑逗" in ambivalent
    assert "普通亲密不受限" in ambivalent
    assert "用户接住后自然进入sex" in uninhibited
    assert "用户没接住时保留高欲望余温" in uninhibited


def test_emotion_manager_filters_dimension_lines_to_current_polarity(tmp_path):
    state_path = tmp_path / "STATE.md"
    state_path.write_text(
        "---\n"
        "emotion_state:\n"
        "  affection: 108\n"
        "  trust: 108\n"
        "  possessiveness: 108\n"
        "  patience: 12\n"
        "  last_update: '2099-01-01T00:00:00'\n"
        "---\n",
        encoding="utf-8",
    )
    mgr = FixedMoodEmotionStateManager(
        hermes_home=tmp_path,
        state_path=state_path,
        mood_entry=MoodEntry(
            day=1,
            active=True,
            intensity="mild",
            profile="focused",
            bias={"affection": 0.01},
            hint="今天状态和平时差不多，只是稍微更集中一点。",
        ),
    )

    block = mgr.get_tone_modifiers()

    assert "【情绪】累但不推开/占有欲与不耐烦同时上升" in block
    assert "【触发】失控风险" in block
    assert "【余温】负向高峰补救压力" in block
    assert "【表达】可见度=" in block
    assert "【日内心情底噪】" in block
    assert "不改真实STATE" in block
    assert "不单独触发sex" in block


def test_aftereffect_does_not_override_current_positive_direction(tmp_path):
    state_path = tmp_path / "STATE.md"
    state_path.write_text(
        "---\n"
        "emotion_state:\n"
        "  affection: 72\n"
        "  trust: 26\n"
        "  possessiveness: 98\n"
        "  patience: 52\n"
        "  previous_emotion_score: 3.5\n"
        "  last_trigger_type: needed\n"
        "  last_update: '2099-01-01T00:00:00'\n"
        "---\n",
        encoding="utf-8",
    )
    mgr = EmotionStateManager(hermes_home=tmp_path, state_path=state_path)

    block = mgr.get_tone_modifiers()
    desire_pos = block.index("【欲望】")
    expression_pos = block.index("【表达】")

    assert "【情绪】受伤后的控制欲/占有欲浮现" in block
    assert "【余温】负向高峰补救压力" in block
    assert "【欲望】" in block
    assert desire_pos < expression_pos
    assert "【维度】" not in block
