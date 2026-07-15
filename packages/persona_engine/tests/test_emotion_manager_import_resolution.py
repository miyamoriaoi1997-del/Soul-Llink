from pathlib import Path

from mood_calendar import MoodEntry


def test_production_emotion_manager_uses_local_calculator_with_blend_fields(tmp_path):
    from emotion_state_manager import EmotionStateManager

    state_path = tmp_path / "STATE.md"
    state_path.write_text(
        "---\n"
        "emotion_state:\n"
        "  affection: 92\n"
        "  trust: 38\n"
        "  possessiveness: 82\n"
        "  patience: 43\n"
        "  last_update: '2099-01-01T00:00:00'\n"
        "---\n",
        encoding="utf-8",
    )

    mgr = EmotionStateManager(hermes_home=tmp_path, state_path=state_path)
    calculator_file = Path(mgr.calculator.__class__.__module__.replace('.', '/'))

    block = mgr.get_tone_modifiers()

    assert mgr.calculator.__class__.__module__ == "emotion_calculator"
    assert calculator_file == Path("emotion_calculator")
    assert "<emotion_modifier>" in block
    assert "【欲望】restrained" in block
    assert "【强度】intense / positive" in block
    assert "靠近但防备" in block
    assert "占有欲与不耐烦同时上升" in block
    assert "【情绪】" in block
    assert "余温=正向高峰余温" in block
    assert "调节=嘴硬压抑+追问确认" in block
    assert "【维度】" in block


def test_production_emotion_manager_injects_mood_ground_tone_without_state_write(monkeypatch, tmp_path):
    from emotion_state_manager import EmotionStateManager

    state_path = tmp_path / "STATE.md"
    original_state = (
        "---\n"
        "emotion_state:\n"
        "  affection: 92\n"
        "  trust: 38\n"
        "  possessiveness: 82\n"
        "  patience: 43\n"
        "  last_update: '2099-01-01T00:00:00'\n"
        "---\n"
    )
    state_path.write_text(original_state, encoding="utf-8")

    def fake_today_mood_entry():
        return MoodEntry(
            day=1,
            active=True,
            intensity="noticeable",
            profile="clingy",
            bias={"affection": 0.1, "possessiveness": 0.1},
            hint="今天不太想放手，对距离更敏感。",
        )

    monkeypatch.setattr("emotion_state_manager.get_today_mood_entry", fake_today_mood_entry)

    mgr = EmotionStateManager(hermes_home=tmp_path, state_path=state_path)
    block = mgr.get_tone_modifiers()

    assert "【日内心情底噪】clingy/noticeable" in block
    assert "今天不太想放手，对距离更敏感。" in block
    assert "不改真实STATE" in block
    assert "不单独触发sex" in block
    assert state_path.read_text(encoding="utf-8") == original_state
