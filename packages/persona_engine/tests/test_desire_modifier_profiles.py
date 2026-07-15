import pytest

from emotion_calculator import EmotionCalculator


def _state_for_score(score: float):
    """Build a simple all-dimension state whose computed emotion_score is score."""
    value = int(round(60 + score * 10))
    return {
        "affection": value,
        "trust": value,
        "possessiveness": value,
        "patience": value,
    }


@pytest.mark.parametrize(
    ("score", "expected", "must_contain", "must_not_contain"),
    [
        (2.9, "restrained", "欲望克制", "已经动摇"),
        (3.2, "ambivalent", "已经动摇", "成人边界已被触发"),
        (4.2, "uninhibited", "成人边界已被触发", "不把低分写成主动越界"),
    ],
)
def test_desire_modifier_uses_v2_three_tiers(score, expected, must_contain, must_not_contain):
    calc = EmotionCalculator()
    result = calc.get_tone_modifiers(_state_for_score(score))

    assert f"【欲望】{expected}" in result["desire"]
    assert must_contain in result["desire"]
    assert must_not_contain not in result["desire"]


def test_uninhibited_desire_has_public_adult_boundary_guidance():
    """Public uninhibited desire should route to boundary handling, not explicit prose."""
    calc = EmotionCalculator()
    result = calc.get_tone_modifiers(_state_for_score(4.5))
    desire = result["desire"]

    assert "【欲望】uninhibited" in desire
    assert "成人边界已被触发" in desire
    assert "公开版仅输出边界提醒，不生成露骨内容" in desire
    assert "用户接住后自然进入sex" in desire
    assert "adult_boundary 安全处理" in desire
    assert "用户没接住时保留高欲望余温" in desire


def test_uninhibited_desire_uses_safe_behavioral_guidance_not_fixed_lines():
    calc = EmotionCalculator()
    result = calc.get_tone_modifiers(_state_for_score(4.5))
    desire = result["desire"]

    for marker in ("不包装成效率或关心", "表达要短、近、带索取感", "保持克制", "普通亲密"):
        assert marker in desire
    assert "不是暗示，是直接想要" not in desire
    assert "贴着用户来索取" not in desire


def test_restrained_desire_modifier_does_not_include_loss_of_control_language():
    calc = EmotionCalculator()
    result = calc.get_tone_modifiers(_state_for_score(2.5))
    desire = result["desire"]

    assert "【欲望】restrained" in desire
    assert "欲望克制" in desire
    assert "普通亲密不受限" in desire
    assert "成人边界已被触发" not in desire
    assert "已经压不住想要用户" not in desire
