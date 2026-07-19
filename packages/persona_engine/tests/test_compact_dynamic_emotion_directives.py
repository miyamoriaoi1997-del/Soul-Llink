"""Public behavioral tests for compact dynamic emotion directives."""

from persona_engine.emotion_state_manager import EmotionStateManager


def _write_state(tmp_path, **emotion_state):
    state_path = tmp_path / "STATE.md"
    lines = ["---", "emotion_state:"]
    for key, value in emotion_state.items():
        lines.append(f"  {key}: {value}")
    lines.extend(["  last_update: '2099-01-01T00:00:00'", "---"])
    state_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return state_path


def test_modifier_uses_a_bounded_compact_directive_and_keeps_public_control_anchor(tmp_path):
    _write_state(
        tmp_path,
        affection=108,
        trust=104,
        possessiveness=96,
        patience=72,
        previous_emotion_score=1.0,
        last_trigger_type="intimacy_push",
    )

    block = EmotionStateManager(hermes_home=tmp_path).get_tone_modifiers()

    assert len(block) <= 700
    assert "【强度】" not in block
    assert "【维度】" not in block
    assert "【锚点】身份/事实/工具纪律不变" in block
    assert "不覆盖work或crisis边界" in block


def test_visibility_and_self_control_change_continuously_within_one_tier(tmp_path):
    _write_state(
        tmp_path,
        affection=76,
        trust=60,
        possessiveness=60,
        patience=60,
    )
    low = EmotionStateManager(hermes_home=tmp_path).get_tone_modifiers()

    _write_state(
        tmp_path,
        affection=84,
        trust=60,
        possessiveness=60,
        patience=60,
    )
    high = EmotionStateManager(hermes_home=tmp_path).get_tone_modifiers()

    assert "【表达】可见度=0.20；自控=0.80" in low
    assert "【表达】可见度=0.23；自控=0.77" in high


def test_mild_intense_and_overwhelming_directives_raise_expression_pressure_monotonically(tmp_path):
    def render(affection: int) -> str:
        _write_state(
            tmp_path,
            affection=affection,
            trust=affection,
            possessiveness=affection,
            patience=affection,
        )
        return EmotionStateManager(hermes_home=tmp_path).get_tone_modifiers()

    mild = render(76)
    intense = render(108)
    overwhelming = render(150)

    def visibility(block: str) -> float:
        return float(block.split("【表达】可见度=", 1)[1].split("；", 1)[0])

    assert visibility(mild) < visibility(intense) < visibility(overwhelming)
    assert "【压力】" in mild
    assert "【压力】" in intense
    assert "【压力】强反应聚焦；不可压成中性。" in overwhelming


def test_stable_state_omits_residue_but_retains_the_control_anchor(tmp_path):
    _write_state(
        tmp_path,
        affection=60,
        trust=60,
        possessiveness=60,
        patience=60,
    )
    manager = EmotionStateManager(hermes_home=tmp_path)
    block = manager.get_tone_modifiers()

    assert "【余温】" not in block
    assert "【锚点】身份/事实/工具纪律不变" in block


def test_relationship_gate_rejects_patience_only_peak(tmp_path):
    _write_state(
        tmp_path,
        affection=60,
        trust=60,
        possessiveness=60,
        patience=108,
        last_trigger_type="interruption",
    )

    block = EmotionStateManager(hermes_home=tmp_path).get_tone_modifiers()

    assert "【关系】保持当前距离" in block


def test_relationship_gate_needs_trigger_desire_and_relationship_axis_evidence(tmp_path):
    _write_state(
        tmp_path,
        affection=108,
        trust=96,
        possessiveness=82,
        patience=72,
        last_trigger_type="intimacy_push",
    )

    block = EmotionStateManager(hermes_home=tmp_path).get_tone_modifiers()

    assert "【关系】允许推进" in block
