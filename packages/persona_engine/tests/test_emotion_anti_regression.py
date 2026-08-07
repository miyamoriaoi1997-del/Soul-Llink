"""Anti-regression checks for dynamic emotion prompt behavior.

These tests intentionally evaluate generated modifier / orchestrator artifacts,
not final model prose. They lock the runtime contract that emotion intensity
changes expression pressure without stealing task mode, bypassing gates, or
turning overwhelming into checklist output.
"""

from emotion_calculator import EmotionCalculator
from emotion_state_manager import EmotionStateManager
from persona_orchestrator import StateOrchestrator


def _write_state(tmp_path, **emotion_state):
    state_path = tmp_path / "STATE.md"
    lines = ["---", "emotion_state:"]
    for key, value in emotion_state.items():
        lines.append(f"  {key}: {value}")
    lines.append("  last_update: '2099-01-01T00:00:00'")
    lines.append("---")
    state_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return state_path


def test_emotion_intensity_guidance_is_length_independent_even_when_overwhelming(tmp_path):
    _write_state(
        tmp_path,
        affection=108,
        trust=108,
        possessiveness=108,
        patience=108,
    )
    block = EmotionStateManager(hermes_home=tmp_path).get_tone_modifiers()

    assert "已经失控" in block
    assert "只抓最核心的一种反应" in block
    assert "必须长篇" not in block
    assert "尽量多写" not in block


def test_high_emotion_technical_request_stays_system_maintenance_with_intimacy_overlay(tmp_path):
    orchestrator = StateOrchestrator(
        ".",
        log_path=tmp_path / "emotion_anti_regression.jsonl",
        enable_semantic_shadow=True,
        semantic_backend="local_lightweight",
        core_source="host_core",
    )

    packet = orchestrator.analyze_turn(
        user_message="老婆，继续检查 emotion_modifier 最后注入和 gateway 日志",
        emotion_state={"emotion_score": 4.6},
        previous_mode="daily",
        platform="telegram",
    )

    assert packet.mode == "work"
    assert "work" in packet.selected_layers
    assert "daily" not in packet.selected_layers
    assert "sex" not in packet.selected_layers


def test_explicit_emotion_value_query_routes_to_system_maintenance_not_intimacy(tmp_path):
    orchestrator = StateOrchestrator(
        ".",
        log_path=tmp_path / "emotion_value_query.jsonl",
        enable_semantic_shadow=True,
        semantic_backend="local_lightweight",
        core_source="host_core",
    )

    packet = orchestrator.analyze_turn(
        user_message="老婆，现在 emotion_score 和情绪值是多少",
        emotion_state={"emotion_score": 4.4},
        previous_mode="daily",
        platform="telegram",
    )

    assert packet.mode == "work"
    assert packet.desire_tier == "uninhibited"
    assert "work" in packet.selected_layers
    assert "sex" not in packet.selected_layers


def test_low_desire_explicit_sex_request_keeps_sex_layer_blocked(tmp_path):
    orchestrator = StateOrchestrator(
        ".",
        log_path=tmp_path / "sex_gate_regression.jsonl",
        enable_active_sex=True,
        core_source="host_core",
    )

    packet = orchestrator.analyze_turn(
        user_message="想和你做爱",
        emotion_state={"emotion_score": 2.8},
        previous_mode="daily",
        platform="telegram",
    )

    assert packet.mode == "daily"
    assert packet.desire_tier == "restrained"
    assert "sex_desire_gate_restrained" in packet.safety_flags
    assert "daily" in packet.selected_layers
    assert "sex" not in packet.selected_layers


def test_overwhelming_guidance_uses_focused_loss_of_control_not_bullet_list():
    result = EmotionCalculator().get_tone_modifiers(
        {
            "affection": 108,
            "trust": 108,
            "possessiveness": 108,
            "patience": 108,
        }
    )

    assert result["overall_intensity"] == "overwhelming"
    assert result["evaluation"] == "强情绪只抓住当下最核心的一种真实反应，不机械堆叠在乎、委屈、占有、解释和身体反应。"
    assert "只抓最核心的一种反应" in result["expression_guidance"]
    assert "其他让位" in result["expression_guidance"]
