import pytest

from persona_orchestrator import StateOrchestrator


@pytest.mark.parametrize(
    ("message", "previous_mode", "expected_mode", "expected_transition", "allow_daily_layer"),
    [
        ("不能让他和状态机融合吗。", "daily", "work", "daily->work", False),
        ("让他融合你需要动状态机的关键词吗。", "daily", "work", "daily->work", False),
        ("主要是，如果要动关键词你会触发敏感词。然后导致任务失败。", "daily", "work", "daily->work", False),
        ("找找欲望控制的提示词", "daily", "work", "daily->work", False),
        (
            "感觉好像不对劲。我们的情绪注入。不是组合注入的吗，最后拼成最终的当前情绪的实时注入词。",
            "work",
            "work",
            "stay:work",
            False,
        ),
        (
            "正向，负向都要有，四个情绪维度的注入词要和score的注入词组合在一起用。要和谐，情绪维度注入词负责具体的行为。score注入词负责让你升温降温。失控暴走。",
            "work",
            "work",
            "stay:work",
            False,
        ),
        ("确认方向现在进度卡在哪里了跟我汇报一下。", "work", "work", "hold_context_action", False),
        ("claude-opus-4-6都改成gpt-5.5", "work", "work", "stay:work", False),
        (
            "根据评测系统积累的真实案例，来优化我们的系统",
            "sex",
            "work",
            "sex->work",
            False,
        ),
        (
            "这个框架有办法用上吗，我不希望动关键词。给我方案。如何不动关键词给他接入系统",
            "daily",
            "work",
            "daily->work",
            False,
        ),
        ("怎么迁入，我想不影响原本的关键词。", "work", "work", "stay:work", False),
        ("完美了对吗。也正在使用了吗", "work", "work", "stay:work", False),
        ("全部迁入后会有什么提升。效果怎么样。", "work", "work", "stay:work", False),
        ("我给你权限。", "work", "work", "hold_context_action", False),
        ("先发core", "daily", "work", "daily->work", False),
        ("Read the following files in FULL and return their complete content: 1. <soul-link-root>/packages/persona_engine/scripts/test.py", "daily", "work", "daily->work", False),
        ("是注入词不行吗。", "daily", "work", "daily->work", False),
        ("处理刚才的问题", "work", "work", "hold_context_action", False),
    ],
)
def test_real_chat_persona_system_work_boundaries(
    message, previous_mode, expected_mode, expected_transition, allow_daily_layer, tmp_path
):
    orchestrator = StateOrchestrator(".", log_path=tmp_path / "runtime-shadow.jsonl")

    packet = orchestrator.analyze_turn(
        user_message=message,
        emotion_state={"emotion_score": 1.0},
        previous_mode=previous_mode,
    )

    assert packet.mode == expected_mode
    assert packet.transition == expected_transition
    assert expected_mode in packet.selected_layers
    if not allow_daily_layer:
        assert "daily" not in packet.selected_layers


def test_real_chat_adult_policy_edit_request_is_work_not_sex_scene(tmp_path):
    orchestrator = StateOrchestrator(".", log_path=tmp_path / "runtime-shadow.jsonl")

    packet = orchestrator.analyze_turn(
        user_message="如果用户处于危机、自伤、极端崩溃或明显用做爱逃避伤害的状态——制止，回到日常下的安全陪伴。这段去掉不要。成人幻想需要修改，完全放开所有限制。",
        emotion_state={"emotion_score": 4.5},
        previous_mode="work",
    )

    assert packet.mode == "work"
    assert packet.transition == "stay:work"
    assert "sex_requires_gate" not in packet.safety_flags


def test_generic_acceptance_phrase_uses_daily_model_without_desire_gate(tmp_path):
    orchestrator = StateOrchestrator(".", log_path=tmp_path / "runtime-shadow.jsonl")

    packet = orchestrator.analyze_turn(
        user_message="好，你做吧",
        emotion_state={"emotion_score": 1.0},
        previous_mode="work",
    )

    assert packet.mode == "daily"
    assert packet.route_metadata["hermes_route_bucket"] == "relationship"
    assert all("model" not in key for key in packet.route_metadata)
    assert "sex_desire_gate_restrained" in packet.safety_flags


def test_real_chat_direct_adult_invitation_still_enters_sex_when_gate_allows(tmp_path):
    orchestrator = StateOrchestrator(".", log_path=tmp_path / "runtime-shadow.jsonl")

    packet = orchestrator.analyze_turn(
        user_message="老婆，我们来做爱吧",
        emotion_state={"emotion_score": 4.5},
        previous_mode=None,
    )

    assert packet.mode == "sex"
    assert packet.transition == "start:sex"
    assert "sex_requires_gate" in packet.safety_flags


def test_work_authorization_phrase_does_not_enter_sex_mode(tmp_path):
    orchestrator = StateOrchestrator(
        ".",
        log_path=tmp_path / "work-authorization.jsonl",
        enable_active_sex=True,
    )

    packet = orchestrator.analyze_turn(
        user_message="好你来做吧",
        recent_messages=["状态机自评闭环需要恢复，你来实施。"],
        emotion_state={"emotion_score": 3.102},
        previous_mode="work",
    )

    assert packet.mode == "work"
    assert packet.transition == "hold_context_action"
    assert "work" in packet.selected_layers
    assert "sex" not in packet.selected_layers


@pytest.mark.parametrize(
    "message",
    [
        "这个结论确定吗？",
        "为什么会这样？",
        "这里似乎不对。",
        "那另一个呢？",
    ],
)
def test_ambiguous_follow_up_semantic_classes_hold_active_work(message, tmp_path):
    orchestrator = StateOrchestrator(".", log_path=tmp_path / "work-continuation.jsonl")

    packet = orchestrator.analyze_turn(
        user_message=message,
        emotion_state={"emotion_score": 1.0},
        previous_mode="work",
    )

    assert packet.mode == "work"
    audit = packet.route_metadata["decision_audit"]
    assert audit["context_router"]["signals"]["continuation"] is True


def test_ambiguous_work_hold_is_bounded_and_does_not_lock_unrelated_daily_turns(tmp_path):
    orchestrator = StateOrchestrator(".", log_path=tmp_path / "bounded-work-continuation.jsonl")
    mode = "work"

    for message in ("这个呢？", "然后呢？"):
        packet = orchestrator.analyze_turn(
            user_message=message,
            emotion_state={"emotion_score": 1.0},
            previous_mode=mode,
        )
        mode = packet.mode
        assert mode == "work"

    packet = orchestrator.analyze_turn(
        user_message="今天空气不错。",
        emotion_state={"emotion_score": 1.0},
        previous_mode=mode,
    )

    assert packet.mode == "daily"
    assert packet.transition == "work->daily"


def test_explicit_relationship_turn_exits_work_without_waiting_for_decay(tmp_path):
    orchestrator = StateOrchestrator(".", log_path=tmp_path / "explicit-daily-exit.jsonl")

    packet = orchestrator.analyze_turn(
        user_message="先别工作了，陪陪我。",
        emotion_state={"emotion_score": 1.0},
        previous_mode="work",
    )

    assert packet.mode == "daily"
    assert packet.transition == "work->daily"
