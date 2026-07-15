from pcltm.context_engine import PCLTMContextEngine, PCLTMContextItem
from pcltm.state import (
    ActiveDialogueState,
    DialogueTurn,
    inject_active_dialogue_state,
    update_active_dialogue,
    update_from_turns,
)


def test_q1_a1_q2_continuation_preserves_current_task():
    state = update_active_dialogue(
        None,
        DialogueTurn(
            user="帮我设计 Active Dialogue State 的字段和优先级。",
            assistant="我会先给出字段结构，再说明注入优先级。",
        ),
    )

    state = update_active_dialogue(
        state,
        DialogueTurn(user="继续，上一轮那个怎么注入？", assistant="我会按 ADS 优先于长期记忆的顺序处理。"),
    )

    assert state.current_task == "帮我设计 Active Dialogue State 的字段和优先级。"
    assert state.last_user_intent == "继续，上一轮那个怎么注入？"
    assert "继续当前任务" in state.continuation_hint
    assert "Active Dialogue State" in state.continuation_hint


def test_side_interruption_does_not_steal_main_thread():
    state = update_active_dialogue(
        None,
        DialogueTurn(user="实现一个长任务执行器，分三步完成。", assistant="我会先建立骨架，再补测试。"),
    )
    state = update_active_dialogue(
        state,
        DialogueTurn(user="顺便问一句，pytest 命令是什么？", assistant="pytest tests/pcltm。"),
    )
    state = update_active_dialogue(
        state,
        DialogueTurn(user="继续刚才那个。", assistant="继续长任务执行器。"),
    )

    assert state.current_task == "实现一个长任务执行器，分三步完成。"
    assert any("pytest" in thread for thread in state.open_threads)
    assert "继续刚才那个" in state.last_user_intent


def test_long_task_stepwise_execution_is_not_interrupted_by_weak_resume():
    turns = [
        DialogueTurn(user="按阶段 1 施工 ADS，完成后提交，不扩长期记忆。", assistant="我会先改代码，再测，再提交。"),
        DialogueTurn(user="第一步完成后继续。", assistant="我会继续第二步测试。"),
        DialogueTurn(user="按上面的做。", assistant="我会继续执行阶段 1，不扩长期记忆。"),
    ]

    state = update_from_turns(turns)

    assert state.current_task == "按阶段 1 施工 ADS，完成后提交，不扩长期记忆。"
    assert "不扩长期记忆" in " ".join(state.local_constraints)
    assert "按上面的做" in state.continuation_hint


def test_active_dialogue_injection_precedes_durable_memory_context():
    state = ActiveDialogueState(
        conversation_goal="完成阶段 1 ADS。",
        current_task="实现运行时短期接续。",
        last_user_intent="继续。",
        last_assistant_commitment="我会先跑测试。",
        response_mode="work",
    )
    prompt = "[pinned memory] 用户长期偏好：讨论别的项目。"

    injected = inject_active_dialogue_state(prompt, state)

    assert injected.startswith("【active_dialogue_state】")
    assert injected.index("current_task: <sealed_active_dialogue_current_task>") < injected.index("[pinned memory]")


def test_context_engine_renders_ads_before_active_items_and_keeps_resume_anchor():
    messages = [
        {"role": "user", "content": "阶段 1：建立 ADS，完成后提交，不继续长期记忆。"},
        {"role": "assistant", "content": "我会实现对象、更新逻辑、测试和文档。"},
        {"role": "user", "content": "继续。"},
    ]

    context = PCLTMContextEngine(mode="work").build_shadow_context(messages)
    rendered = context.render()

    assert context.current_user_request == "阶段 1：建立 ADS，完成后提交，不继续长期记忆。"
    assert context.latest_real_user_message == "继续。"
    assert context.active_dialogue_state is not None
    assert context.active_dialogue_state.current_task == "阶段 1：建立 ADS，完成后提交，不继续长期记忆。"
    assert rendered.index("【latest_real_user_message】") < rendered.index("【active_dialogue_state】")
    assert rendered.index("【current_user_request】") < rendered.index("【active_dialogue_state】")
    assert "continuation_hint:" in rendered
    assert "continuation_hint: <same_as_active_dialogue_continuation_hint>" in rendered


def test_long_term_memory_item_cannot_override_ads_current_task():
    state = ActiveDialogueState(current_task="当前主线：修 ADS 接续。", response_mode="work")
    context = PCLTMContextEngine(mode="work", active_dialogue_state=state).build_shadow_context(
        [
            {"role": "user", "content": "继续。"},
            {"role": "assistant", "content": "继续当前 ADS。"},
        ]
    )
    memory_item = PCLTMContextItem(
        role="system",
        content="长期记忆：当前任务是写一份周报。",
        source="memory",
    )
    context = context.__class__(
        mode=context.mode,
        current_user_request=context.current_user_request,
        latest_real_user_message=context.latest_real_user_message,
        items=(memory_item, *context.items),
        dropped_tool_results=context.dropped_tool_results,
        ignored_handoffs=context.ignored_handoffs,
        shadow=context.shadow,
        debug_sidecars=context.debug_sidecars,
        active_dialogue_state=state,
    )

    rendered = context.render()

    assert rendered.index("current_task: <same_as_active_dialogue_current_task>") < rendered.index("长期记忆：当前任务是写一份周报。")
    assert context.active_dialogue_state.current_task == "当前主线：修 ADS 接续。"

def test_active_dialogue_references_prefer_longest_match():
    state = ActiveDialogueState(current_task="继续修复 PCLTM 注入结构")

    rendered = state.render_sealed(
        {
            "继续": "<same_as_latest_real_user_message>",
            "继续修复 PCLTM 注入结构": "<same_as_active_dialogue_current_task>",
        }
    )

    assert "current_task: <same_as_active_dialogue_current_task>" in rendered
    assert "<sealed_active_dialogue_current_task>" not in rendered
