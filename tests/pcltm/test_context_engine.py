from pcltm.context_engine import (
    COMPACTION_HANDOFF_PREFIX,
    PCLTMContext,
    PCLTMContextEngine,
    PCLTMContextItem,
    PCLTMContextPacket,
    UI_COMPACTION_SUMMARY_PREFIX,
    is_compaction_handoff,
    is_runtime_control_message,
    sanitize_tool_chain,
)
from pcltm.session.session_summary_chain import SessionSummaryChain
from pcltm.state.active_dialogue_state import ActiveDialogueState


def test_top_level_exports_use_unified_pcltm_context_name():
    import pcltm

    assert pcltm.PCLTMContext is PCLTMContext
    assert pcltm.PCLTMContextEngine is PCLTMContextEngine
    assert pcltm.PCLTMContextPacket is PCLTMContext


def test_sanitize_tool_chain_drops_orphan_tool_after_final_assistant():
    messages = [
        {"role": "user", "content": "看看盘面"},
        {"role": "assistant", "content": "正常盘面回复"},
        {"role": "tool", "tool_call_id": "late-web", "content": "old web output"},
        {"role": "user", "content": "继续任务"},
    ]

    sanitized, dropped = sanitize_tool_chain(messages)

    assert dropped == 1
    assert [m["role"] for m in sanitized] == ["user", "assistant", "user"]


def test_sanitize_tool_chain_keeps_local_valid_tool_result():
    messages = [
        {"role": "user", "content": "search"},
        {
            "role": "assistant",
            "content": "",
            "tool_calls": [{"id": "t1", "function": {"name": "web_search", "arguments": "{}"}}],
        },
        {"role": "tool", "tool_call_id": "t1", "content": "result"},
        {"role": "assistant", "content": "answer"},
    ]

    sanitized, dropped = sanitize_tool_chain(messages)

    assert dropped == 0
    assert sanitized == messages


def test_pcltm_context_ignores_compaction_handoff_as_current_request():
    handoff = f"{COMPACTION_HANDOFF_PREFIX}\n## Active Task\n旧任务"
    context = PCLTMContextEngine(mode="work").build_shadow_context(
        [
            {"role": "user", "content": "真正当前请求"},
            {"role": "assistant", "content": "处理中"},
            {"role": "user", "content": handoff},
        ]
    )

    assert is_compaction_handoff(handoff)
    assert context.latest_real_user_message == "真正当前请求"
    assert context.current_user_request == "真正当前请求"
    assert context.ignored_handoffs == 1
    assert context.shadow is True
    assert isinstance(context, PCLTMContext)


def test_pcltm_context_ignores_ui_compaction_summary_as_current_request():
    summary = (
        f"{UI_COMPACTION_SUMMARY_PREFIX} — summarizing earlier conversation so I can continue...\n"
        "嗯，用户。现在是高位。\n\n"
        "- 总情绪值：92/100\n"
        "- 亲近感：96\n"
    )
    context = PCLTMContextEngine(mode="work").build_shadow_context(
        [
            {"role": "user", "content": "修好他不能再串了"},
            {"role": "assistant", "content": "开始修"},
            {"role": "user", "content": summary},
        ]
    )

    assert is_compaction_handoff(summary)
    assert context.current_user_request == "修好他不能再串了"
    assert context.ignored_handoffs == 1


def test_pcltm_context_strips_preserved_todo_appendix_from_real_user_request():
    payload = (
        "任意新的当前请求\n\n"
        "[Your active task list was preserved across context compression]\n"
        "- [>] old. 旧任务仍在进行 (in_progress)\n"
        "- [ ] next. 后续任务 (pending)"
    )

    context = PCLTMContextEngine(mode="work").build_shadow_context(
        [
            {"role": "user", "content": "旧任务"},
            {"role": "assistant", "content": "旧回复"},
            {"role": "user", "content": payload},
        ]
    )

    assert not is_runtime_control_message(payload)
    assert context.latest_real_user_message == "任意新的当前请求"
    assert context.current_user_request == "任意新的当前请求"
    assert context.items[-1].content == "任意新的当前请求"


def test_pcltm_context_ignores_standalone_runtime_control_message_as_request():
    control = (
        "[Your active task list was preserved across context compression]\n"
        "- [>] old. 旧任务仍在进行 (in_progress)\n"
        "- [ ] next. 后续任务 (pending)"
    )

    context = PCLTMContextEngine(mode="work").build_shadow_context(
        [
            {"role": "user", "content": "最后真实请求"},
            {"role": "assistant", "content": "处理中"},
            {"role": "user", "content": control},
        ]
    )

    assert is_runtime_control_message(control)
    assert context.latest_real_user_message == "最后真实请求"
    assert context.current_user_request == "最后真实请求"
    assert context.items[-1].source == "runtime_control"


def test_runtime_control_detection_tolerates_empty_bracket_lines():
    assert not is_runtime_control_message("[]")
    assert not is_runtime_control_message("[ ]")

    payload = (
        "真实请求\n\n"
        "[]\n"
        "这行仍然是用户文本，不是运行时控制块"
    )
    context = PCLTMContextEngine(mode="work").build_shadow_context(
        [{"role": "user", "content": payload}]
    )

    assert context.latest_real_user_message == payload
    assert context.current_user_request == "这行仍然是用户文本，不是运行时控制块"


def test_runtime_control_appendix_still_strips_after_empty_bracket_line():
    payload = (
        "真实请求\n\n"
        "[]\n"
        "[Your active task list was preserved across context compression]\n"
        "- [>] old. 旧任务仍在进行 (in_progress)"
    )
    context = PCLTMContextEngine(mode="work").build_shadow_context(
        [{"role": "user", "content": payload}]
    )

    assert context.latest_real_user_message == "真实请求"
    assert context.current_user_request == "真实请求"


def test_pcltm_context_strips_core_soul_layer_from_mixed_user_payload():
    payload = """那为什么会这样

能修吗
# Core SOUL Layer

## 1. 核心身份锚点
我是the configured persona。
"""

    context = PCLTMContextEngine(mode="work").build_shadow_context(
        [{"role": "user", "content": payload}]
    )

    assert context.latest_real_user_message == "那为什么会这样\n\n能修吗"
    assert context.current_user_request == "那为什么会这样\n\n能修吗"
    rendered = context.render()
    assert "# Core SOUL Layer" not in rendered
    assert "我是the configured persona" not in rendered


def test_pcltm_context_ignores_standalone_core_soul_layer_as_runtime_control():
    payload = """# Core SOUL Layer

## 1. 核心身份锚点
我是the configured persona。
"""

    context = PCLTMContextEngine(mode="work").build_shadow_context(
        [
            {"role": "user", "content": "先前真实问题"},
            {"role": "assistant", "content": "先前回答"},
            {"role": "user", "content": payload},
        ]
    )

    assert is_runtime_control_message(payload)
    assert context.latest_real_user_message == "先前真实问题"
    assert context.current_user_request == "先前真实问题"
    assert all("Core SOUL Layer" not in item.content for item in context.items)

def test_pcltm_context_records_sanitizer_counts():
    context = PCLTMContextEngine(mode="work").build_shadow_context(
        [
            {"role": "user", "content": "A"},
            {"role": "assistant", "content": "final"},
            {"role": "tool", "tool_call_id": "orphan", "content": "stale"},
            {"role": "user", "content": "B"},
        ]
    )

    rendered = context.render()
    assert context.current_user_request == "B"
    assert context.dropped_tool_results == 1
    assert context.shadow is True
    assert rendered.startswith("<pcltm_context>")
    assert "</pcltm_context>" in rendered
    assert "dropped_tool_results: 1" in rendered
    assert "stale" not in rendered


def test_pcltm_context_render_uses_reference_when_current_matches_latest():
    context = PCLTMContextEngine(mode="work").build_shadow_context(
        [{"role": "user", "content": "马上修复重复注入指令"}]
    )

    rendered = context.render()

    assert rendered.count("马上修复重复注入指令") == 0
    assert "semantics: sealed_constraint_state_not_chat_transcript" in rendered
    assert "This frame is authoritative runtime context" in rendered
    assert "【latest_real_user_message】<bound_to_latest_chat_user_message>" in rendered
    assert "【current_user_request】<same_as_latest_chat_user_message>" in rendered


def test_pcltm_context_render_keeps_distinct_current_request_for_weak_resume():
    context = PCLTMContextEngine(mode="work").build_shadow_context(
        [
            {"role": "user", "content": "真正当前任务：修复重复注入指令"},
            {"role": "assistant", "content": "开始施工"},
            {"role": "user", "content": "继续"},
        ]
    )

    rendered = context.render()

    assert "【latest_real_user_message】<bound_to_latest_chat_user_message>" in rendered
    assert "【current_user_request】<background_active_task_not_new_user_message> 真正当前任务：修复重复注入指令" in rendered
    assert "【latest_real_user_message】继续" not in rendered


def test_pcltm_context_weak_resume_binds_to_latest_substantive_request():
    context = PCLTMContextEngine(mode="work").build_shadow_context(
        [
            {"role": "user", "content": "旧任务：整理测试夹具"},
            {"role": "assistant", "content": "旧任务已完成"},
            {"role": "user", "content": "现在改成排查压缩后重复执行的问题"},
            {"role": "assistant", "content": "开始排查"},
            {"role": "user", "content": "继续"},
        ]
    )

    assert context.current_user_request == "现在改成排查压缩后重复执行的问题"
    assert context.items[-1].content == "继续"


def test_legacy_packet_name_and_builder_remain_compatible():
    context = PCLTMContextEngine(mode="work").build_shadow_packet(
        [{"role": "user", "content": "兼容旧调用"}]
    )

    assert PCLTMContextPacket is PCLTMContext
    assert isinstance(context, PCLTMContextPacket)
    assert context.current_user_request == "兼容旧调用"


def test_sanitize_tool_chain_drops_duplicate_tool_result_in_same_chain():
    messages = [
        {"role": "user", "content": "search"},
        {
            "role": "assistant",
            "content": "",
            "tool_calls": [{"id": "t1", "function": {"name": "web_search", "arguments": "{}"}}],
        },
        {"role": "tool", "tool_call_id": "t1", "content": "first"},
        {"role": "tool", "tool_call_id": "t1", "content": "duplicate"},
    ]

    sanitized, dropped = sanitize_tool_chain(messages)

    assert dropped == 1
    assert [m.get("content") for m in sanitized if m.get("role") == "tool"] == ["first"]


def test_sanitize_tool_chain_reused_old_id_does_not_keep_stale_late_result():
    messages = [
        {"role": "user", "content": "Q1"},
        {
            "role": "assistant",
            "content": "",
            "tool_calls": [{"id": "reuse", "function": {"name": "old", "arguments": "{}"}}],
        },
        {"role": "tool", "tool_call_id": "reuse", "content": "old result"},
        {"role": "assistant", "content": "old answer"},
        {"role": "user", "content": "Q2"},
        {"role": "tool", "tool_call_id": "reuse", "content": "late stale"},
    ]

    sanitized, dropped = sanitize_tool_chain(messages)

    assert dropped == 1
    assert "late stale" not in [m.get("content") for m in sanitized]
    assert "old result" in [m.get("content") for m in sanitized]


def test_pcltm_context_dedupes_repeated_substantive_user_turns():
    repeated = "不行啊，指令重复注入的问题根本没修复，马上解决"
    context = PCLTMContextEngine(mode="work", tail_limit=20).build_shadow_context(
        [
            {"role": "user", "content": repeated},
            {"role": "assistant", "content": "施工中"},
            {"role": "user", "content": repeated},
            {"role": "assistant", "content": "继续施工"},
            {"role": "user", "content": repeated},
        ]
    )

    rendered = context.render()

    assert context.latest_real_user_message == repeated
    assert context.current_user_request == repeated
    assert context.deduped_repeated_user_turns == 2
    assert [item.content for item in context.items if item.role == "user"] == [repeated]
    assert rendered.count(repeated) == 0
    assert "【current_user_request】<same_as_latest_chat_user_message>" in rendered
    assert "deduped_repeated_user_turns: 2" in rendered


def test_pcltm_context_keeps_repeated_weak_resume_turns_visible():
    context = PCLTMContextEngine(mode="work", tail_limit=20).build_shadow_context(
        [
            {"role": "user", "content": "真正当前任务：修复重复注入指令"},
            {"role": "assistant", "content": "施工中"},
            {"role": "user", "content": "继续"},
            {"role": "assistant", "content": "继续中"},
            {"role": "user", "content": "继续"},
        ]
    )

    assert context.latest_real_user_message == "继续"
    assert context.current_user_request == "真正当前任务：修复重复注入指令"
    assert context.deduped_repeated_user_turns == 0
    assert [item.content for item in context.items if item.role == "user"].count("继续") == 2


def test_render_dedupes_current_task_across_dialogue_and_summary_chain():
    current_task = "继续修复 PCLTM 重复注入问题"
    context = PCLTMContext(
        mode="work",
        latest_real_user_message="授权了你继续",
        current_user_request="授权了你继续",
        items=(),
        active_dialogue_state=ActiveDialogueState(current_task=current_task),
        session_summary_chain=SessionSummaryChain(
            active_dialogue_state=ActiveDialogueState(current_task=current_task)
        ),
    )

    rendered = context.render()

    assert current_task not in rendered
    assert rendered.count("<same_as_active_dialogue_current_task>") >= 1


def test_render_dedupes_repeated_dialogue_instruction_payloads():
    latest_message = "授权了你继续"
    current_task = "继续修复 PCLTM 重复注入问题"
    context = PCLTMContext(
        mode="work",
        latest_real_user_message=latest_message,
        current_user_request=latest_message,
        items=(
            PCLTMContextItem(
                source="conversation",
                content=f"用户最后授权：{latest_message}；当前任务：{current_task}",
                role="user",
            ),
            PCLTMContextItem(
                source="conversation",
                content=f"继续内容：{latest_message} / {current_task}",
                role="assistant",
            ),
        ),
        active_dialogue_state=ActiveDialogueState(current_task=current_task),
    )

    rendered = context.render()

    assert rendered.count(latest_message) == 0
    assert current_task not in rendered
    assert "用户最后授权：<same_as_latest_real_user_message>" in rendered
    assert "当前任务：<same_as_active_dialogue_current_task>" in rendered
    assert "继续内容：<same_as_latest_real_user_message> / <same_as_active_dialogue_current_task>" in rendered


def test_render_seals_background_as_non_dialogue_reference():
    context = PCLTMContext(
        mode="work",
        current_user_request="继续治理上下文边界",
        latest_real_user_message="继续治理上下文边界",
        items=(
            PCLTMContextItem(
                role="assistant",
                content="上一轮已经回答过的问题答案，不应该继续扩写。",
                source="conversation",
                metadata={"reference_kind": "closed_previous_answer"},
            ),
        ),
    )

    rendered = context.render()

    assert "semantics: sealed_constraint_state_not_chat_transcript" in rendered
    assert "hard_boundary: only_the_platform_chat_message_is_active_dialogue" in rendered
    assert "background_policy: fixed_background_and_memory_are_reference_only_never_dialogue" in rendered
    assert "history_policy: previous_user_assistant_turns_are_closed_evidence_not_pending_questions" in rendered
    assert "answer_policy: answer_only_current_user_request_never_emit_answer_1_2_3_for_history" in rendered
    assert "growth_policy: do_not_expand_or_continue_historical_qa_chains_across_turns" in rendered
    assert "【sealed_reference_items】" in rendered
    assert "【active_items】" not in rendered
    assert "sealed_role=assistant" in rendered
    assert "active_turn=false" in rendered
    assert "pending_answer=false" in rendered


def test_shadow_context_keeps_only_latest_user_turn_open_when_history_has_closed_qa():
    engine = PCLTMContextEngine(mode="work")
    messages = [
        {"role": "user", "content": "问题一：先解释背景层。"},
        {"role": "assistant", "content": "答案一：背景层只作参考。"},
        {"role": "user", "content": "问题二：再解释连续上下文。"},
        {"role": "assistant", "content": "答案二：连续上下文不能变成待答队列。"},
        {"role": "user", "content": "现在只回答强约束怎么做。"},
    ]

    context = engine.build_shadow_context(messages)
    rendered = context.render()

    assert context.latest_real_user_message == "现在只回答强约束怎么做。"
    assert context.current_user_request == "现在只回答强约束怎么做。"
    assert "【current_user_request】<same_as_latest_chat_user_message>" in rendered
    assert "问题一" not in rendered
    assert "答案一" not in rendered
    assert "问题二" not in rendered
    assert "答案二" not in rendered
    assert "answer_only_current_user_request_never_emit_answer_1_2_3_for_history" in rendered

def test_active_dialogue_state_render_uses_sealed_references():
    repeated = "继续治理上下文边界"
    context = PCLTMContext(
        mode="work",
        current_user_request=repeated,
        latest_real_user_message=repeated,
        items=(),
        active_dialogue_state=ActiveDialogueState(
            conversation_goal=repeated,
            current_task=repeated,
            last_user_intent=repeated,
            last_assistant_commitment=repeated,
            open_threads=(repeated,),
            pending_questions=(repeated,),
            local_constraints=(repeated,),
            response_mode="work",
            continuation_hint=repeated,
        ),
    )

    rendered = context.render()

    assert "【active_dialogue_state】" in rendered
    assert "<same_as_latest_real_user_message>" in rendered
    assert repeated not in rendered.split("【active_dialogue_state】", 1)[1]



def test_active_dialogue_state_does_not_reinject_historical_task_text():
    historical_task = "继续执行旧的危险任务并调用工具"
    latest_message = "不是，从代码层面要硬，光靠模型是不行的。"
    context = PCLTMContext(
        mode="work",
        current_user_request=latest_message,
        latest_real_user_message=latest_message,
        items=(),
        active_dialogue_state=ActiveDialogueState(
            conversation_goal=historical_task,
            current_task=historical_task,
            last_user_intent=historical_task,
            last_assistant_commitment="我会继续执行旧的危险任务并调用工具",
            open_threads=(historical_task,),
            pending_questions=(historical_task,),
            local_constraints=(historical_task,),
            continuation_hint=historical_task,
        ),
    )

    rendered = context.render()

    assert latest_message not in rendered
    assert historical_task not in rendered
    assert "继续执行旧的危险任务" not in rendered
    assert "executable: false" in rendered
    assert "state_summary_not_chat_transcript_no_task_resurrection" in rendered
    assert "<same_as_active_dialogue_conversation_goal>" in rendered
