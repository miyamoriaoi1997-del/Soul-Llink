from pcltm.context_engine import PCLTMContextEngine, PCLTMContextItem
from pcltm.state import (
    DialogueTurn,
    SessionSummaryChain,
    append_turns_to_chain,
    build_session_summary_chain,
)
from pcltm.session.segmenter import segment_turns
import pytest


def test_chain_is_built_from_raw_turn_segments_not_prior_summary_text():
    raw_turns = [
        DialogueTurn(user="阶段 2：实现 Session Summary Chain。", assistant="我会按 raw turns 分段。"),
        DialogueTurn(user="不要做多次压缩继承。", assistant="我会避免 summary-of-summary。"),
        DialogueTurn(user="继续。", assistant="继续当前阶段 2。"),
    ]
    old_chain = SessionSummaryChain.from_dict(
        {
            "segments": [
                {
                    "segment_id": 1,
                    "start_turn": 1,
                    "end_turn": 1,
                    "current_task": "错误旧摘要：改长期记忆。",
                    "evidence": ["SHOULD_NOT_SURVIVE"],
                }
            ],
            "source_turn_count": 1,
        }
    )

    chain = append_turns_to_chain(old_chain, raw_turns, segment_size=2, max_segments=4)
    rendered = chain.render()

    assert "SHOULD_NOT_SURVIVE" not in rendered
    assert "错误旧摘要" not in rendered
    assert "Session Summary Chain" in rendered
    assert "raw_turn_segments_not_summary_of_summaries" in rendered


def test_chain_keeps_current_task_across_long_session_segments():
    turns = [DialogueTurn(user="主任务：实现长会话主线保持。", assistant="我会分段记录主线。")]
    for index in range(1, 12):
        turns.append(
            DialogueTurn(
                user=f"继续第 {index} 步。",
                assistant=f"第 {index} 步完成后继续主任务。",
            )
        )

    chain = build_session_summary_chain(turns, segment_size=3, max_segments=6)

    assert chain.source_turn_count == 12
    assert len(chain.segments) == 4
    assert chain.current_task == "主任务：实现长会话主线保持。"
    assert chain.active_dialogue_state is not None
    assert chain.active_dialogue_state.current_task == "主任务：实现长会话主线保持。"


def test_chain_tracks_side_threads_without_stealing_main_task():
    turns = [
        DialogueTurn(user="主任务：修复会话变长后的主线保持。", assistant="我会建立链。"),
        DialogueTurn(user="顺便问一句，pytest 怎么跑？", assistant="用 uv run pytest。"),
        DialogueTurn(user="继续刚才那个。", assistant="继续修主线保持。"),
    ]

    chain = build_session_summary_chain(turns, segment_size=2)

    assert chain.current_task == "主任务：修复会话变长后的主线保持。"
    assert chain.active_dialogue_state is not None
    assert chain.active_dialogue_state.current_task == "主任务：修复会话变长后的主线保持。"
    assert any("pytest" in thread for segment in chain.segments for thread in segment.open_threads)


def test_context_engine_injects_ads_before_chain_before_memory_items():
    context = PCLTMContextEngine(mode="work").build_shadow_context(
        [
            {"role": "user", "content": "主任务：阶段 2 Session Summary Chain。"},
            {"role": "assistant", "content": "我会按原始 turns 建链。"},
            {"role": "user", "content": "继续。"},
        ]
    )
    memory_item = PCLTMContextItem(
        role="system",
        content="长期记忆：当前任务是另一个项目。",
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
        active_dialogue_state=context.active_dialogue_state,
        session_summary_chain=context.session_summary_chain,
    )

    rendered = context.render()

    assert context.session_summary_chain is not None
    assert rendered.index("【active_dialogue_state】") < rendered.index("【session_summary_chain】")
    assert rendered.index("【session_summary_chain】") < rendered.index("长期记忆：当前任务是另一个项目。")
    assert "current_task: <same_as_current_user_request>" in rendered
    assert "【current_user_request】<background_active_task_not_new_user_message> 主任务：阶段 2 Session Summary Chain。" in rendered


def test_bounded_chain_drops_old_segments_but_keeps_latest_raw_derived_state():
    turns = [DialogueTurn(user=f"任务 {index}", assistant=f"处理 {index}") for index in range(10)]

    chain = build_session_summary_chain(turns, segment_size=2, max_segments=3)

    assert len(chain.segments) == 3
    assert chain.segments[0].start_turn == 4
    assert chain.segments[-1].end_turn == 10
    assert chain.source_turn_count == 10
    assert chain.current_task == "任务 9"


def test_chain_rejects_non_positive_max_segments():
    turns = [DialogueTurn(user="task", assistant="done")]

    for value in (0, -1):
        with pytest.raises(ValueError, match="max_segments must be positive"):
            build_session_summary_chain(turns, max_segments=value)


def test_segmenter_rejects_single_turn_larger_than_token_limit():
    turns = [DialogueTurn(user="x" * 100, assistant="")]

    with pytest.raises(ValueError, match="single turn exceeds"):
        segment_turns(turns, max_tokens_per_segment=10)


def test_scope_correction_revokes_prior_extension_across_segments_and_resume():
    turns = [
        DialogueTurn(user="实现现有 JSONL 审计器。", assistant="我会按现有契约实现。"),
        DialogueTurn(user="再新增一个批处理扩展。", assistant="我会增加批处理扩展。"),
        DialogueTurn(
            user="取消批处理扩展，只修现有审计器，不要增加新功能。",
            assistant="我会删除扩展并回到现有审计器范围。",
        ),
        DialogueTurn(user="继续。", assistant="继续修现有审计器。"),
    ]

    chain = build_session_summary_chain(turns, segment_size=2, max_segments=4)

    assert "再新增一个批处理扩展。" in chain.revoked_index
    assert "再新增一个批处理扩展。" not in chain.unresolved_index
    assert chain.current_task == "取消批处理扩展，只修现有审计器，不要增加新功能。"


def test_negative_constraint_is_not_misclassified_as_revoked_work():
    turns = [
        DialogueTurn(user="实现现有审计器。", assistant="继续实现。"),
        DialogueTurn(user="继续，但不要增加新功能。", assistant="只处理现有范围。"),
    ]

    chain = build_session_summary_chain(turns, segment_size=1)

    assert chain.revoked_index == ()
    assert chain.segments[-1].revoked_items == ()
    assert "不要增加新功能" in " ".join(chain.segments[-1].local_constraints)


def test_bounded_chain_retains_revoke_event_at_first_kept_segment():
    turns = [
        DialogueTurn(user="实现现有审计器。", assistant="开始实现。"),
        DialogueTurn(user="新增批处理扩展。", assistant="开始扩展。"),
        DialogueTurn(user="取消批处理扩展，只修现有审计器。", assistant="删除扩展。"),
        DialogueTurn(user="继续。", assistant="继续现有审计器。"),
    ]

    chain = build_session_summary_chain(turns, segment_size=1, max_segments=2)

    assert len(chain.segments) == 2
    assert chain.segments[0].start_turn == 2
    assert "新增批处理扩展。" in chain.revoked_index
    assert "新增批处理扩展。" not in chain.unresolved_index
