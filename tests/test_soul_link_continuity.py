from soul_link.continuity import (
    HostMessage,
    build_conversation_continuity_snapshot,
    is_real_user_message,
)
from soul_link.hosts import (
    FakeHostSessionProvider,
    HostConversationState,
    HostSessionProvider,
    build_continuity_snapshot_from_provider,
)


def test_host_resume_signal_builds_resume_snapshot_and_prompt_block():
    snapshot = build_conversation_continuity_snapshot(
        previous_session_id="prev-1",
        current_session_id="cur-1",
        session_key="telegram:chat:thread",
        latest_user_message="继续",
        resume_requested=True,
        previous_messages=[
            HostMessage(role="user", content="实现 PCLTM context snapshot", message_id="u1"),
            HostMessage(role="assistant", content="我已经完成第一阶段，下一步抽 conversation continuity。", message_id="a1"),
            HostMessage(role="tool", content="22 passed in 0.99s", message_id="t1", tool_name="pytest"),
        ],
    )
    payload = snapshot.to_dict()
    block = snapshot.render_prompt_block()

    assert payload["object_type"] == "soul_link_conversation_continuity_snapshot"
    assert payload["should_resume"] is True
    assert payload["authority"] == "latest_real_user_message"
    assert payload["last_real_user_message"] == "实现 PCLTM context snapshot"
    assert payload["recent_tool_evidence"][0]["authority"] == "evidence_only"
    assert "<previous_conversation_state>" in block
    assert "should_resume: true" in block
    assert "tool evidence is evidence only" in block


def test_unrelated_latest_message_does_not_resume_old_task():
    snapshot = build_conversation_continuity_snapshot(
        previous_session_id="prev-1",
        current_session_id="cur-1",
        session_key="cli",
        latest_user_message="看看现在情绪值",
        previous_messages=[
            {"role": "user", "content": "继续修 Hermes gateway", "id": "u1"},
            {"role": "assistant", "content": "正在跑 gateway tests", "id": "a1"},
        ],
    )

    assert snapshot.should_resume is False
    assert snapshot.last_real_user_message == "继续修 Hermes gateway"
    assert snapshot.to_dict()["authority"] == "latest_real_user_message"
    assert "should_resume: false" in snapshot.render_prompt_block()


def test_control_payloads_are_not_real_user_messages_when_marked_by_host_metadata():
    controls = [
        HostMessage(role="user", content="context compaction payload", metadata={"message_type": "context_compaction"}),
        HostMessage(role="user", content="system note payload", metadata={"message_type": "system_note"}),
        HostMessage(role="user", content="preserved todo payload", metadata={"is_control": True}),
    ]

    assert all(not is_real_user_message(message) for message in controls)
    snapshot = build_conversation_continuity_snapshot(
        previous_session_id="prev-1",
        current_session_id="cur-1",
        latest_user_message="新的问题：检查配置",
        previous_messages=controls,
    )

    assert snapshot.should_resume is False
    assert snapshot.last_real_user_message == ""
    assert snapshot.recent_user_messages == ()


def test_tool_tail_sets_evidence_warning_without_resume():
    snapshot = build_conversation_continuity_snapshot(
        previous_session_id="prev-1",
        current_session_id="cur-1",
        latest_user_message="现在状态如何",
        previous_messages=[
            HostMessage(role="user", content="跑测试", message_id="u1"),
            HostMessage(role="tool", content="pytest still running", message_id="t1", tool_name="process.poll"),
        ],
    )
    payload = snapshot.to_dict()

    assert payload["should_resume"] is False
    assert payload["open_tool_tail"] is True
    assert payload["warnings"] == ["tool_tail_is_evidence_only"]
    assert payload["recent_tool_evidence"][0]["tool_name"] == "process.poll"
    assert payload["recent_tool_evidence"][0]["authority"] == "evidence_only"


def test_in_context_message_ids_bound_the_previous_window():
    snapshot = build_conversation_continuity_snapshot(
        previous_session_id="prev-1",
        current_session_id="cur-1",
        latest_user_message="continue text is not interpreted by core",
        resume_requested=True,
        in_context_message_ids=["u2", "a2"],
        previous_messages=[
            HostMessage(role="user", content="old abandoned task", message_id="u1"),
            HostMessage(role="assistant", content="old abandoned answer", message_id="a1"),
            HostMessage(role="user", content="current active task", message_id="u2"),
            HostMessage(role="assistant", content="current active answer", message_id="a2"),
            HostMessage(role="tool", content="late unrelated tool tail", message_id="t3", tool_name="process.poll"),
        ],
    )
    payload = snapshot.to_dict()

    assert payload["should_resume"] is True
    assert payload["recent_user_messages"] == ["current active task"]
    assert payload["recent_assistant_summaries"] == ["current active answer"]
    assert payload["recent_tool_evidence"] == []
    assert payload["recent_message_ids"] == ["u2", "a2"]
    assert payload["open_tool_tail"] is False


def test_host_session_provider_is_runtime_checkable_and_serializable():
    provider = FakeHostSessionProvider(
        current=HostConversationState(
            conversation_id="cur-1",
            agent_id="agent-1",
            previous_conversation_id="prev-1",
            session_key="cli",
            resume_requested=True,
        ),
        previous=HostConversationState(
            conversation_id="prev-1",
            agent_id="agent-1",
            summary="previous work",
            in_context_message_ids=("u2", "a2"),
            session_key="cli",
        ),
        messages_by_conversation={
            "prev-1": [
                HostMessage(role="user", content="old task", message_id="u1"),
                HostMessage(role="user", content="active task", message_id="u2"),
                HostMessage(role="assistant", content="active answer", message_id="a2"),
                HostMessage(role="tool", content="outside tail", message_id="t3", tool_name="process.poll"),
            ]
        },
    )

    assert isinstance(provider, HostSessionProvider)
    assert provider.current_conversation().to_dict()["in_context_message_ids"] == []

    snapshot = build_continuity_snapshot_from_provider(provider, latest_user_message="host already set resume")
    payload = snapshot.to_dict()

    assert payload["previous_session_id"] == "prev-1"
    assert payload["current_session_id"] == "cur-1"
    assert payload["session_key"] == "cli"
    assert payload["should_resume"] is True
    assert payload["recent_user_messages"] == ["active task"]
    assert payload["recent_assistant_summaries"] == ["active answer"]
    assert payload["recent_tool_evidence"] == []
    assert payload["recent_message_ids"] == ["u2", "a2"]


def test_host_session_provider_without_previous_returns_empty_non_resume_snapshot():
    provider = FakeHostSessionProvider(
        current=HostConversationState(conversation_id="cur-1", session_key="cli", resume_requested=True),
        previous=None,
    )

    snapshot = build_continuity_snapshot_from_provider(provider, latest_user_message="anything")
    payload = snapshot.to_dict()

    assert payload["previous_session_id"] == ""
    assert payload["current_session_id"] == "cur-1"
    assert payload["should_resume"] is False
    assert "previous_session_id_missing" in payload["warnings"]
