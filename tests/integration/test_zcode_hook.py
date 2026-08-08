from __future__ import annotations

import json
from pathlib import Path
from typing import ClassVar

import pytest

from soul_link.zcode_hook import handle_hook

ALL_EVENTS = {
    "SessionStart", "UserPromptSubmit", "PreToolUse", "PermissionRequest",
    "PostToolUse", "PostToolUseFailure", "Stop",
}


@pytest.fixture(autouse=True)
def _isolated_zcode_root(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Every hook test must write audits to a temp ZCode root, never the real user dir."""
    monkeypatch.setenv("ZCODE_ROOT", str(tmp_path / "zcode"))
    monkeypatch.setenv("HERMES_PCLTM_MEMFS_ROOT", str(tmp_path / "memfs"))
    monkeypatch.setenv("HERMES_PCLTM_DB", str(tmp_path / "pcltm.db"))


def test_session_start_injects_bounded_identity_context(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("ZCODE_ROOT", str(tmp_path / "zcode"))
    result = handle_hook({"hook_event_name": "SessionStart", "session_id": "s1", "source": "startup"})
    assert result["hookSpecificOutput"]["hookEventName"] == "SessionStart"
    context = result["hookSpecificOutput"]["additionalContext"]
    assert "SoulLink/PCLTM" in context
    assert "final-forward" in context
    assert len(context) <= 12000


def test_session_start_emotion_block_is_not_double_nested(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("ZCODE_ROOT", str(tmp_path / "zcode"))
    result = handle_hook({"hook_event_name": "SessionStart", "session_id": "s1", "source": "startup"})
    context = result["hookSpecificOutput"]["additionalContext"]
    # The persona engine already wraps the tone in <emotion_modifier> tags;
    # a second wrapper would double-nest the block.
    assert context.count("<emotion_modifier>") <= 1


def test_user_prompt_emotion_block_is_not_double_nested(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("ZCODE_ROOT", str(tmp_path / "zcode"))
    monkeypatch.setenv("HERMES_PCLTM_MEMFS_ROOT", str(tmp_path / "memfs"))
    monkeypatch.setenv("HERMES_PCLTM_DB", str(tmp_path / "pcltm.db"))
    result = handle_hook({
        "hook_event_name": "UserPromptSubmit", "session_id": "s1",
        "prompt": "你太厉害了，我爱你！",
    })
    context = result.get("hookSpecificOutput", {}).get("additionalContext", "")
    assert context.count("<emotion_modifier>") <= 1


class _FakeItem:
    claim_id = 1
    target = "memory"
    memory_type = "memory_note"
    sensitivity = "normal"
    content = "synthetic durable preference"


class _FakeResult:
    items: ClassVar[list[_FakeItem]] = [_FakeItem()]


class _FakeStore:
    def __init__(self, *args: object, **kwargs: object) -> None:
        pass

    def close(self) -> None:
        pass


def _patch_retrieval(monkeypatch: pytest.MonkeyPatch, result: object) -> None:
    monkeypatch.setattr(
        "pcltm.memory_retrieval.search_governed_memories",
        lambda store, request: result,
    )
    monkeypatch.setattr("pcltm.store.EventStore", _FakeStore)


def test_user_prompt_hook_retrieves_bounded_memory(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("ZCODE_ROOT", str(tmp_path / "zcode"))
    _patch_retrieval(monkeypatch, _FakeResult())
    result = handle_hook({"hookEventName": "UserPromptSubmit", "session_id": "s1", "prompt": "preference"})
    context = result["hookSpecificOutput"]["additionalContext"]
    assert "synthetic durable preference" in context
    assert "typed background memory" in context


def test_user_prompt_hook_closes_previous_tool_chain(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("ZCODE_ROOT", str(tmp_path / "zcode"))
    _patch_retrieval(monkeypatch, _FakeResult())
    handle_hook({"hookEventName": "UserPromptSubmit", "session_id": "s1", "prompt": "x"})
    records = [json.loads(line) for line in (tmp_path / "zcode" / "soullink" / "hook-events.jsonl").read_text(encoding="utf-8").splitlines() if line.strip()]
    assert records[-1]["event"] == "UserPromptSubmit"
    assert records[-1]["chain"] == "closed"


def test_user_prompt_hook_empty_prompt_is_audit_only(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("ZCODE_ROOT", str(tmp_path / "zcode"))
    result = handle_hook({"hook_event_name": "UserPromptSubmit", "session_id": "s1", "prompt": ""})
    assert result == {}
    assert (tmp_path / "zcode" / "soullink" / "hook-events.jsonl").is_file()


def test_pre_tool_use_denies_write_without_authorization(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("ZCODE_ROOT", str(tmp_path / "zcode"))
    monkeypatch.delenv("SOULLINK_ZCODE_ALLOW_MEMORY_WRITES", raising=False)
    result = handle_hook({
        "hook_event_name": "PreToolUse", "session_id": "s1",
        "tool_name": "soullink_memory_remember", "tool_use_id": "call_1",
    })
    output = result["hookSpecificOutput"]
    assert output["hookEventName"] == "PreToolUse"
    assert output["permissionDecision"] == "deny"


def test_pre_tool_use_allows_write_with_authorization(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("ZCODE_ROOT", str(tmp_path / "zcode"))
    monkeypatch.setenv("SOULLINK_ZCODE_ALLOW_MEMORY_WRITES", "1")
    result = handle_hook({
        "hook_event_name": "PreToolUse", "session_id": "s1",
        "tool_name": "soullink_memory_remember", "tool_use_id": "call_1",
    })
    assert result["hookSpecificOutput"]["permissionDecision"] == "allow"


def test_permission_request_decision_matches_authorization(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("ZCODE_ROOT", str(tmp_path / "zcode"))
    monkeypatch.setenv("SOULLINK_ZCODE_ALLOW_MEMORY_WRITES", "1")
    result = handle_hook({
        "hook_event_name": "PermissionRequest", "session_id": "s1",
        "tool_name": "soullink_memory_remember", "tool_use_id": "call_1",
    })
    assert result["hookSpecificOutput"]["decision"]["behavior"] == "allow"


def test_pre_tool_use_read_tools_always_pass(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("ZCODE_ROOT", str(tmp_path / "zcode"))
    monkeypatch.delenv("SOULLINK_ZCODE_ALLOW_MEMORY_WRITES", raising=False)
    result = handle_hook({
        "hook_event_name": "PreToolUse", "session_id": "s1",
        "tool_name": "soullink_memory_search", "tool_use_id": "call_1",
    })
    assert result["hookSpecificOutput"]["permissionDecision"] == "allow"


def test_permission_request_non_write_tool_is_untouched(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("ZCODE_ROOT", str(tmp_path / "zcode"))
    result = handle_hook({
        "hook_event_name": "PermissionRequest", "session_id": "s1",
        "tool_name": "Bash", "tool_use_id": "call_1",
    })
    assert result == {}


def test_post_tool_use_captures_evidence(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("ZCODE_ROOT", str(tmp_path / "zcode"))
    captured: dict[str, object] = {}

    def fake_write_evidence_capsule(**kwargs: object) -> dict:
        captured.update(kwargs)
        return {}

    monkeypatch.setattr("pcltm.memory_adapter.write_evidence_capsule", fake_write_evidence_capsule)
    result = handle_hook({
        "hook_event_name": "PostToolUse", "session_id": "s1",
        "tool_name": "Bash", "tool_use_id": "call_2",
        "tool_response": {"text": "exit 0"},
    })
    assert result == {}
    assert captured["source_tool"] == "Bash"
    assert captured["evidence_id"] == "call_2"


def test_post_tool_use_failure_closes_chain_and_is_audit_only(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("ZCODE_ROOT", str(tmp_path / "zcode"))
    result = handle_hook({
        "hook_event_name": "PostToolUseFailure", "session_id": "s1",
        "tool_name": "Bash", "error": {"message": "boom"},
    })
    assert result == {}
    records = [json.loads(line) for line in (tmp_path / "zcode" / "soullink" / "hook-events.jsonl").read_text(encoding="utf-8").splitlines() if line.strip()]
    assert records[-1]["event"] == "PostToolUseFailure"
    assert records[-1]["chain"] == "closed"


def test_stop_returns_empty_without_emotion_state(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("ZCODE_ROOT", str(tmp_path / "zcode"))
    result = handle_hook({"hook_event_name": "Stop", "session_id": "s1"})
    assert result == {}


def _drive_emotion_to_continue(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Feed strong positive turns so the persona emotion score crosses the
    continuation threshold, then return whether Stop requests continuation."""
    for _ in range(8):
        handle_hook({
            "hook_event_name": "UserPromptSubmit",
            "session_id": "s1",
            "prompt": "你太厉害了，我爱你，我永远都信任你！",
        })


def test_stop_continuation_requests_when_emotion_strong(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("ZCODE_ROOT", str(tmp_path / "zcode"))
    _drive_emotion_to_continue(tmp_path, monkeypatch)
    result = handle_hook({"hook_event_name": "Stop", "session_id": "s1"})
    assert result.get("continue") is True


def test_stop_continuation_is_bounded_to_three(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("ZCODE_ROOT", str(tmp_path / "zcode"))
    _drive_emotion_to_continue(tmp_path, monkeypatch)
    for _ in range(3):
        result = handle_hook({"hook_event_name": "Stop", "session_id": "s1"})
        assert result.get("continue") is True
    result = handle_hook({"hook_event_name": "Stop", "session_id": "s1"})
    assert result == {}


def test_unknown_event_is_audit_only_and_never_blocks(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("ZCODE_ROOT", str(tmp_path / "zcode"))
    result = handle_hook({"event": "SomethingElse", "session_id": "s1"})
    assert result == {}
