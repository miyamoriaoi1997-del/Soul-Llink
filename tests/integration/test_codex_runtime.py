from __future__ import annotations

import json
from pathlib import Path

from soul_link.codex_hook import handle_hook
from soul_link.codex_mcp import runtime_status


def test_session_start_hook_returns_bounded_identity_context(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.setenv("CODEX_HOME", str(tmp_path / "codex"))
    result = handle_hook({"hook_event_name": "SessionStart", "session_id": "s1", "source": "startup"})
    assert result["hookSpecificOutput"]["hookEventName"] == "SessionStart"
    context = result["hookSpecificOutput"]["additionalContext"]
    assert "SoulLink/PCLTM" in context
    assert "# Core Identity Layer" in context
    assert "final-forward" in context
    assert len(context) <= 12000


def test_user_prompt_hook_retrieves_bounded_memory(monkeypatch) -> None:
    monkeypatch.setattr(
        "soul_link.codex_hook.search_archival_memories",
        lambda **kwargs: [{"memory_id": "db/MEMORY.md/1", "excerpt": "synthetic durable preference"}],
    )
    result = handle_hook({"hookEventName": "UserPromptSubmit", "session_id": "s1", "prompt": "preference"})
    context = result["hookSpecificOutput"]["additionalContext"]
    assert "synthetic durable preference" in context
    assert "typed background memory" in context


def test_non_injection_hook_is_audit_only(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.setenv("CODEX_HOME", str(tmp_path / "codex"))
    result = handle_hook({"hook_event_name": "Stop", "session_id": "s1"})
    assert result == {}
    lines = (tmp_path / "codex" / "soullink" / "hook-events.jsonl").read_text(encoding="utf-8").splitlines()
    assert json.loads(lines[-1])["event"] == "Stop"


def test_runtime_status_discloses_codex_observation_boundary(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.setenv("CODEX_HOME", str(tmp_path / "codex"))
    status = runtime_status()
    assert status["host"] == "codex"
    assert status["final_forward_observation"] == "unavailable_host_boundary"
