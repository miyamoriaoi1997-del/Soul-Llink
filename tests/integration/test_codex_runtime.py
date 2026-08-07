from __future__ import annotations

import asyncio
import json
import os
import sys
from pathlib import Path

from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client

from soul_link.codex_hook import handle_hook
from soul_link.codex_mcp import runtime_status, soullink_memory_remember


def test_session_start_hook_returns_bounded_identity_context(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.setenv("CODEX_HOME", str(tmp_path / "codex"))
    result = handle_hook({"hook_event_name": "SessionStart", "session_id": "s1", "source": "startup"})
    assert result["hookSpecificOutput"]["hookEventName"] == "SessionStart"
    context = result["hookSpecificOutput"]["additionalContext"]
    assert "SoulLink/PCLTM" in context
    assert "final-forward" in context
    assert "<soullink_identity>" not in context
    assert "persona_engine" not in context
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


def test_memory_write_is_disabled_without_explicit_authorization(monkeypatch) -> None:
    monkeypatch.delenv("SOULLINK_CODEX_ALLOW_MEMORY_WRITES", raising=False)
    monkeypatch.setattr("soul_link.codex_mcp.sync_memory_tool_write", lambda **kwargs: (_ for _ in ()).throw(AssertionError("write reached")))
    assert soullink_memory_remember("synthetic") == {
        "success": False,
        "error": "write_disabled",
        "target": "memory",
    }


def test_memory_write_requires_explicit_authorization(monkeypatch) -> None:
    monkeypatch.setenv("SOULLINK_CODEX_ALLOW_MEMORY_WRITES", "1")
    monkeypatch.setattr("soul_link.codex_mcp.sync_memory_tool_write", lambda **kwargs: True)
    assert soullink_memory_remember("synthetic", "user") == {"success": True, "target": "user"}


def test_stdio_mcp_server_starts_and_reports_boundary(tmp_path: Path) -> None:
    async def probe() -> None:
        env = dict(os.environ)
        env.update({
            "CODEX_HOME": str(tmp_path / "codex"),
            "HERMES_PCLTM_DB": str(tmp_path / "runtime" / "pcltm.db"),
            "HERMES_PCLTM_MEMFS_ROOT": str(tmp_path / "runtime" / "memfs"),
        })
        server = StdioServerParameters(
            command=sys.executable,
            args=["-m", "soul_link.codex_mcp"],
            env=env,
            cwd=str(Path(__file__).resolve().parents[2]),
        )
        async with stdio_client(server) as (read, write):
            async with ClientSession(read, write) as session:
                await session.initialize()
                tools = {tool.name for tool in (await session.list_tools()).tools}
                assert "soullink_runtime_status" in tools
                result = await session.call_tool("soullink_runtime_status", {})
                payload = json.loads(result.content[0].text)
                assert payload["host"] == "codex"
                assert payload["final_forward_observation"] == "unavailable_host_boundary"

    asyncio.run(probe())
