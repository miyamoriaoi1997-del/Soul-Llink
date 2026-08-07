from __future__ import annotations

import json
import os
import subprocess
import sys
import textwrap
from pathlib import Path


SOUL_ROOT = Path(__file__).resolve().parents[2]
PACKAGES_ROOT = SOUL_ROOT / "packages"
FORBIDDEN_HOST_MODULES = {
    "agent",
    "gateway",
    "hermes_state",
    "model_tools",
    "run_agent",
}


def _run_host_neutral_probe(source: str, tmp_path: Path) -> dict:
    env = os.environ.copy()
    env["PYTHONPATH"] = str(PACKAGES_ROOT)
    env.pop("HERMES_STATE_PATH", None)
    proc = subprocess.run(
        [sys.executable, "-c", textwrap.dedent(source)],
        cwd=tmp_path,
        env=env,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        check=True,
    )
    return json.loads(proc.stdout)


def test_pcltm_context_core_accepts_fake_host_messages_without_hermes_imports(tmp_path: Path) -> None:
    data = _run_host_neutral_probe(
        """
        import json
        import sys
        from pcltm.context_engine import PCLTMContextEngine

        before = set(sys.modules)
        context = PCLTMContextEngine(mode="work").build_shadow_context([
            {"role": "user", "content": "现在做 host-neutral acceptance"},
            {"role": "assistant", "content": "开始"},
            {"role": "tool", "tool_call_id": "orphan", "content": "stale tool result"},
            {
                "role": "user",
                "content": "[CONTEXT COMPACTION — REFERENCE ONLY]\\n## Current Active User Request\\n旧任务",
            },
            {"role": "user", "content": "继续"},
        ])
        imported = sorted(set(sys.modules) - before)
        print(json.dumps({
            "module": PCLTMContextEngine.__module__,
            "mode": context.mode,
            "latest_real_user_message": context.latest_real_user_message,
            "current_user_request": context.current_user_request,
            "ignored_handoffs": context.ignored_handoffs,
            "dropped_tool_results": context.dropped_tool_results,
            "host_modules": [name for name in imported if name.split(".", 1)[0] in %r],
        }, ensure_ascii=False))
        """ % sorted(FORBIDDEN_HOST_MODULES),
        tmp_path,
    )

    assert data["module"] == "pcltm.context_engine"
    assert data["mode"] == "work"
    assert data["latest_real_user_message"] == "继续"
    assert data["current_user_request"] == "现在做 host-neutral acceptance"
    assert data["ignored_handoffs"] == 1
    assert data["dropped_tool_results"] == 1
    assert data["host_modules"] == []


def test_soul_link_facade_composes_fake_host_prompt_without_hermes_runtime(tmp_path: Path) -> None:
    data = _run_host_neutral_probe(
        """
        import json
        import sys
        from pathlib import Path
        from soul_link import SoulLink

        state_path = Path("fake-host-state.md").resolve()
        before = set(sys.modules)
        soul_link = SoulLink(
            state_path=state_path,
            enable_context_router=False,
        )
        request = soul_link.ingest(
            "继续验证 host-neutral acceptance",
            recent_context=[{"role": "user", "content": "上一轮：分离验收"}],
            previous_mode="work",
            platform="fake-host",
        )
        resolution = soul_link.resolve(request, host_system_prompt="Fake host system prompt")
        imported = sorted(set(sys.modules) - before)
        prompt = resolution.prompt_candidate or {}
        print(json.dumps({
            "mode": resolution.mode,
            "selected_layers": resolution.selected_layers,
            "route_bucket": resolution.route_bucket,
            "prompt_has_host": "Fake host system prompt" in prompt.get("prompt_text", ""),
            "prompt_hash": str(prompt.get("prompt_hash", "")),
            "state_exists": state_path.exists(),
            "platform": resolution.audit_packet["request"]["platform"],
            "host_modules": [name for name in imported if name.split(".", 1)[0] in %r],
        }, ensure_ascii=False))
        """ % sorted(FORBIDDEN_HOST_MODULES),
        tmp_path,
    )

    assert data["mode"] == "work"
    assert data["selected_layers"] == ["core", "work"]
    assert isinstance(data["route_bucket"], str) and data["route_bucket"]
    assert data["prompt_has_host"] is True
    assert len(data["prompt_hash"]) >= 12
    assert all(char in "0123456789abcdef" for char in data["prompt_hash"])
    assert data["state_exists"] is False
    assert data["platform"] == "fake-host"
    assert data["host_modules"] == []
