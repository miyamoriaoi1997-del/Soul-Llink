from __future__ import annotations

import importlib.util
import json
from pathlib import Path


def _load_tool_capsules_module():
    path = Path(__file__).resolve().parents[1] / "soul_link/hermes_plugin/tool_capsules.py"
    spec = importlib.util.spec_from_file_location("pcltm_tool_capsules_test", path)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_render_terminal_capsule_redacts_secrets_and_preserves_completion() -> None:
    module = _load_tool_capsules_module()
    raw = {
        "output": "authorization: Bearer abcdefghijklmnopqrstuvwxyz123456\n"
        + ("progress\n" * 300)
        + "RESULT: 37 passed\n",
        "exit_code": 0,
        "error": None,
    }

    capsule = module.render_tool_capsule(
        tool_name="terminal",
        tool_call_id="call-test",
        text=json.dumps(raw),
        raw_content=raw,
        kind="terminal",
        preserve_edges=True,
    )

    assert "abcdefghijklmnopqrstuvwxyz123456" not in capsule
    assert "RESULT: 37 passed" in capsule
    assert '"exit_code": 0' in capsule
    assert "capsule_kind=terminal" in capsule
