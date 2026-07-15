from __future__ import annotations

import importlib.util
import json
import os
import sys
from pathlib import Path

import pytest


def _load_engine_module():
    host_root = os.environ.get("SOULLINK_HERMES_AGENT_ROOT")
    if not host_root:
        pytest.skip("set SOULLINK_HERMES_AGENT_ROOT to run installed-host telemetry tests")
    path = Path(host_root) / "plugins/context_engine/pcltm-context/__init__.py"
    if not path.is_file():
        pytest.skip(f"Hermes context-engine plugin is not installed under {host_root}")
    name = "pcltm_context_runtime_telemetry_test"
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def test_update_from_response_writes_exact_host_context_usage(tmp_path) -> None:
    module = _load_engine_module()
    engine = module.PCLTMContextCompressionEngine(model="test-model")
    engine.configure({"budget_tokens": 256_000})
    engine.on_session_start("session-live", hermes_home=tmp_path)

    engine.update_from_response(
        {
            "prompt_tokens": 64_000,
            "completion_tokens": 1_200,
            "total_tokens": 65_200,
            "input_tokens": 64_000,
            "output_tokens": 1_200,
            "cache_read_tokens": 48_000,
            "cache_write_tokens": 20,
            "reasoning_tokens": 300,
        }
    )

    path = tmp_path / "runtime/soullink-context-telemetry.json"
    payload = json.loads(path.read_text(encoding="utf-8"))
    assert payload["source"] == "exact_host_context_usage"
    assert payload["session_id"] == "session-live"
    assert payload["engine"] == "pcltm-context"
    assert payload["prompt_tokens"] == 64_000
    assert payload["completion_tokens"] == 1_200
    assert payload["total_tokens"] == 65_200
    assert payload["budget_tokens"] == 256_000
    assert payload["cache_read_tokens"] == 48_000
    assert payload["observed_at"]
