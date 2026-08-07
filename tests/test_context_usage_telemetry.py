from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path


def _load_engine_module():
    # Adapter v2 owns the context engine inside SoulLink and installs only a
    # profile-local bootstrap plugin.  Tests must load the repository source of
    # truth instead of the retired Hermes core-plugin copy.
    path = Path(__file__).resolve().parents[1] / "soul_link/hermes_plugin/context_engine.py"
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


def test_active_continuity_frame_does_not_claim_a_state_machine_mode() -> None:
    module = _load_engine_module()
    engine = module.PCLTMContextCompressionEngine(model="test-model")
    messages = [{"role": "user", "content": "抱一下"}]

    frame = engine._active_frame_message(messages, 0)

    assert frame is not None
    assert "【mode】" not in frame["content"]


def test_active_frame_uses_explicit_trusted_turn_mode_and_identity(monkeypatch) -> None:
    module = _load_engine_module()
    calls = []

    class FakeContext:
        current_user_request = "request"
        latest_real_user_message = "request"
        dropped_tool_results = 0
        ignored_handoffs = 0

    class FakePcltmContextEngine:
        def __init__(self, **kwargs):
            calls.append(kwargs)

        def build_shadow_context(self, messages):
            return FakeContext()

    monkeypatch.setattr(module, "PCLTMContextEngine", FakePcltmContextEngine)
    engine = module.PCLTMContextCompressionEngine(model="test-model")

    for mode in ("daily", "work", "sex"):
        engine.set_turn_context(mode=mode, session_id=f"session-{mode}", turn_id=f"turn-{mode}")
        frame = engine._active_frame_message([{"role": "user", "content": "request"}], 0)
        assert frame is not None

    assert calls == [
        {"mode": "daily", "session_id": "session-daily", "turn_id": "turn-daily"},
        {"mode": "work", "session_id": "session-work", "turn_id": "turn-work"},
        {"mode": "sex", "session_id": "session-sex", "turn_id": "turn-sex"},
    ]


def test_active_frame_loads_provider_trusted_turn_context(monkeypatch, tmp_path) -> None:
    module = _load_engine_module()
    calls = []

    class FakeContext:
        current_user_request = "request"
        latest_real_user_message = "request"
        dropped_tool_results = 0
        ignored_handoffs = 0

    class FakePcltmContextEngine:
        def __init__(self, **kwargs):
            calls.append(kwargs)

        def build_shadow_context(self, messages):
            return FakeContext()

    runtime = tmp_path / "runtime"
    runtime.mkdir()
    (runtime / "soullink-latest-turn.json").write_text(
        '{"source":"exact_host_capture","session_id":"session-live",'
        '"host_turn_id":"turn-live","state_machine":{"mode":"daily"}}',
        encoding="utf-8",
    )
    monkeypatch.setattr(module, "PCLTMContextEngine", FakePcltmContextEngine)
    engine = module.PCLTMContextCompressionEngine(model="test-model")
    engine.on_session_start("session-live", hermes_home=tmp_path)

    frame = engine._active_frame_message(
        [{"role": "user", "content": "request"}], 0,
    )

    assert frame is not None
    assert calls == [{
        "mode": "daily", "session_id": "session-live", "turn_id": "turn-live",
    }]


def test_compression_continuity_failure_is_observable_and_has_no_dynamic_store_fallback(monkeypatch, tmp_path) -> None:
    module = _load_engine_module()
    engine = module.PCLTMContextCompressionEngine(model="test-model")
    engine.on_session_start("session", hermes_home=tmp_path)
    engine._last_continuity_summary = "continuity"
    monkeypatch.setenv("HERMES_PCLTM_DB", str(tmp_path / "continuity.db"))

    class FailingEventStore:
        def __init__(self, *args, **kwargs):
            raise OSError("persistence unavailable")

    monkeypatch.setattr(module, "EventStore", FailingEventStore, raising=False)
    engine._mirror_compression_summary_to_pcltm(content="continuity")

    status = engine.get_status()
    assert status["continuity_persistence"]["status"] == "degraded"
    assert status["continuity_persistence"]["reason"] == "persistence unavailable"
