from __future__ import annotations

import importlib
import json
import sqlite3
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest


def _load_provider_module():
    return importlib.import_module("soul_link.hermes_plugin.memory_provider")


def _load_provider_class():
    return _load_provider_module().SoulLinkMemoryProvider


def test_state_machine_runtime_config_reads_explicit_boolean_gates(tmp_path, monkeypatch):
    module = _load_provider_module()
    (tmp_path / "config.yaml").write_text(
        "plugins:\n"
        "  entries:\n"
        "    soullink:\n"
        "      state_machine:\n"
        "        transition_table_shadow: true\n"
        "        bounded_activation: false\n",
        encoding="utf-8",
    )
    monkeypatch.setattr(module, "_hermes_home", lambda: tmp_path)

    assert module._state_machine_runtime_config() == {
        "transition_table_shadow": True,
        "bounded_activation": False,
        "semantic_shadow": False,
        "semantic_authority": False,
        "semantic_backend": "local",
    }


def test_state_machine_runtime_config_ignores_retired_protected_authority(tmp_path, monkeypatch):
    module = _load_provider_module()
    (tmp_path / "config.yaml").write_text(
        "plugins:\n  entries:\n    soullink:\n      state_machine:\n        protected_authority: experimental\n",
        encoding="utf-8",
    )
    monkeypatch.setattr(module, "_hermes_home", lambda: tmp_path)

    assert "protected_authority" not in module._state_machine_runtime_config()


@pytest.mark.parametrize(
    "config_text",
    (
        "plugins: [invalid-shape]\n",
        "plugins:\n  entries:\n    soullink:\n      state_machine: invalid\n",
        "plugins:\n  entries:\n    soullink:\n      state_machine: 1\n",
    ),
)
def test_state_machine_runtime_config_fails_closed(tmp_path, monkeypatch, config_text):
    module = _load_provider_module()
    (tmp_path / "config.yaml").write_text(config_text, encoding="utf-8")
    monkeypatch.setattr(module, "_hermes_home", lambda: tmp_path)

    assert module._state_machine_runtime_config() == {
        "transition_table_shadow": False,
        "bounded_activation": False,
        "semantic_shadow": False,
        "semantic_authority": False,
        "semantic_backend": "local",
    }


def test_state_orchestrator_applies_bounded_activation_from_host_config(tmp_path, monkeypatch):
    module = _load_provider_module()
    (tmp_path / "config.yaml").write_text(
        "plugins:\n"
        "  entries:\n"
        "    soullink:\n"
        "      state_machine:\n"
        "        bounded_activation: true\n",
        encoding="utf-8",
    )
    monkeypatch.setattr(module, "_hermes_home", lambda: tmp_path)
    transitions = SimpleNamespace(
        enable_shadow=False,
        enable_bounded_activation=False,
        shadow_table=object(),
    )
    provider = module.SoulLinkMemoryProvider()
    provider._state_orchestrator_factory = lambda *args, **kwargs: SimpleNamespace(
        transitions=transitions
    )

    provider._get_state_orchestrator()

    assert transitions.enable_bounded_activation is True
    assert transitions.enable_shadow is True


def test_inprocess_router_skips_when_profile_is_not_configured(tmp_path, monkeypatch):
    module = _load_provider_module()
    monkeypatch.setattr(module, "_hermes_home", lambda: tmp_path)

    assert module.ensure_inprocess_model_router() == {"enabled": False, "running": False}


def test_inprocess_router_starts_once_inside_hermes_process(tmp_path, monkeypatch):
    module = _load_provider_module()
    (tmp_path / "config.yaml").write_text(
        "model:\n  base_url: http://127.0.0.1:18080/v1\n  default: persona-auto\n",
        encoding="utf-8",
    )
    monkeypatch.setattr(module, "_hermes_home", lambda: tmp_path)
    monkeypatch.setattr(module, "_ensure_paths", lambda: tmp_path)
    module._ROUTER_SERVER = None
    module._ROUTER_THREAD = None

    class FakeThread:
        def __init__(self, **kwargs):
            self.started = False
        def start(self):
            self.started = True
        def is_alive(self):
            return self.started

    class FakeServer:
        def __init__(self, address, handler, cfg):
            self.address = address
        def serve_forever(self):
            pass

    fake_app = SimpleNamespace(
        Handler=object,
        RouterConfig=lambda path: SimpleNamespace(listen_host="127.0.0.1", listen_port=18080),
        RouterServer=FakeServer,
    )
    monkeypatch.setitem(sys.modules, "model_router.app", fake_app)
    monkeypatch.setattr(module.threading, "Thread", FakeThread)

    first = module.ensure_inprocess_model_router()
    second = module.ensure_inprocess_model_router()

    assert first == {"enabled": True, "running": True, "owner": "hermes_process"}
    assert second == first


@pytest.fixture
def provider_factory(tmp_path, monkeypatch):
    provider_class = _load_provider_class()

    def create():
        provider = provider_class()
        monkeypatch.setattr(provider, "_runtime_capture_path", tmp_path / "latest-turn.json")
        monkeypatch.setattr(provider, "_state_orchestrator_factory", FakeStateOrchestrator)
        monkeypatch.setattr(provider, "_read_soul_mode_layer", lambda mode: f"# {mode} layer")
        return provider

    return create


class FakeEmotionManager:
    updates: list[list[dict]] = []

    def __init__(self, *args, **kwargs):
        pass

    def update_emotion_state(self, messages):
        self.updates.append(messages)
        return True

    def get_current_emotion_state(self):
        return {
            "affection": 88,
            "trust": 77,
            "possessiveness": 66,
            "patience": 55,
            "emotion_score": 1.75,
            "current_emotion": 2.05,
            "last_trigger_type": "recognition",
        }

    def get_tone_modifiers(self):
        return "<emotion_modifier>updated-before-reply</emotion_modifier>"


class FakeStateOrchestrator:
    calls = []

    def __init__(self, *args, **kwargs):
        pass

    def analyze_turn(self, **kwargs):
        self.calls.append(kwargs)
        return SimpleNamespace(
            mode="work",
            transition="daily_to_work",
            confidence=0.97,
            selected_layers=["core", "work"],
            safety_flags=[],
            desire_tier="restrained",
            semantic_shadow={
                "primary_mode": "work",
                "confidence": 0.91,
                "backend": "rules+local-lightweight",
                "shadow_only": True,
            },
            route_metadata={
                "reason_codes": ["work_override"],
                "hermes_route_bucket": "task",
                "hermes_model_hint": "technical",
                "hermes_selected_model": "work-model",
                "decision_audit": {
                    "transition": {"authority_source": "new_table"},
                },
            },
        )


def test_turn_start_updates_before_prefetch_and_prefetch_injects_updated_emotion(provider_factory, monkeypatch):
    provider = provider_factory()
    FakeEmotionManager.updates = []
    monkeypatch.setattr(provider, "_emotion_manager_factory", FakeEmotionManager)
    monkeypatch.setattr(provider, "_load_memory_context", lambda **kwargs: "<pcltm_context>memory</pcltm_context>")

    provider.on_turn_start(7, "老师消息", session_id="s1")
    context = provider.prefetch("老师消息", session_id="s1")

    assert FakeEmotionManager.updates == [[{"role": "user", "content": "老师消息"}]]
    assert "<soullink_turn_state>" in context
    assert "affection: 88" in context
    assert "current_emotion: 2.05" in context
    assert "updated-before-reply" in context
    assert context.index("<pcltm_context>") < context.index("<soullink_turn_state>")


def test_production_plugin_entry_keeps_memory_then_mode_then_emotion(monkeypatch):
    from soul_link.hermes_plugin.memory_provider import SoulLinkMemoryProvider

    provider = SoulLinkMemoryProvider()
    monkeypatch.setattr(provider, "_emotion_manager_factory", FakeEmotionManager)
    monkeypatch.setattr(provider, "_state_orchestrator_factory", FakeStateOrchestrator)
    monkeypatch.setattr(provider, "_read_soul_mode_layer", lambda mode: f"# {mode} layer exact body")
    monkeypatch.setattr(provider, "_load_memory_context", lambda **kwargs: "<pcltm_context>memory</pcltm_context>")

    provider.on_turn_start(8, "老师消息", session_id="production-entry")
    context = provider.prefetch("老师消息", session_id="production-entry")

    assert context.index("<pcltm_context>") < context.index("<state_machine_injection>")
    assert context.index("<state_machine_injection>") < context.index("<soullink_turn_state>")
    assert context.rstrip().endswith("</soullink_turn_state>")


def test_production_plugin_entry_emits_same_correlated_active_route(monkeypatch, tmp_path):
    from soul_link.hermes_plugin.memory_provider import SoulLinkMemoryProvider

    provider = SoulLinkMemoryProvider()
    monkeypatch.setattr(provider, "_emotion_manager_factory", FakeEmotionManager)
    monkeypatch.setattr(provider, "_state_orchestrator_factory", FakeStateOrchestrator)
    monkeypatch.setattr(provider, "_runtime_capture_path", tmp_path / "latest-turn.json")
    monkeypatch.setattr(provider, "_read_soul_mode_layer", lambda mode: f"# {mode}")

    provider.on_turn_start(4, "检查生产链", session_id="production-entry")

    metadata = provider.request_overrides()["extra_body"]["metadata"]
    capture = json.loads((tmp_path / "latest-turn.json").read_text(encoding="utf-8"))
    assert metadata["hermes_turn_correlation_id"] == capture["turn_correlation_id"]
    assert FakeStateOrchestrator.calls[-1]["runtime_authority"] == "active"


def test_turn_start_forwards_bounded_recent_messages_to_state_machine(provider_factory, monkeypatch):
    provider = provider_factory()
    FakeStateOrchestrator.calls = []
    monkeypatch.setattr(provider, "_emotion_manager_factory", FakeEmotionManager)
    monkeypatch.setattr(provider, "_state_orchestrator_factory", FakeStateOrchestrator)
    monkeypatch.setattr(provider, "_read_soul_mode_layer", lambda mode: f"# {mode}")
    recent_messages = [
        {"role": "user", "content": "上一轮任务"},
        {"role": "assistant", "content": "已经处理"},
    ]

    provider.on_turn_start(
        5,
        "继续",
        session_id="s1",
        recent_messages=recent_messages,
    )

    assert FakeStateOrchestrator.calls[-1]["recent_messages"] == recent_messages


def test_request_overrides_expose_only_current_state_machine_route(provider_factory, monkeypatch):
    provider = provider_factory()
    monkeypatch.setattr(provider, "_emotion_manager_factory", FakeEmotionManager)
    monkeypatch.setattr(provider, "_state_orchestrator_factory", FakeStateOrchestrator)
    monkeypatch.setattr(provider, "_read_soul_mode_layer", lambda mode: f"# {mode}")

    assert provider.request_overrides() == {}

    provider.on_turn_start(1, "处理任务", session_id="s1")

    overrides = provider.request_overrides()
    assert overrides == {
        "extra_body": {
            "metadata": {
                "hermes_route_bucket": "task",
                "hermes_turn_correlation_id": overrides["extra_body"]["metadata"]["hermes_turn_correlation_id"],
            }
        }
    }
    correlation_id = overrides["extra_body"]["metadata"]["hermes_turn_correlation_id"]
    assert len(correlation_id) == 24
    assert correlation_id == provider._runtime_capture_payload["turn_correlation_id"]


def test_failed_new_turn_clears_previous_route_before_raising(provider_factory, monkeypatch):
    provider = provider_factory()
    manager = FakeEmotionManager()
    monkeypatch.setattr(provider, "_emotion_manager", manager)
    monkeypatch.setattr(provider, "_state_orchestrator_factory", FakeStateOrchestrator)
    monkeypatch.setattr(provider, "_read_soul_mode_layer", lambda mode: f"# {mode}")
    provider.on_turn_start(1, "处理任务", session_id="s1")
    assert provider.request_overrides()

    monkeypatch.setattr(manager, "update_emotion_state", lambda messages: False)
    with pytest.raises(RuntimeError, match="emotion update failed"):
        provider.on_turn_start(2, "下一轮", session_id="s1")

    assert provider.request_overrides() == {}


def test_turn_start_injects_state_machine_selected_soul_and_writes_exact_capture(tmp_path, monkeypatch):
    provider = _load_provider_class()()
    FakeStateOrchestrator.calls = []
    monkeypatch.setattr(provider, "_emotion_manager_factory", FakeEmotionManager)
    monkeypatch.setattr(provider, "_state_orchestrator_factory", FakeStateOrchestrator)
    monkeypatch.setattr(provider, "_runtime_capture_path", tmp_path / "latest-turn.json")
    monkeypatch.setattr(provider, "_read_soul_mode_layer", lambda mode: f"# {mode} layer exact body")
    monkeypatch.setattr(provider, "_load_memory_context", lambda **kwargs: "memory")

    provider.on_turn_start(3, "检查生产状态", session_id="s1")
    context = provider.prefetch("检查生产状态", session_id="s1")
    capture = json.loads((tmp_path / "latest-turn.json").read_text(encoding="utf-8"))

    assert "<state_machine_injection>" in context
    assert "mode: work" in context
    assert "# work layer exact body" in context
    assert capture["source"] == "exact_host_capture"
    assert capture["state_machine"]["mode"] == "work"
    assert capture["state_machine"]["selected_layers"] == ["core", "work"]
    assert capture["state_machine"]["semantic_shadow"] == {
        "primary_mode": "work",
        "confidence": 0.91,
        "backend": "rules+local-lightweight",
        "shadow_only": True,
    }
    assert capture["state_machine"]["authority_source"] == "new_table"
    assert capture["mode_sync"] == {
        "state_machine_mode": "work",
        "pcltm_mode": "work",
        "status": "consistent",
    }
    assert capture["emotion_modifier"] == "<emotion_modifier>updated-before-reply</emotion_modifier>"
    assert capture["soul_mode_layer"]["content"] == "# work layer exact body"
    assert capture["turn_injection"] in context
    assert context.index("<state_machine_injection>") < context.index("<soullink_turn_state>")
    assert capture["turn_injection"].rstrip().endswith("</soullink_turn_state>")
    assert context.index("memory") < context.index("<state_machine_injection>")
    assert context.rstrip().endswith("</soullink_turn_state>")


def test_final_forward_capture_exposes_only_records_actually_present_in_outbound(provider_factory, monkeypatch):
    provider = provider_factory()
    monkeypatch.setattr(provider, "_emotion_manager_factory", FakeEmotionManager)
    memory = (
        "<pcltm_context>\n"
        "【governed_memory_view】\n"
        "【selected_records】\n"
        "- [user] first governed memory\n"
        "- [runtime_boundary] second governed memory\n"
        "</pcltm_context>"
    )
    monkeypatch.setattr(provider, "_load_memory_context", lambda **kwargs: memory)

    provider.on_turn_start(5, "检查接口", session_id="s1")
    injected = provider.prefetch("检查接口", session_id="s1")
    messages = [{"role": "user", "content": f"request\n\n{injected}"}]
    before = json.loads(json.dumps(messages))

    provider.on_before_model_forward(messages)
    capture = json.loads(provider._runtime_capture_path.read_text(encoding="utf-8"))

    assert messages == before
    assert capture["forwarded_model_boundary"] == {
        "status": "captured", "source": "final_model_forward",
    }
    assert capture["memory_selection"]["selected_count"] == 2
    assert capture["memory_selection"]["candidate_records"] == {"status": "unavailable"}
    assert capture["memory_selection"]["judgment_workset"] == {"status": "unavailable"}
    assert capture["memory_selection"]["selected_records"][0]["bucket"] == "user"
    assert capture["memory_selection"]["selected_records"][0]["content"] == "first governed memory"
    assert len(capture["memory_selection"]["selected_records"][0]["content_sha256"]) == 64


def test_final_forward_capture_merges_same_prefetch_candidate_and_judgment_observation(provider_factory, monkeypatch):
    provider = provider_factory()
    monkeypatch.setattr(provider, "_emotion_manager_factory", FakeEmotionManager)
    memory = "<pcltm_context>\n【selected_records】\n- [user] selected\n</pcltm_context>"
    observation = {
        "status": "captured",
        "context_sha256": __import__("hashlib").sha256(memory.encode("utf-8")).hexdigest(),
        "candidate_records": {"status": "captured", "records": [{"record_id": 41}]},
        "judgment_workset": {
            "status": "captured",
            "records": [{"record_id": 41, "selection_decision": "selected", "budget_decision": "admitted"}],
        },
        "governor_result": {"within_budget": True, "omitted_chars": 0},
    }
    monkeypatch.setattr(provider, "_load_memory_context", lambda **kwargs: memory)
    monkeypatch.setattr(provider, "_load_memory_selection_observation", lambda: observation)

    provider.on_turn_start(9, "检查完整漏斗", session_id="s1")
    injected = provider.prefetch("检查完整漏斗", session_id="s1")
    messages = [{"role": "user", "content": injected}]
    before = json.loads(json.dumps(messages))
    provider.on_before_model_forward(messages)
    capture = json.loads(provider._runtime_capture_path.read_text(encoding="utf-8"))

    assert messages == before
    assert capture["memory_selection"]["candidate_records"] == observation["candidate_records"]
    assert capture["memory_selection"]["judgment_workset"] == observation["judgment_workset"]
    assert capture["memory_selection"]["governor_result"] == observation["governor_result"]


def test_final_forward_capture_fails_closed_when_injected_context_was_removed(provider_factory, monkeypatch):
    provider = provider_factory()
    monkeypatch.setattr(provider, "_emotion_manager_factory", FakeEmotionManager)
    memory = "<pcltm_context>\n【selected_records】\n- [user] selected\n</pcltm_context>"
    monkeypatch.setattr(provider, "_load_memory_context", lambda **kwargs: memory)

    provider.on_turn_start(6, "检查缺失", session_id="s1")
    provider.prefetch("检查缺失", session_id="s1")
    provider.on_before_model_forward([{"role": "user", "content": "context removed"}])
    capture = json.loads(provider._runtime_capture_path.read_text(encoding="utf-8"))

    assert capture["forwarded_model_boundary"]["status"] == "unavailable"
    assert capture["memory_selection"] == {
        "status": "unavailable",
        "reason": "injected_memory_context_not_present_in_final_messages",
        "selected_records": [],
        "candidate_records": {"status": "unavailable"},
        "judgment_workset": {"status": "unavailable"},
    }


def test_pcltm_failure_records_error_instead_of_false_consistency(tmp_path, monkeypatch):
    provider = _load_provider_class()()
    monkeypatch.setattr(provider, "_emotion_manager_factory", FakeEmotionManager)
    monkeypatch.setattr(provider, "_state_orchestrator_factory", FakeStateOrchestrator)
    monkeypatch.setattr(provider, "_runtime_capture_path", tmp_path / "latest-turn.json")
    monkeypatch.setattr(provider, "_read_soul_mode_layer", lambda mode: f"# {mode}")
    monkeypatch.setattr(
        provider,
        "_load_memory_context",
        lambda **kwargs: (_ for _ in ()).throw(RuntimeError("pcltm failed")),
    )

    provider.on_turn_start(4, "检查生产状态", session_id="s1")
    pending = json.loads((tmp_path / "latest-turn.json").read_text(encoding="utf-8"))
    assert pending["mode_sync"]["status"] == "pending"

    try:
        provider.prefetch("检查生产状态", session_id="s1")
    except RuntimeError as error:
        assert str(error) == "pcltm failed"
    else:
        raise AssertionError("PCLTM failure must remain visible")

    failed = json.loads((tmp_path / "latest-turn.json").read_text(encoding="utf-8"))
    assert failed["mode_sync"] == {
        "state_machine_mode": "work",
        "pcltm_mode": None,
        "status": "error",
    }


def test_same_turn_start_is_idempotent_but_next_turn_updates(provider_factory, monkeypatch):
    provider = provider_factory()
    FakeEmotionManager.updates = []
    monkeypatch.setattr(provider, "_emotion_manager_factory", FakeEmotionManager)

    provider.on_turn_start(7, "同一消息", session_id="s1")
    provider.on_turn_start(7, "同一消息", session_id="s1")
    provider.on_turn_start(8, "下一消息", session_id="s1")

    assert [x[0]["content"] for x in FakeEmotionManager.updates] == ["同一消息", "下一消息"]


def test_capture_uses_unique_host_turn_identity_not_resettable_host_counter(provider_factory, monkeypatch):
    provider = provider_factory()
    FakeEmotionManager.updates = []
    monkeypatch.setattr(provider, "_emotion_manager_factory", FakeEmotionManager)

    provider.on_turn_start(3, "压缩前", session_id="s1", turn_id="s1:task-a:turn-a")
    first = json.loads(provider._runtime_capture_path.read_text(encoding="utf-8"))
    provider.on_turn_start(2, "压缩后", session_id="s1", turn_id="s1:task-b:turn-b")
    second = json.loads(provider._runtime_capture_path.read_text(encoding="utf-8"))

    assert "turn_number" not in first
    assert "turn_number" not in second
    assert first["host_turn_count"] == 3
    assert second["host_turn_count"] == 2
    assert second["host_turn_count_semantics"] == "session_local_non_authoritative"
    assert first["host_turn_id"] == "s1:task-a:turn-a"
    assert second["host_turn_id"] == "s1:task-b:turn-b"
    assert first["turn_correlation_id"] != second["turn_correlation_id"]
    assert second["turn_correlation_provenance"] == "hermes_turn_id"


def test_capture_exposes_semantic_and_authority_audit_for_observability(provider_factory, monkeypatch):
    provider = provider_factory()
    monkeypatch.setattr(provider, "_emotion_manager_factory", FakeEmotionManager)

    provider.on_turn_start(3, "检查状态机", session_id="s1", turn_id="turn-semantic")
    capture = json.loads(provider._runtime_capture_path.read_text(encoding="utf-8"))
    state = capture["state_machine"]

    assert capture["host_turn_count"] == 3
    assert "turn_number" not in capture
    assert "semantic_shadow" in state
    assert "semantic_fusion" in state
    assert "authority_source" in state


def test_capture_marks_generated_turn_identity_as_degraded(provider_factory, monkeypatch):
    provider = provider_factory()
    FakeEmotionManager.updates = []
    monkeypatch.setattr(provider, "_emotion_manager_factory", FakeEmotionManager)

    provider.on_turn_start(1, "旧宿主", session_id="s1")
    capture = json.loads(provider._runtime_capture_path.read_text(encoding="utf-8"))

    assert capture["host_turn_id"] is None
    assert capture["turn_correlation_provenance"] == "generated_fallback"
    assert capture["turn_correlation_id"]


class FailedEmotionManager(FakeEmotionManager):
    def update_emotion_state(self, messages):
        self.updates.append(messages)
        return False


def test_failed_new_turn_clears_previous_emotion_injection(provider_factory, monkeypatch):
    provider = provider_factory()
    FakeEmotionManager.updates = []
    monkeypatch.setattr(provider, "_emotion_manager_factory", FakeEmotionManager)
    monkeypatch.setattr(provider, "_load_memory_context", lambda **kwargs: "memory-only")

    provider.on_turn_start(1, "第一条", session_id="s1")
    assert "updated-before-reply" in provider.prefetch("第一条")

    provider._emotion_manager = FailedEmotionManager()
    try:
        provider.on_turn_start(2, "第二条", session_id="s1")
    except RuntimeError:
        pass
    else:
        raise AssertionError("failed emotion update must be visible to the lifecycle caller")

    assert provider.prefetch("第二条") == "memory-only"


def test_skill_scaffolding_is_not_scored_as_user_emotion(provider_factory, monkeypatch):
    provider = provider_factory()
    FakeEmotionManager.updates = []
    monkeypatch.setattr(provider, "_emotion_manager_factory", FakeEmotionManager)

    provider.on_turn_start(
        1,
        "[IMPORTANT: The user has invoked the test skill. The full skill content is loaded below.]",
        session_id="s1",
    )

    assert FakeEmotionManager.updates == []


def test_session_switch_restores_last_mode_for_known_session(provider_factory, monkeypatch):
    provider = provider_factory()
    monkeypatch.setattr(provider, "_emotion_manager_factory", FakeEmotionManager)
    FakeStateOrchestrator.calls = []

    provider.on_session_switch("s1")
    provider.on_turn_start(1, "检查状态机", session_id="s1")
    assert provider._active_mode == "work"

    provider.on_session_switch("s2")
    assert provider._active_mode is None
    provider.on_session_switch("s1")
    assert provider._active_mode == "work"

    provider.on_turn_start(2, "下一步呢", session_id="s1")
    assert FakeStateOrchestrator.calls[-1]["previous_mode"] == "work"


def test_compression_session_inherits_parent_mode(provider_factory, monkeypatch):
    provider = provider_factory()
    monkeypatch.setattr(provider, "_emotion_manager_factory", FakeEmotionManager)
    FakeStateOrchestrator.calls = []

    provider.on_session_switch("parent")
    provider.on_turn_start(10, "检查状态机", session_id="parent")
    assert provider._active_mode == "work"

    provider.on_session_switch(
        "compressed-child",
        parent_session_id="parent",
        reset=True,
        reason="compression",
    )

    assert provider._active_mode == "work"
    assert provider._session_modes["compressed-child"] == "work"
    provider.on_turn_start(11, "好下一步", session_id="compressed-child")
    assert FakeStateOrchestrator.calls[-1]["previous_mode"] == "work"


def test_new_session_reset_does_not_inherit_parent_mode(provider_factory, monkeypatch):
    provider = provider_factory()
    monkeypatch.setattr(provider, "_emotion_manager_factory", FakeEmotionManager)

    provider.on_session_switch("parent")
    provider.on_turn_start(1, "检查状态机", session_id="parent")
    provider.on_session_switch(
        "fresh-session",
        parent_session_id="parent",
        reset=True,
        reason="new_session",
    )

    assert provider._active_mode is None
    assert "fresh-session" not in provider._session_modes


def test_explicit_session_reset_clears_saved_mode(provider_factory, monkeypatch):
    provider = provider_factory()
    monkeypatch.setattr(provider, "_emotion_manager_factory", FakeEmotionManager)

    provider.on_session_switch("s1")
    provider.on_turn_start(1, "检查状态机", session_id="s1")
    provider.on_session_switch("s1", reset=True)

    assert provider._active_mode is None
    assert "s1" not in provider._session_modes


def test_empty_or_control_like_turn_does_not_update(provider_factory, monkeypatch):
    provider = provider_factory()
    FakeEmotionManager.updates = []
    monkeypatch.setattr(provider, "_emotion_manager_factory", FakeEmotionManager)

    provider.on_turn_start(1, "", session_id="s1")
    provider.on_turn_start(2, "   ", session_id="s1")

    assert FakeEmotionManager.updates == []


def test_sync_turn_ingests_canonical_hermes_session_messages(tmp_path, monkeypatch):
    provider = _load_provider_class()()
    hermes_home = tmp_path / "hermes"
    hermes_home.mkdir()
    state_db = hermes_home / "state.db"
    with sqlite3.connect(state_db) as conn:
        conn.executescript(
            """
            CREATE TABLE sessions (
                id TEXT PRIMARY KEY, source TEXT NOT NULL, parent_session_id TEXT,
                started_at REAL NOT NULL, ended_at REAL, end_reason TEXT,
                archived INTEGER NOT NULL DEFAULT 0, rewind_count INTEGER NOT NULL DEFAULT 0,
                system_prompt TEXT
            );
            CREATE TABLE messages (
                id INTEGER PRIMARY KEY, session_id TEXT NOT NULL, role TEXT NOT NULL,
                content TEXT, tool_call_id TEXT, tool_calls TEXT, tool_name TEXT,
                timestamp REAL NOT NULL, token_count INTEGER, finish_reason TEXT,
                reasoning TEXT, reasoning_content TEXT, reasoning_details TEXT,
                codex_reasoning_items TEXT, codex_message_items TEXT,
                platform_message_id TEXT, observed INTEGER DEFAULT 0,
                active INTEGER NOT NULL DEFAULT 1, compacted INTEGER NOT NULL DEFAULT 0
            );
            INSERT INTO sessions VALUES ('s1','tui',NULL,1,NULL,NULL,0,0,'system body');
            INSERT INTO messages (id,session_id,role,content,timestamp) VALUES
                (10,'s1','user','实时问题',2),
                (11,'s1','assistant','实时回答',3);
            """
        )
    pcltm_db = tmp_path / "pcltm.db"
    monkeypatch.setenv("HERMES_HOME", str(hermes_home))
    monkeypatch.setenv("HERMES_PCLTM_DB", str(pcltm_db))

    provider.sync_turn("实时问题", "实时回答", session_id="s1", messages=[])
    provider.sync_turn("实时问题", "实时回答", session_id="s1", messages=[])

    with sqlite3.connect(pcltm_db) as conn:
        assert conn.execute("SELECT count(*) FROM events").fetchone()[0] == 3
        assert conn.execute("SELECT count(*) FROM ingest_events").fetchone()[0] == 3
        assert conn.execute("SELECT count(*) FROM event_fts").fetchone()[0] == 3
        assert [row[0] for row in conn.execute("SELECT content FROM events WHERE role != 'lifecycle' ORDER BY event_id")] == [
            "实时问题",
            "实时回答",
        ]
        assert conn.execute("SELECT count(*) FROM events WHERE role='lifecycle' AND inject_policy='retrieve_only'").fetchone()[0] == 1
