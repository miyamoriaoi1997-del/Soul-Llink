from __future__ import annotations

import importlib.util
import hashlib
from pathlib import Path
import sys
from types import ModuleType, SimpleNamespace

import pytest


def _load_outer_provider_module():
    try:
        import agent.memory_provider  # type: ignore[import-not-found]  # noqa: F401
    except ModuleNotFoundError:
        agent_module = ModuleType("agent")
        agent_module._soullink_test_stub = True
        memory_provider_module = ModuleType("agent.memory_provider")
        skill_commands_module = ModuleType("agent.skill_commands")
        skill_commands_module._soullink_test_stub = True

        class MemoryProvider:
            pass

        memory_provider_module.MemoryProvider = MemoryProvider
        skill_commands_module.extract_user_instruction_from_skill_message = lambda message: message
        agent_module.memory_provider = memory_provider_module
        agent_module.skill_commands = skill_commands_module
        sys.modules["agent"] = agent_module
        sys.modules["agent.memory_provider"] = memory_provider_module
        sys.modules["agent.skill_commands"] = skill_commands_module
    plugin_path = Path(__file__).resolve().parents[1] / "adapters/hermes/memory_provider/__init__.py"
    spec = importlib.util.spec_from_file_location("soullink_outer_plugin_under_test", plugin_path)
    module = importlib.util.module_from_spec(spec)
    assert spec and spec.loader
    spec.loader.exec_module(module)
    return module


def _load_provider_class():
    return _load_outer_provider_module().SoulLinkMemoryProvider


def _load_actual_provider_module():
    module_path = Path(__file__).resolve().parents[1] / "soul_link/hermes_plugin/memory_provider.py"
    spec = importlib.util.spec_from_file_location("soullink_actual_plugin_under_test", module_path)
    module = importlib.util.module_from_spec(spec)
    assert spec and spec.loader
    spec.loader.exec_module(module)
    return module


@pytest.mark.parametrize(
    ("config_text", "expected"),
    [
        (
            """
plugins:
  entries:
    soullink:
      state_machine:
        transition_table_shadow: true
        bounded_activation: true
""",
            {"transition_table_shadow": True, "bounded_activation": True},
        ),
        ("", {"transition_table_shadow": False, "bounded_activation": False}),
        (
            """
plugins:
  entries:
    soullink:
      state_machine:
        transition_table_shadow: 'true'
        bounded_activation: 1
""",
            {"transition_table_shadow": False, "bounded_activation": False},
        ),
    ],
)
def test_outer_adapter_state_machine_runtime_config_requires_explicit_boolean_host_flags(tmp_path, monkeypatch, config_text, expected):
    module = _load_outer_provider_module()
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    if config_text:
        (tmp_path / "config.yaml").write_text(config_text, encoding="utf-8")

    assert module._state_machine_runtime_config() == expected


@pytest.mark.parametrize(
    ("config_text", "expected"),
    [
        (
            """
plugins:
  entries:
    soullink:
      state_machine:
        transition_table_shadow: true
        bounded_activation: true
""",
            {"transition_table_shadow": True, "bounded_activation": True},
        ),
        ("", {"transition_table_shadow": False, "bounded_activation": False}),
        (
            """
plugins:
  entries:
    soullink:
      state_machine:
        transition_table_shadow: 'true'
        bounded_activation: 1
""",
            {"transition_table_shadow": False, "bounded_activation": False},
        ),
    ],
)
def test_actual_provider_state_machine_runtime_config_requires_explicit_boolean_host_flags(tmp_path, monkeypatch, config_text, expected):
    module = _load_actual_provider_module()
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    if config_text:
        (tmp_path / "config.yaml").write_text(config_text, encoding="utf-8")

    assert module._state_machine_runtime_config() == expected


def test_outer_provider_applies_host_state_machine_flags_after_construction(tmp_path, monkeypatch):
    provider = _load_provider_class()()
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    (tmp_path / "config.yaml").write_text(
        """
plugins:
  entries:
    soullink:
      state_machine:
        bounded_activation: true
""",
        encoding="utf-8",
    )
    transitions = SimpleNamespace(
        enable_shadow=False,
        enable_bounded_activation=False,
        shadow_table=None,
        shadow_comparator=None,
    )
    orchestrator = SimpleNamespace(transitions=transitions)
    provider._state_orchestrator_factory = lambda *args, **kwargs: orchestrator

    assert provider._get_state_orchestrator() is orchestrator
    assert transitions.enable_bounded_activation is True
    assert transitions.enable_shadow is True
    assert transitions.shadow_table is not None
    assert transitions.shadow_comparator is not None


def test_actual_provider_applies_bounded_host_flag_to_shadow_and_comparator(tmp_path, monkeypatch):
    module = _load_actual_provider_module()
    provider = module.SoulLinkMemoryProvider()
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    (tmp_path / "config.yaml").write_text(
        """
plugins:
  entries:
    soullink:
      state_machine:
        bounded_activation: true
""",
        encoding="utf-8",
    )
    transitions = SimpleNamespace(
        enable_shadow=False,
        enable_bounded_activation=False,
        shadow_table=None,
        shadow_comparator=None,
    )
    orchestrator = SimpleNamespace(transitions=transitions)
    provider._state_orchestrator_factory = lambda *args, **kwargs: orchestrator

    assert provider._get_state_orchestrator() is orchestrator
    assert transitions.enable_bounded_activation is True
    assert transitions.enable_shadow is True
    assert transitions.shadow_table is not None
    assert transitions.shadow_comparator is not None


def test_outer_prefetch_mode_hint_separates_work_and_relationship_queries():
    provider = _load_provider_class()()

    assert provider._prefetch_mode_for_query("检查soullink运行情况") == "work"
    assert provider._prefetch_mode_for_query("看看情绪值") == "work"
    assert provider._prefetch_mode_for_query("恋爱时候恋爱，工作时候工作，不会互相干扰了对吗") == "work"
    assert provider._prefetch_mode_for_query("我爱你") == "daily"
    assert provider._prefetch_mode_for_query("揉揉你") == "daily"


def test_outer_prefetch_mode_hint_keeps_ambiguous_short_turn_default():
    provider = _load_provider_class()()

    assert provider._prefetch_mode_for_query("那你做吧") is None


def test_outer_prefetch_mode_hint_allows_explicit_adult_boundary_mode():
    provider = _load_provider_class()()

    assert provider._prefetch_mode_for_query("我们做爱") == "sex"


def test_provider_reuses_only_captured_recall_intent_from_same_session(monkeypatch):
    provider = _load_provider_class()()
    provider._active_mode = "work"
    provider._runtime_capture_payload = None
    calls = []
    observations = iter((
        {"context_sha256": hashlib.sha256(b"memory-1").hexdigest(), "recall_intent": {"intent": "memory_retrieval_diagnostics"}},
        {"context_sha256": hashlib.sha256(b"memory-2").hexdigest(), "recall_intent": {"intent": "memory_retrieval_diagnostics"}},
        {"context_sha256": hashlib.sha256(b"memory-3").hexdigest(), "recall_intent": {"intent": "default"}},
    ))

    def fake_load_memory_context(**kwargs):
        calls.append(kwargs)
        return f"memory-{len(calls)}"

    monkeypatch.setattr(provider, "_load_memory_context", fake_load_memory_context)
    monkeypatch.setattr(provider, "_load_memory_selection_observation", lambda: next(observations))

    provider.prefetch("优化长期记忆检索精准度", session_id="session-a")
    provider.prefetch("也就是说现在达到预期了吗", session_id="session-a")
    provider.prefetch("也就是说现在达到预期了吗", session_id="session-b")

    assert calls[0]["continuity_evidence"] is None
    assert calls[1]["continuity_evidence"].session_id == "session-a"
    assert calls[1]["continuity_evidence"].prior_intent.value == "memory_retrieval_diagnostics"
    assert calls[2]["continuity_evidence"] is None


def test_provider_rejects_stale_recall_observation(monkeypatch):
    provider = _load_provider_class()()
    provider._active_mode = "work"
    provider._runtime_capture_payload = None
    monkeypatch.setattr(provider, "_load_memory_context", lambda **kwargs: "current-memory")
    monkeypatch.setattr(
        provider,
        "_load_memory_selection_observation",
        lambda: {
            "context_sha256": hashlib.sha256(b"previous-memory").hexdigest(),
            "recall_intent": {"intent": "memory_retrieval_diagnostics"},
        },
    )

    provider.prefetch("普通当前问题", session_id="session-a")

    assert "session-a" not in provider._recall_intents_by_session


def test_provider_does_not_advance_continuity_when_load_fails(monkeypatch):
    provider = _load_provider_class()()
    provider._active_mode = "work"
    provider._runtime_capture_payload = None
    provider._recall_intents_by_session["session-a"] = "memory_retrieval_diagnostics"

    def fail(**kwargs):
        raise RuntimeError("load failed")

    monkeypatch.setattr(provider, "_load_memory_context", fail)

    try:
        provider.prefetch("切换到代码测试", session_id="session-a")
    except RuntimeError:
        pass
    else:
        raise AssertionError("expected load failure")

    assert provider._recall_intents_by_session["session-a"] == "memory_retrieval_diagnostics"


def test_owner_provider_loaded_by_installed_shim_has_session_recall_continuity(monkeypatch):
    provider = _load_actual_provider_module().SoulLinkMemoryProvider()
    provider._active_mode = "work"
    provider._runtime_capture_payload = None
    provider._turn_emotion_context = ""
    calls = []

    def fake_load_memory_context(**kwargs):
        calls.append(kwargs)
        return f"owner-memory-{len(calls)}"

    observations = iter((
        {
            "context_sha256": hashlib.sha256(b"owner-memory-1").hexdigest(),
            "recall_intent": {"intent": "memory_retrieval_diagnostics"},
        },
        {
            "context_sha256": hashlib.sha256(b"owner-memory-2").hexdigest(),
            "recall_intent": {"intent": "memory_retrieval_diagnostics"},
        },
    ))
    monkeypatch.setattr(provider, "_load_memory_context", fake_load_memory_context)
    monkeypatch.setattr(provider, "_load_memory_selection_observation", lambda: next(observations))

    provider.prefetch("诊断长期记忆召回的准确性", session_id="owner-session")
    provider.prefetch("也就是说现在达到预期了吗", session_id="owner-session")

    assert calls[0]["continuity_evidence"] is None
    assert calls[1]["continuity_evidence"].session_id == "owner-session"
    assert calls[1]["continuity_evidence"].prior_intent.value == "memory_retrieval_diagnostics"
