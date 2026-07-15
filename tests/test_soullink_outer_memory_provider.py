from __future__ import annotations

import importlib.util
from pathlib import Path
import sys
from types import ModuleType


def _load_provider_class():
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
    plugin_path = (
        Path(__file__).resolve().parents[1]
        / "adapters/hermes/memory_provider/__init__.py"
    )
    spec = importlib.util.spec_from_file_location("soullink_outer_plugin_under_test", plugin_path)
    module = importlib.util.module_from_spec(spec)
    assert spec and spec.loader
    spec.loader.exec_module(module)
    return module.SoulLinkMemoryProvider


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
