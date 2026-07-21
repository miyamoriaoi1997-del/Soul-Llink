from __future__ import annotations

import importlib.util
import json
import sqlite3
import sys
from pathlib import Path
from types import ModuleType, SimpleNamespace

import pytest


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
        skill_commands_module.extract_user_instruction_from_skill_message = (
            lambda message: "" if message.startswith("[IMPORTANT: The user has invoked") else message
        )
        agent_module.memory_provider = memory_provider_module
        agent_module.skill_commands = skill_commands_module
        sys.modules["agent"] = agent_module
        sys.modules["agent.memory_provider"] = memory_provider_module
        sys.modules["agent.skill_commands"] = skill_commands_module
    skill_commands = sys.modules.get("agent.skill_commands")
    if skill_commands is not None and getattr(skill_commands, "_soullink_test_stub", False):
        skill_commands.extract_user_instruction_from_skill_message = (
            lambda message: "" if message.startswith("[IMPORTANT: The user has invoked") else message
        )
    plugin_path = Path(__file__).resolve().parents[1] / "adapters/hermes/memory_provider/__init__.py"
    spec = importlib.util.spec_from_file_location("soullink_realtime_emotion_provider", plugin_path)
    module = importlib.util.module_from_spec(spec)
    assert spec and spec.loader
    spec.loader.exec_module(module)
    return module.SoulLinkMemoryProvider


def test_installed_nested_provider_resolves_sibling_production_repo(tmp_path, monkeypatch):
    source = Path(__file__).resolve().parents[1] / "adapters/hermes/memory_provider/__init__.py"
    plugins = tmp_path / "plugins"
    installed = plugins / "soullink" / "memory_provider" / "__init__.py"
    installed.parent.mkdir(parents=True)
    installed.write_bytes(source.read_bytes())
    production = plugins / "Soul-Llink" / "packages" / "persona_engine"
    production.mkdir(parents=True)
    monkeypatch.delenv("SOULLINK_ROOT", raising=False)
    spec = importlib.util.spec_from_file_location("installed_soullink_provider", installed)
    module = importlib.util.module_from_spec(spec)
    assert spec and spec.loader
    spec.loader.exec_module(module)
    assert module._soullink_root() == plugins / "Soul-Llink"


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
            route_metadata={"reason_codes": ["work_override"]},
        )


def test_turn_start_updates_before_prefetch_and_prefetch_injects_updated_emotion(provider_factory, monkeypatch):
    provider = provider_factory()
    FakeEmotionManager.updates = []
    monkeypatch.setattr(provider, "_emotion_manager_factory", FakeEmotionManager)
    monkeypatch.setattr(provider, "_load_memory_context", lambda **kwargs: "<pcltm_context>memory</pcltm_context>")

    provider.on_turn_start(7, "用户消息", session_id="s1")
    context = provider.prefetch("用户消息", session_id="s1")

    assert FakeEmotionManager.updates == [[{"role": "user", "content": "用户消息"}]]
    assert "<soullink_turn_state>" in context
    assert "affection: 88" in context
    assert "current_emotion: 2.05" in context
    assert "updated-before-reply" in context
    assert context.index("<soullink_turn_state>") < context.index("<pcltm_context>")


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
    assert capture["mode_sync"] == {
        "state_machine_mode": "work",
        "pcltm_mode": "work",
        "status": "consistent",
    }
    assert capture["emotion_modifier"] == "<emotion_modifier>updated-before-reply</emotion_modifier>"
    assert capture["soul_mode_layer"]["content"] == "# work layer exact body"
    assert capture["turn_injection"] in context
    assert capture["turn_correlation_id"]
    overrides = provider.request_overrides()
    assert overrides["extra_body"]["metadata"]["hermes_turn_correlation_id"] == capture["turn_correlation_id"]


def test_request_overrides_expose_only_current_state_machine_route(provider_factory):
    provider = provider_factory()
    provider._turn_route_overrides = {
        "extra_body": {
            "metadata": {
                "hermes_route_bucket": "task",
                "hermes_model_hint": "technical",
                "hermes_selected_model": "work-model",
                "hermes_turn_correlation_id": "turn-id",
                "private_internal": "must-not-leak",
            }
        }
    }

    assert provider.request_overrides() == {
        "extra_body": {
            "metadata": {
                "hermes_route_bucket": "task",
                "hermes_model_hint": "technical",
                "hermes_selected_model": "work-model",
                "hermes_turn_correlation_id": "turn-id",
            }
        }
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
