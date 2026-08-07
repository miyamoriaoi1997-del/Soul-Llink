from hashlib import sha256
from pathlib import Path

from soul_link import SoulLink
from soul_link.contracts import PERSONA_ACTIVE, resolve_persona_engine_base_dir


class FakeEmotionStateManager:
    calls = []

    def __init__(self, *args, **kwargs):
        self.__class__.calls.append(("init", args, kwargs))

    def apply_time_decay_if_needed(self):
        self.__class__.calls.append(("decay",))
        return True

    def update_emotion_state(self, messages):
        self.__class__.calls.append(("update", messages))
        return True

    def get_current_emotion_state(self):
        self.__class__.calls.append(("state",))
        return {"emotion_score": 1.5, "current_emotion": 1.5, "mode": "work"}

    def get_tone_modifiers(self):
        self.__class__.calls.append(("modifier",))
        return "test modifier"


class _FakeActiveFrame:
    active_text = "[system]\nruntime boundary"
    audit = {
        "active_layers": ["system", "pinned", "transient"],
        "selected_layers": ["system", "pinned", "transient"],
        "prompt_active_layers": ["system", "pinned", "transient"],
        "selected_buckets": ["runtime_boundary"],
        "compression": {"is_reference_only": True},
        "layers": [{"layer": "system"}],
    }

    def to_dict(self):
        return dict(self.audit)


class _FakeLayeredMemoryView:
    def active_frame(self):
        return _FakeActiveFrame()


def _install_structured_memory_view(monkeypatch):
    import pcltm.memory_adapter as memory_adapter

    monkeypatch.setattr(memory_adapter, "load_layered_prompt_context", lambda **_: _FakeLayeredMemoryView())


def test_soullink_facade_exports_and_resolves_minimally():
    link = SoulLink()
    request = link.ingest(
        "继续做收口",
        recent_context=[{"role": "user", "content": "前文"}],
        previous_mode="daily",
        emotion_state={"emotion_score": 1.5, "current_emotion": 1.5},
        platform="telegram",
    )

    resolution = link.resolve(request)

    assert request.message == "继续做收口"
    assert resolution.mode in {"daily", "work", "sex"}
    assert resolution.route_bucket in {"relationship", "task", "sex", ""}
    assert isinstance(resolution.selected_layers, list)
    assert isinstance(resolution.shadow_packet, dict)
    assert isinstance(resolution.audit_packet, dict)
    assert not hasattr(link, "orchestrator")


def test_persona_engine_base_dir_falls_back_from_non_soullink_cwd(tmp_path, monkeypatch):
    external_cwd = tmp_path / "external"
    external_cwd.mkdir()
    monkeypatch.chdir(external_cwd)

    assert resolve_persona_engine_base_dir(Path(".")) == PERSONA_ACTIVE


def test_soullink_compose_active_prompt_owns_emotion_boundary(monkeypatch):
    _install_structured_memory_view(monkeypatch)
    monkeypatch.setattr(
        "persona_engine.emotion_state_manager.EmotionStateManager",
        FakeEmotionStateManager,
    )
    FakeEmotionStateManager.calls = []
    link = SoulLink()

    resolution = link.compose_active_prompt(
        host_system_prompt=(
            "Host Prompt\n\n"
            "<pcltm_context>\n[system]\nstale active memory\n</pcltm_context>\n\n"
            "<pcltm_context>\n[system]\nhost active memory\n</pcltm_context>"
        ),
        user_message="继续做收口",
        recent_context=[{"role": "assistant", "content": "前文"}],
        platform="telegram",
    )

    assert resolution.prompt_candidate is not None
    assert "prompt_text" in resolution.prompt_candidate
    prompt_text = resolution.prompt_candidate["prompt_text"]
    assert prompt_text.count("<pcltm_context>\n") == 1
    assert prompt_text.count("\n</pcltm_context>") == 1
    assert "stale active memory" not in prompt_text
    assert "host active memory" not in prompt_text
    assert "runtime boundary" not in prompt_text
    assert "<pcltm_memory_view>" not in prompt_text
    assert "<memory_profile_notes>" not in prompt_text
    request = resolution.audit_packet["request"]
    assert request["emotion_state"]["mode"] == "work"
    assert request["emotion_modifier"] == "test modifier"
    assert request["recent_context"] == [{"role": "assistant", "content": "前文"}]
    call_names = [call[0] for call in FakeEmotionStateManager.calls]
    assert "decay" not in call_names
    assert call_names.index("update") < call_names.index("state") < call_names.index("modifier")
    update_messages = next(call[1] for call in FakeEmotionStateManager.calls if call[0] == "update")
    assert update_messages == [
        {"role": "assistant", "content": "前文"},
        {"role": "user", "content": "继续做收口"},
    ]


def test_resolve_uses_single_analysis_for_prompt_candidate(monkeypatch):
    link = SoulLink()
    calls = {"analyze": 0}
    original_analyze = link._orchestrator._analyze

    def counting_analyze(*args, **kwargs):
        calls["analyze"] += 1
        return original_analyze(*args, **kwargs)

    monkeypatch.setattr(link._orchestrator, "_analyze", counting_analyze)
    request = link.ingest(
        "好按你的来优化状态机。",
        emotion_state={"emotion_score": 1.5, "current_emotion": 1.5},
        platform="telegram",
    )

    resolution = link.resolve(
        request,
        host_system_prompt="host\n<pcltm_context>\nold duplicate\n</pcltm_context>",
    )

    assert calls["analyze"] == 1
    assert resolution.prompt_candidate is not None
    prompt_text = resolution.prompt_candidate["prompt_text"]
    assert resolution.prompt_candidate["prompt_hash"] == sha256(prompt_text.encode("utf-8")).hexdigest()[:16]
    assert resolution.prompt_candidate["prompt_hash"] == resolution.shadow_packet["prompt_hash"]
    assert prompt_text.count("<persona_orchestrator_prompt>") == 1
    assert prompt_text.count("<pcltm_context>") == 1
    assert prompt_text.count("<pcltm_context_boundary>") == 1
    assert "old duplicate" not in prompt_text
