from __future__ import annotations

from pathlib import Path

from persona_engine.persona_orchestrator.prompt_composer import PromptComposer


def test_prompt_composer_caches_layer_templates_between_composes(monkeypatch):
    composer = PromptComposer(Path(__file__).resolve().parents[1] / "packages" / "persona_engine")
    daily_template = composer.layers_dir / "SOUL.daily.template.md"
    real_read_text = Path.read_text
    reads: list[str] = []

    def counting_read_text(self: Path, *args, **kwargs):
        if self == daily_template:
            reads.append(str(self))
        return real_read_text(self, *args, **kwargs)

    monkeypatch.setattr(Path, "read_text", counting_read_text)

    first = composer.compose(["daily"])
    second = composer.compose(["daily"])

    assert first.prompt_text == second.prompt_text
    assert first.prompt_hash == second.prompt_hash
    assert reads == [str(daily_template)]


def test_compose_active_keeps_single_active_memory_and_persona_surface():
    composer = PromptComposer(Path(__file__).resolve().parents[1] / "packages" / "persona_engine")
    host_prompt = """host rules
<persona_orchestrator_prompt>
stale managed
</persona_orchestrator_prompt>
<pcltm_context>old context</pcltm_context>
<memory_profile_notes>old notes</memory_profile_notes>
<pcltm_memory_view>legacy view</pcltm_memory_view>
<emotion_modifier>old emotion</emotion_modifier>
<pcltm_context>fresh context</pcltm_context>
"""

    composition = composer.compose_active(
        host_system_prompt=host_prompt,
        selected_layers=["daily"],
        emotion_modifier="<emotion_modifier>new emotion</emotion_modifier>",
    )

    prompt = composition.prompt_text
    assert prompt.count("<persona_orchestrator_prompt>") == 1
    assert prompt.count("</persona_orchestrator_prompt>") == 1
    assert prompt.count("<pcltm_context>") == 1
    assert "fresh context" in prompt
    assert "old context" not in prompt
    assert "<pcltm_memory_view>" not in prompt
    assert "<memory_profile_notes>" not in prompt
    assert "old emotion" not in prompt
    assert prompt.index("</soul_runtime_boundary>") < prompt.index("<emotion_modifier>new emotion</emotion_modifier>")
    assert prompt.rstrip().endswith("<emotion_modifier>new emotion</emotion_modifier>")


def test_compose_active_replaces_host_pcltm_with_fresh_frame():
    composer = PromptComposer(Path(__file__).resolve().parents[1] / "packages" / "persona_engine", core_source=PromptComposer.CORE_SOURCE_HOST)
    host = "host prefix\n\n<pcltm_context>\nold duplicate\n</pcltm_context>\n\nhost suffix"

    result = composer.compose_active(
        host_system_prompt=host,
        selected_layers=["work"],
        memory_view_text="fresh active frame",
    )

    assert result.prompt_text.count("<pcltm_context>") == 1
    assert "fresh active frame" in result.prompt_text
    assert "old duplicate" not in result.prompt_text
    assert result.prompt_text.index("fresh active frame") < result.prompt_text.index("<persona_orchestrator_prompt>")


def test_compose_active_removes_stale_host_pcltm_when_no_fresh_frame():
    composer = PromptComposer(Path(__file__).resolve().parents[1] / "packages" / "persona_engine", core_source=PromptComposer.CORE_SOURCE_HOST)

    result = composer.compose_active(
        host_system_prompt="host prefix\n<pcltm_context>\nstale\n</pcltm_context>",
        selected_layers=["work"],
        memory_view_text="",
    )

    assert result.prompt_text.count("<pcltm_context>") == 1
    assert "stale" in result.prompt_text
