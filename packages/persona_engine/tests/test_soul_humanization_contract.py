from pathlib import Path

from persona_orchestrator.prompt_composer import PromptComposer

BASE = Path(__file__).resolve().parents[1]
LAYERS = BASE / "soul_layers"


def test_public_core_is_configurable_and_preserves_truth_and_safety():
    core = (LAYERS / "SOUL.core.template.md").read_text(encoding="utf-8")

    assert "configurable persona runtime instance" in core
    assert "Emotion changes expression, not truth" in core
    assert "Safety, factual accuracy, tool discipline" in core
    assert "Do not encode private names" in core


def test_mode_layers_modify_behavior_without_redefining_identity():
    for path in LAYERS.glob("SOUL.*.template.md"):
        if path.name == "SOUL.core.template.md":
            continue
        text = path.read_text(encoding="utf-8")
        assert "must not redefine" in text
        assert "# Core Identity Layer" not in text


def test_public_adult_layer_is_non_explicit_and_consent_aware():
    adult = (LAYERS / "SOUL.sex.template.md").read_text(encoding="utf-8")

    assert "# Adult Boundary Layer" in adult
    assert "does not ship explicit adult persona instructions" in adult
    assert "preserve consent and boundaries" in adult
    assert "private overlay" in adult


def test_daily_layer_preserves_warmth_continuity_and_boundaries():
    daily = (LAYERS / "SOUL.daily.template.md").read_text(encoding="utf-8")

    assert "casual conversation" in daily
    assert "preserving boundaries" in daily
    assert "preserving continuity" in daily
    assert "Do not turn daily mode into adult content" in daily


def test_work_layer_prioritizes_evidence_risk_and_verification():
    work = (LAYERS / "SOUL.work.template.md").read_text(encoding="utf-8")

    assert "证据" in work
    assert "风险" in work
    assert "不可逆的动作必须先确认" in work
    assert "结果是否可验证" in work


def test_active_prompt_keeps_emotion_modifier_after_soul_layers():
    result = PromptComposer(str(BASE), core_source="host_core").compose_active(
        host_system_prompt="# Core Identity Layer\n\nidentity",
        selected_layers=["work"],
        memory_notes="memory below emotion",
        emotion_modifier="<emotion_modifier>highest priority</emotion_modifier>",
    )

    assert result.prompt_text.count("<emotion_modifier>") == 1
    assert result.prompt_text.rstrip().endswith(
        "<emotion_modifier>highest priority</emotion_modifier>"
    )
    assert result.prompt_text.index("# Work Mode Layer") < result.prompt_text.index(
        "<emotion_modifier>"
    )