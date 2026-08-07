"""Regression gates for the semantic-shadow wiring.

The shadow classifier existed in StateOrchestrator but was never wired:
`enable_semantic_shadow` defaulted to False and the production
SoulLinkMemoryProvider never passed it, so Layer 3 never ran in production.
These tests pin the config gate and the production hand-off.
"""
from __future__ import annotations

import sys
from pathlib import Path

import pytest

PACKAGES = Path(__file__).resolve().parents[2] / "packages"
PE = PACKAGES / "persona_engine"
sys.path.insert(0, str(PE))

from persona_engine.persona_orchestrator.state_orchestrator import StateOrchestrator


def test_semantic_shadow_default_off(tmp_path: Path) -> None:
    orch = StateOrchestrator(tmp_path)
    assert orch.semantic_classifier is None


def test_semantic_shadow_enabled_builds_classifier(tmp_path: Path) -> None:
    orch = StateOrchestrator(tmp_path, enable_semantic_shadow=True)
    assert orch.semantic_classifier is not None


def test_runtime_config_exposes_semantic_shadow_key() -> None:
    """_state_machine_runtime_config must surface a semantic_shadow bool.

    Reads the real config.yaml (read-only); only asserts the key exists and is a
    bool so a future default-flip does not silently break the wiring contract.
    """
    sys.path.insert(0, str(PACKAGES.parent / "soul_link" / "hermes_plugin"))
    from memory_provider import _state_machine_runtime_config

    cfg = _state_machine_runtime_config()
    assert "semantic_shadow" in cfg
    assert isinstance(cfg["semantic_shadow"], bool)


def test_runtime_config_exposes_semantic_backend_key() -> None:
    """_state_machine_runtime_config must surface a semantic_backend str,
    defaulting to 'local' (fail closed) when unset.
    """
    sys.path.insert(0, str(PACKAGES.parent / "soul_link" / "hermes_plugin"))
    from memory_provider import _state_machine_runtime_config

    cfg = _state_machine_runtime_config()
    assert "semantic_backend" in cfg
    assert isinstance(cfg["semantic_backend"], str)
    assert cfg["semantic_backend"] in {"local", "local_lightweight", "rules+local"}


def test_runtime_config_exposes_semantic_authority_key() -> None:
    """Production fusion is on by default but remains explicitly disableable."""
    sys.path.insert(0, str(PACKAGES.parent / "soul_link" / "hermes_plugin"))
    from memory_provider import _state_machine_runtime_config

    cfg = _state_machine_runtime_config()
    assert isinstance(cfg["semantic_authority"], bool)


def test_semantic_backend_local_lightweight_runs_emotion_path(tmp_path: Path) -> None:
    """semantic_backend='local_lightweight' must route the shadow through the
    sentiment-augmented path and mark backend as rules+local-lightweight.

    The neural model is only asserted when it actually loaded; when it is not
    available the path must degrade to LOCAL_SENTIMENT_UNAVAILABLE instead of
    silently falling back to the deterministic backend.
    """
    orch = StateOrchestrator(
        tmp_path,
        enable_semantic_shadow=True,
        semantic_backend="local_lightweight",
    )
    assert orch.semantic_classifier is not None

    # Trigger a blocking model load so _available is set; the shadow path
    # skips neural sentiment while _available is not True.
    from persona_engine.sentiment_analyzer import SentimentAnalyzer

    model_ready = SentimentAnalyzer.get_instance().available
    shadow = orch.semantic_classifier.classify(
        user_message="老师，我今天真的很累，想靠着你待一会儿。",
        previous_mode="work",
        platform="cli",
    )
    assert shadow["backend"] == "rules+local-lightweight"
    assert shadow["shadow_only"] is True
    codes = shadow.get("reason_codes", [])
    if model_ready:
        assert any(c.startswith("LOCAL_SENTIMENT:") for c in codes), (
            f"emotion model ready but sentiment missing: {codes}"
        )
    else:
        assert "LOCAL_SENTIMENT_UNAVAILABLE" in codes, f"unexpected codes: {codes}"
