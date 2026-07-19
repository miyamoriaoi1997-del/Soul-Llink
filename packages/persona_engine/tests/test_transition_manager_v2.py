"""Tests for Phase 5.3: Integrated shadow transition comparison.

Tests TransitionManagerV2 with both legacy authority and shadow table comparison.
"""

import os
import pytest

from packages.persona_engine.persona_orchestrator.transition_manager_v2 import TransitionManagerV2
from packages.persona_engine.persona_orchestrator.types import ModeDecision


class TestTransitionManagerV2Initialization:
    """Test initialization and feature gating."""

    def test_default_shadow_disabled(self):
        """Shadow mode should be disabled by default."""
        manager = TransitionManagerV2()
        assert manager.enable_shadow is False
        assert manager.shadow_table is None
        assert manager.shadow_comparator is None

    def test_explicit_enable_shadow(self):
        """Can explicitly enable shadow mode."""
        manager = TransitionManagerV2(enable_shadow_table=True)
        assert manager.enable_shadow is True
        assert manager.shadow_table is not None
        assert manager.shadow_comparator is not None

    def test_explicit_disable_overrides_env(self, monkeypatch):
        """Explicit disable should override environment variable."""
        monkeypatch.setenv("SOULLINK_ENABLE_TRANSITION_TABLE_SHADOW", "1")
        manager = TransitionManagerV2(enable_shadow_table=False)
        assert manager.enable_shadow is False

    def test_legacy_manager_always_initialized(self):
        """Legacy manager should always be present."""
        manager_no_shadow = TransitionManagerV2(enable_shadow_table=False)
        manager_with_shadow = TransitionManagerV2(enable_shadow_table=True)

        assert manager_no_shadow.legacy_manager is not None
        assert manager_with_shadow.legacy_manager is not None


class TestLegacyAuthorityPreserved:
    """Test that legacy behavior is never affected by shadow mode."""

    def test_legacy_result_returned_without_shadow(self):
        """Without shadow, should return legacy result directly."""
        manager = TransitionManagerV2(enable_shadow_table=False)

        decision = ModeDecision(
            mode="work",
            submode="explicit_task",
            confidence=0.9,
            reason="test",
            safety_flags=[],
            signals={"explicit_task_request": True, "normalized_text": "test"},
        )

        result = manager.transition(
            previous_mode="daily",
            decision=decision,
            enable_active_sex=True,
        )

        assert result.active_mode == "work"
        assert result.transition == "daily->work"

    def test_legacy_result_returned_with_shadow(self):
        """With shadow enabled, should still return legacy result."""
        manager = TransitionManagerV2(enable_shadow_table=True, enable_shadow_logging=False)

        decision = ModeDecision(
            mode="work",
            submode="explicit_task",
            confidence=0.9,
            reason="test",
            safety_flags=[],
            signals={"explicit_task_request": True, "normalized_text": "test"},
        )

        result = manager.transition(
            previous_mode="daily",
            decision=decision,
            enable_active_sex=True,
        )

        # Result should be identical to legacy
        assert result.active_mode == "work"
        assert result.transition == "daily->work"

    def test_crisis_guard_legacy_behavior(self):
        """Crisis guard should force daily mode (legacy behavior)."""
        manager = TransitionManagerV2(enable_shadow_table=True, enable_shadow_logging=False)

        decision = ModeDecision(
            mode="work",
            submode="explicit_task",
            confidence=0.9,
            reason="test",
            safety_flags=["crisis_guard"],
            signals={"normalized_text": "test"},
        )

        result = manager.transition(
            previous_mode="work",
            decision=decision,
            enable_active_sex=True,
        )

        # Crisis guard should force daily
        assert result.active_mode == "daily"
        assert "crisis" in result.reason.lower()

    def test_sex_gate_fail_legacy_behavior(self):
        """Sex gate failure should block entry (legacy behavior)."""
        manager = TransitionManagerV2(enable_shadow_table=True, enable_shadow_logging=False)

        decision = ModeDecision(
            mode="sex",
            submode="explicit_progression",
            confidence=0.9,
            reason="test",
            safety_flags=[],
            signals={"normalized_text": "test"},
        )

        result = manager.transition(
            previous_mode="work",
            decision=decision,
            desire_tier="restrained",  # Gate fails
            enable_active_sex=True,
        )

        # Gate should block entry
        assert result.active_mode == "daily"
        assert "blocked" in result.transition or "restrained" in result.reason


class TestShadowComparison:
    """Test shadow comparison recording and statistics."""

    def test_shadow_comparison_recorded(self):
        """Shadow comparison should be recorded when enabled."""
        manager = TransitionManagerV2(enable_shadow_table=True, enable_shadow_logging=False)

        decision = ModeDecision(
            mode="work",
            submode="explicit_task",
            confidence=0.9,
            reason="test",
            safety_flags=[],
            signals={"explicit_task_request": True, "normalized_text": "test"},
        )

        manager.transition(
            previous_mode="daily",
            decision=decision,
            enable_active_sex=True,
        )

        summary = manager.get_shadow_summary()
        assert summary is not None
        assert summary["total_decisions"] == 1

    def test_multiple_transitions_accumulate(self):
        """Multiple transitions should accumulate in shadow stats."""
        manager = TransitionManagerV2(enable_shadow_table=True, enable_shadow_logging=False)

        decisions = [
            ModeDecision(
                mode="work", submode="explicit_task", confidence=0.9, reason="test",
                safety_flags=[], signals={"explicit_task_request": True, "normalized_text": "test"}
            ),
            ModeDecision(
                mode="daily", submode=None, confidence=0.7, reason="test",
                safety_flags=[], signals={"normalized_text": "test"}
            ),
            ModeDecision(
                mode="work", submode="explicit_task", confidence=0.9, reason="test",
                safety_flags=[], signals={"explicit_task_request": True, "normalized_text": "test"}
            ),
        ]

        for decision in decisions:
            manager.transition(
                previous_mode="daily",
                decision=decision,
                enable_active_sex=True,
            )

        summary = manager.get_shadow_summary()
        assert summary["total_decisions"] == 3

    def test_reset_shadow_stats(self):
        """Reset should clear shadow statistics."""
        manager = TransitionManagerV2(enable_shadow_table=True, enable_shadow_logging=False)

        decision = ModeDecision(
            mode="work", submode="explicit_task", confidence=0.9, reason="test",
            safety_flags=[], signals={"explicit_task_request": True, "normalized_text": "test"}
        )

        manager.transition(previous_mode="daily", decision=decision, enable_active_sex=True)
        assert manager.get_shadow_summary()["total_decisions"] == 1

        manager.reset_shadow_stats()
        assert manager.get_shadow_summary()["total_decisions"] == 0

    def test_shadow_summary_none_when_disabled(self):
        """Shadow summary should be None when shadow disabled."""
        manager = TransitionManagerV2(enable_shadow_table=False)
        assert manager.get_shadow_summary() is None

    def test_disagreements_accessible(self):
        """Shadow disagreements should be accessible for analysis."""
        manager = TransitionManagerV2(enable_shadow_table=True, enable_shadow_logging=False)

        # Create a decision that might have different results in shadow
        decision = ModeDecision(
            mode="daily", submode=None, confidence=0.6, reason="test",
            safety_flags=[], signals={"normalized_text": "ok", "message_length": 2}
        )

        manager.transition(previous_mode="work", decision=decision, enable_active_sex=True)

        # Can access disagreements (may or may not have any)
        disagreements = manager.get_shadow_disagreements()
        assert isinstance(disagreements, list)


class TestForceEventMapping:
    """Test mapping of signals to ForceEvent enum."""

    def test_crisis_guard_mapped(self):
        """Crisis guard should map to CRISIS force event."""
        manager = TransitionManagerV2(enable_shadow_table=True, enable_shadow_logging=False)

        decision = ModeDecision(
            mode="work", submode=None, confidence=0.9, reason="test",
            safety_flags=["crisis_guard"],
            signals={"normalized_text": "help"}
        )

        result = manager.transition(previous_mode="work", decision=decision)

        # Verify crisis handled correctly
        assert result.active_mode == "daily"

        # Should have recorded comparison
        summary = manager.get_shadow_summary()
        assert summary["total_decisions"] == 1

    def test_explicit_task_mapped(self):
        """Explicit task should map to EXPLICIT_TASK force event."""
        manager = TransitionManagerV2(enable_shadow_table=True, enable_shadow_logging=False)

        decision = ModeDecision(
            mode="work", submode="explicit_task", confidence=0.9, reason="test",
            safety_flags=[],
            signals={"explicit_task_request": True, "normalized_text": "test"}
        )

        result = manager.transition(previous_mode="sex", decision=decision)

        # Should exit sex to work
        assert result.active_mode == "work"

    def test_scene_close_mapped(self):
        """Scene close should map to SCENE_CLOSE force event."""
        manager = TransitionManagerV2(enable_shadow_table=True, enable_shadow_logging=False)

        decision = ModeDecision(
            mode="daily", submode=None, confidence=0.8, reason="test",
            safety_flags=[],
            signals={"sex_scene_close": True, "normalized_text": "test"}
        )

        result = manager.transition(previous_mode="sex", decision=decision)

        # Should exit sex to daily
        assert result.active_mode == "daily"


class TestGateResultMapping:
    """Test mapping of desire tier to GateResult enum."""

    def test_gate_pass_uninhibited(self):
        """Uninhibited tier with active sex should pass gate."""
        manager = TransitionManagerV2(enable_shadow_table=True, enable_shadow_logging=False)

        decision = ModeDecision(
            mode="sex", submode="explicit_progression", confidence=0.9, reason="test",
            safety_flags=[], signals={"normalized_text": "test"}
        )

        result = manager.transition(
            previous_mode="work",
            decision=decision,
            desire_tier="uninhibited",
            enable_active_sex=True,
        )

        # Should allow entry
        assert result.active_mode == "sex"

    def test_gate_fail_restrained(self):
        """Restrained tier should fail gate."""
        manager = TransitionManagerV2(enable_shadow_table=True, enable_shadow_logging=False)

        decision = ModeDecision(
            mode="sex", submode="explicit_progression", confidence=0.9, reason="test",
            safety_flags=[], signals={"normalized_text": "test"}
        )

        result = manager.transition(
            previous_mode="work",
            decision=decision,
            desire_tier="restrained",
            enable_active_sex=True,
        )

        # Should block entry
        assert result.active_mode == "daily"

    def test_gate_not_applicable_for_non_sex(self):
        """Gate should not apply to non-sex nominations."""
        manager = TransitionManagerV2(enable_shadow_table=True, enable_shadow_logging=False)

        decision = ModeDecision(
            mode="work", submode="explicit_task", confidence=0.9, reason="test",
            safety_flags=[],
            signals={"explicit_task_request": True, "normalized_text": "test"}
        )

        result = manager.transition(
            previous_mode="daily",
            decision=decision,
            desire_tier="restrained",  # Should not affect work
            enable_active_sex=True,
        )

        # Work should succeed regardless of desire tier
        assert result.active_mode == "work"


class TestBackwardCompatibility:
    """Test that the module provides backward-compatible exports."""

    def test_transitionmanager_exported(self):
        """TransitionManager should be exported for backward compatibility."""
        from packages.persona_engine.persona_orchestrator.transition_manager_v2 import TransitionManager

        # Should be the V2 version
        manager = TransitionManager()
        assert isinstance(manager, TransitionManagerV2)
        assert hasattr(manager, 'legacy_manager')
        assert hasattr(manager, 'enable_shadow')
