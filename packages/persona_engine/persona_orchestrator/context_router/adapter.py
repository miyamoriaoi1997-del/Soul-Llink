"""Adapter: bridges production state into abstract tags for the context router.

This is the 'first mile' — converts raw ModeDecision signals and emotion state
into the abstract input format the context router expects.

No raw text stored. No keyword lists. Only abstract label mapping.
"""

from __future__ import annotations

from typing import Any

from .types import AbstractInput, AbstractTurn, ContextRouteResult


class AbstractStateAdapter:
    """Convert production orchestrator state into abstract router input."""

    # ─── Tag Extraction (production signals → abstract tags) ──────────────

    def build_abstract_input(
        self,
        mode_decision: Any,
        previous_mode: str | None = None,
        previous_submode: str | None = None,
        turns_since_last_switch: int = 99,
        emotion_score: float | None = None,
        desire_tier: str | None = None,
        recent_decisions: list[Any] | None = None,
    ) -> AbstractInput:
        """Build abstract input from production ModeDecision and state."""
        current_tags = self._extract_tags_from_decision(mode_decision)
        current_abstract = self._extract_abstract_label(mode_decision)

        current_turn = AbstractTurn(
            abstract=current_abstract,
            tags=current_tags,
            role="user",
        )

        recent_history = self._build_history(recent_decisions)

        previous_state = {
            "top_mode": self._map_top_mode(previous_mode),
            "submode": previous_submode,
            "turns_since_last_switch": turns_since_last_switch,
        }

        affective_state = {
            "score_band": self._score_to_band(emotion_score),
            "desire_band": self._desire_to_band(desire_tier),
        }

        return AbstractInput(
            current_turn=current_turn,
            recent_history=recent_history,
            previous_state=previous_state,
            affective_state=affective_state,
        )

    # ─── Model Selection Mapping ──────────────────────────────────────────

    def apply_model_selection(
        self,
        result: ContextRouteResult,
        model_map: dict[str, str] | None = None,
    ) -> ContextRouteResult:
        """Map routing decision to actual model name."""
        if not model_map:
            model_map = self._default_model_map()

        top = result.top_mode
        sub = result.work_submode or result.relationship_submode or ""

        # Try specific submode first, then top mode
        key = f"{top}:{sub}" if sub else top
        model = model_map.get(key) or model_map.get(top) or model_map.get("default", "")

        result.selected_model = model
        result.model_reason = f"map:{key}" if model_map.get(key) else f"map:{top}"
        return result

    # ─── Internal: Tag Extraction ─────────────────────────────────────────

    def _extract_tags_from_decision(self, decision: Any) -> list[str]:
        """Extract abstract tags from a ModeDecision's signals."""
        tags: list[str] = []
        if decision is None:
            return tags

        signals = getattr(decision, "signals", {}) or {}
        mode = getattr(decision, "mode", "")
        submode = getattr(decision, "submode", "")
        safety_flags = getattr(decision, "safety_flags", []) or []

        # Meta/debug signals
        if signals.get("explicit_system_request") or "meta_discussion" in safety_flags:
            tags.append("task_debug")
        if signals.get("explicit_task_request"):
            tags.append("task_request")

        # Quoted/hypothetical detection
        if "meta_discussion" in safety_flags and self._has_relationship_marker(mode, submode, signals):
            tags.append("quoted_context")

        # Relationship markers (abstract strength only)
        marker = self._classify_marker_strength(mode, submode, signals)
        if marker:
            tags.append(marker)

        # Context signals
        if self._is_relationship_context(mode, submode, signals):
            tags.append("relationship_context")
        if signals.get("sex_scene_continue"):
            tags.append("continuation_cue")
        if self._is_progression(mode, submode, signals):
            tags.append("progression_setup")

        # Boundary signals
        if signals.get("safety_stop") or "crisis_guard" in safety_flags:
            tags.append("boundary_signal")
        # sex_scene_close is only a real boundary if not in work mode and no explicit task
        if signals.get("sex_scene_close") and mode != "work" and not signals.get("explicit_task_request") and not signals.get("explicit_system_request"):
            tags.append("boundary_signal")

        # Task context marker
        if "task_context" not in tags and mode == "work":
            tags.append("task_context")

        return tags

    def _extract_abstract_label(self, decision: Any) -> str:
        """Generate an abstract label for the current turn."""
        if decision is None:
            return "UNKNOWN"

        signals = getattr(decision, "signals", {}) or {}
        mode = getattr(decision, "mode", "")
        submode = getattr(decision, "submode", "")
        safety_flags = getattr(decision, "safety_flags", []) or []

        if "meta_discussion" in safety_flags:
            if self._has_relationship_marker(mode, submode, signals):
                return "ROUTE_DEBUG_WITH_REFERENCED_SCENE"
            return "SYSTEM_BEHAVIOR_QUESTION"

        if signals.get("explicit_task_request") or signals.get("explicit_system_request"):
            return "EXPLICIT_TASK_REQUEST"

        if signals.get("safety_stop") or "crisis_guard" in safety_flags:
            return "BOUNDARY_SIGNAL"

        if mode == "work":
            return "TASK_CONTEXT"

        if self._is_progression(mode, submode, signals):
            return "SCENE_ADVANCE_INDIRECT"

        if self._is_relationship_context(mode, submode, signals):
            return "RELATIONSHIP_CONTINUITY"

        return "NEUTRAL_INPUT"

    # ─── Internal: Helpers ────────────────────────────────────────────────

    def _build_history(self, recent_decisions: list[Any] | None) -> list[AbstractTurn]:
        """Convert recent ModeDecisions into abstract history turns."""
        if not recent_decisions:
            return []
        history: list[AbstractTurn] = []
        for dec in recent_decisions[-6:]:
            tags = self._extract_tags_from_decision(dec)
            abstract = self._extract_abstract_label(dec)
            role = getattr(dec, "role", "user") if hasattr(dec, "role") else "user"
            history.append(AbstractTurn(abstract=abstract, tags=tags, role=role))
        return history

    @staticmethod
    def _map_top_mode(mode: str | None) -> str:
        """Map production 3-state mode to router's 2-level top mode."""
        if mode == "work":
            return "work"
        return "relationship"

    @staticmethod
    def _score_to_band(score: float | None) -> str:
        """Map emotion score to abstract band."""
        if score is None:
            return "neutral"
        if -5.0 <= score <= 5.0:
            if score < 1.0:
                return "low"
            if score < 2.5:
                return "neutral"
            if score < 3.5:
                return "warm"
            return "high"
        # Legacy percentage-style score
        if score < 25:
            return "low"
        if score < 50:
            return "neutral"
        if score < 75:
            return "warm"
        return "high"

    @staticmethod
    def _desire_to_band(tier: str | None) -> str:
        """Map desire tier to abstract band."""
        mapping = {
            "restrained": "low",
            "ambivalent": "ambivalent",
            "uninhibited": "elevated",
            "unknown": "low",
        }
        return mapping.get(tier or "unknown", "low")

    @staticmethod
    def _has_relationship_marker(mode: str, submode: str, signals: dict) -> bool:
        """Check if decision contains any relationship-level marker."""
        if mode in ("daily",) and submode in ("relationship_closeness",):
            return True
        if signals.get("sex_scene_continue") or signals.get("aftercare_request"):
            return True
        return False

    @staticmethod
    def _classify_marker_strength(mode: str, submode: str, signals: dict) -> str | None:
        """Classify relationship marker into abstract strength level."""
        # Sex mode is always strong
        if mode == "sex":
            return "relationship_marker_strong"

        # Strong markers
        strong_submodes = ("explicit_progression",)
        if submode in strong_submodes:
            return "relationship_marker_strong"

        # Medium markers
        medium_submodes = ("hint_progression", "relationship_closeness")
        if submode in medium_submodes:
            return "relationship_marker_medium"

        # Weak markers (affectionate address without progression)
        if mode == "daily" and submode == "default":
            if signals.get("aftercare_request"):
                return "relationship_marker_weak"

        return None

    @staticmethod
    def _is_relationship_context(mode: str, submode: str, signals: dict) -> bool:
        """Check if current turn has relationship context."""
        if mode == "sex":
            return True
        if mode == "daily" and submode in ("relationship_closeness", "default"):
            return True
        if signals.get("sex_scene_continue") or signals.get("aftercare_request"):
            return True
        return False

    @staticmethod
    def _is_progression(mode: str, submode: str, signals: dict) -> bool:
        """Check if current turn advances closeness."""
        if mode == "sex":
            return True
        progression_submodes = ("explicit_progression", "hint_progression")
        if submode in progression_submodes:
            return True
        if signals.get("sex_scene_continue") and not signals.get("sex_scene_close"):
            return True
        return False

    @staticmethod
    def _default_model_map() -> dict[str, str]:
        """Default model selection map. Reads from router config at runtime."""
        return {
            "work": "claude-opus-4-6",
            "work:route_inspection": "claude-opus-4-6",
            "work:code_change": "claude-opus-4-6",
            "relationship": "claude-opus-4-6",
            "relationship:daily": "claude-opus-4-6",
            "relationship:affectionate": "claude-opus-4-6",
            "relationship:confirmed_intimacy": "claude-opus-4-6",
            "relationship:cooldown": "claude-opus-4-6",
            "default": "claude-opus-4-6",
        }
