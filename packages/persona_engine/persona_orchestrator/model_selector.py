"""Dynamic Model Selector.

Selects the appropriate model based on routing context.
Priority chain (highest wins):
  1. context_result.selected_model (adapter already chose)
  2. mode_overrides from config
  3. emotion intensity override
  4. platform override
  5. None (caller uses system default)
"""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

import yaml

if TYPE_CHECKING:
    from persona_orchestrator.context_router.types import ContextRouteResult

# ─── Constants ───────────────────────────────────────────────────────────────

MODE_WORK = "work"
MODE_SEX = "sex"

DEFAULT_CONFIG = {
    "default_model": "persona-auto",
    "mode_overrides": {
        "work": None,
        "daily": None,
        "active_layer": None,
    },
    "platform_overrides": {
        "telegram": None,
        "cli": None,
    },
    "emotion_overrides": {
        "overwhelming": None,
        "intense": None,
        "moderate": None,
        "mild": None,
    },
    "model_switch_cooldown": 3,
}


# ─── Main Class ──────────────────────────────────────────────────────────────

class ModelSelector:
    """Dynamic model selection based on routing context.

    Encapsulates the priority chain and cooldown logic that was previously
    inline in StateOrchestrator._select_model.
    """

    def __init__(
        self,
        config_path: str | Path | None = None,
        config_dict: dict | None = None,
    ):
        if config_dict:
            self.config = config_dict
        elif config_path:
            self.config = self._load_config(Path(config_path))
        else:
            self.config = DEFAULT_CONFIG

        self._cooldown = self.config.get("model_switch_cooldown", 3)
        self._last_model: str | None = None
        self._turns_on_current_model: int = 99  # high initial = no block

    @staticmethod
    def _load_config(path: Path) -> dict:
        if not path.exists():
            return DEFAULT_CONFIG
        with open(path) as f:
            return yaml.safe_load(f) or DEFAULT_CONFIG

    def select(
        self,
        mode: str,
        submode: str | None = None,
        platform: str = "cli",
        emotion_score: float | None = None,
        context_result: "ContextRouteResult | None" = None,
    ) -> str | None:
        """Select model based on current context. Returns model name or None.

        Priority:
          1. context_result.selected_model
          2. mode_overrides
          3. emotion intensity override
          4. platform override
          5. None (use system default)
        """

        # Priority 1.5: Final active sex mode must own the model choice.
        # The context router's selected_model is computed before the final
        # transition layer resolves hold/continue behavior, so it can lag
        # behind when a scene stays in sex. In that case, honor the sex model
        # override directly instead of reusing a pre-transition default.
        mode_overrides = self.config.get("mode_overrides", {})
        mode_key = self._build_mode_key(mode, submode)
        if mode_key == "active_layer" and mode_overrides.get("active_layer"):
            candidate = mode_overrides["active_layer"]
            self._last_model = candidate
            self._turns_on_current_model = 0
            return candidate

        # Priority 1: Context router already selected a model via adapter.
        # This is the state-machine authority path and must not be blocked by
        # cooldown, otherwise the selector can lag behind a real mode switch.
        selected = getattr(context_result, "selected_model", None) if context_result else None
        if selected:
            self._last_model = selected
            self._turns_on_current_model = 0
            return selected

        # Priority 2: Mode override
        mode_overrides = self.config.get("mode_overrides", {})
        mode_key = self._build_mode_key(mode, submode)
        if mode_key in mode_overrides and mode_overrides[mode_key]:
            candidate = mode_overrides[mode_key]
            return self._apply_cooldown(candidate)

        # Priority 3: Emotion intensity override
        intensity = self._emotion_intensity(emotion_score)
        emotion_overrides = self.config.get("emotion_overrides", {})
        if intensity and intensity in emotion_overrides and emotion_overrides[intensity]:
            candidate = emotion_overrides[intensity]
            return self._apply_cooldown(candidate)

        # Priority 4: Platform override
        platform_overrides = self.config.get("platform_overrides", {})
        if platform in platform_overrides and platform_overrides[platform]:
            candidate = platform_overrides[platform]
            return self._apply_cooldown(candidate)

        # No override — increment turns on current model
        self._turns_on_current_model += 1
        return None

    def _apply_cooldown(self, candidate: str) -> str | None:
        """Apply model switch cooldown to prevent flapping."""
        if candidate == self._last_model:
            self._turns_on_current_model += 1
            return candidate

        # Different model requested — check cooldown
        if self._turns_on_current_model < self._cooldown:
            # Too soon to switch, keep current
            self._turns_on_current_model += 1
            return self._last_model

        # Allow switch
        self._last_model = candidate
        self._turns_on_current_model = 0
        return candidate

    def build_model_map(self) -> dict[str, str]:
        """Build model map for adapter.apply_model_selection.

        Maps config keys → router keys so the adapter can pick the right model
        based on context router's submode output.
        """
        mode_overrides = self.config.get("mode_overrides", {})
        default = self.config.get("default_model", "")
        result: dict[str, str] = {"default": default}
        for key, model in mode_overrides.items():
            if model:
                if key == "active_layer":
                    result["relationship:confirmed_intimacy"] = model
                    result["relationship:intimacy_candidate"] = model
                elif key == "work":
                    result["work"] = model
                elif key == "daily":
                    result["relationship"] = model
                    result["relationship:daily"] = model
                    result["relationship:affectionate"] = model
        return result

    # ─── Static Helpers ──────────────────────────────────────────────────────

    @staticmethod
    def _build_mode_key(mode: str, submode: str | None) -> str:
        """Map mode+submode to config key."""
        if mode == MODE_WORK:
            return "work"
        if mode == MODE_SEX:
            return "active_layer"
        return "daily"

    @staticmethod
    def _emotion_intensity(emotion_score: float | None) -> str | None:
        """Map emotion score to intensity band for override lookup."""
        if emotion_score is None:
            return None
        if -5.0 <= emotion_score <= 5.0:
            abs_score = abs(emotion_score)
            if abs_score >= 4.5:
                return "overwhelming"
            if abs_score >= 3.0:
                return "intense"
            if abs_score >= 1.5:
                return "moderate"
            return "mild"
        # Legacy percentage-style score
        if emotion_score >= 90:
            return "overwhelming"
        if emotion_score >= 60:
            return "intense"
        if emotion_score >= 30:
            return "moderate"
        return "mild"
