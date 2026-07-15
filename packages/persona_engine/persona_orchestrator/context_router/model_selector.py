"""Model selection mapping for the context router.

Maps routing decisions (top_mode + submode) to actual upstream model names.
Reads from the model-router runtime config at runtime.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml


DEFAULT_CONFIG_PATH = str(Path.home() / "soul-link" / "config" / "model-router runtime config")


class ModelSelector:
    """Map context route results to upstream model names."""

    def __init__(self, config_path: str | None = None):
        self.config_path = Path(config_path or DEFAULT_CONFIG_PATH)
        self._cache: dict[str, str] | None = None

    def select(self, top_mode: str, submode: str | None = None) -> str:
        """Return the model name for a given route decision."""
        mapping = self._load_mapping()
        key = f"{top_mode}:{submode}" if submode else top_mode
        return mapping.get(key) or mapping.get(top_mode) or mapping.get("default", "")

    def reload(self) -> None:
        """Force reload from disk."""
        self._cache = None

    def _load_mapping(self) -> dict[str, str]:
        if self._cache is not None:
            return self._cache

        try:
            raw = yaml.safe_load(self.config_path.read_text())
        except Exception:
            self._cache = self._fallback_mapping()
            return self._cache

        routing = raw.get("routing") or {}
        default_model = routing.get("default_model", "")
        work_model = routing.get("work_model") or default_model
        intimate_model = routing.get("sex_model", default_model)

        self._cache = {
            "work": work_model,
            "work:route_inspection": work_model,
            "work:code_change": work_model,
            "work:architecture_design": work_model,
            "work:fixture_authoring": work_model,
            "work:validation": work_model,
            "work:report_only": work_model,
            "relationship": default_model,
            "relationship:daily": default_model,
            "relationship:affectionate": default_model,
            "relationship:intimacy_candidate": default_model,
            "relationship:confirmed_intimacy": intimate_model,
            "relationship:cooldown": default_model,
            "relationship:aftercare": default_model,
            "default": default_model,
        }
        return self._cache

    @staticmethod
    def _fallback_mapping() -> dict[str, str]:
        return {"default": "claude-opus-4-6"}
