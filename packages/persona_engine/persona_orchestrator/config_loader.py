"""Load routing rules from YAML config instead of hardcoded Python lists.

Provides a singleton-style loader that reads config/routing_rules.yaml once
and exposes term lists, thresholds, and confidence values as plain Python
objects. Supports hot-reload for development via reload().
"""
from __future__ import annotations

import os
from pathlib import Path
from typing import Any

import yaml

_CONFIG_DIR = Path(__file__).resolve().parents[1] / "config"
_ROUTING_RULES_PATH = _CONFIG_DIR / "routing_rules.yaml"

# Allow override via environment variable for testing
_OVERRIDE_PATH = os.environ.get("PERSONA_ROUTING_RULES_PATH")
if _OVERRIDE_PATH:
    _ROUTING_RULES_PATH = Path(_OVERRIDE_PATH)


class RoutingConfig:
    """Parsed routing rules config with typed accessors."""

    def __init__(self, data: dict[str, Any]) -> None:
        self._data = data
        self.confidence: dict[str, float] = data.get("confidence", {})
        self.thresholds: dict[str, float | int] = data.get("thresholds", {})
        self._lexicon: dict[str, list] = data.get("lexicon", {})
        self._classifier: dict[str, list] = data.get("classifier", {})
        self._intent_rules: dict[str, Any] = data.get("intent_rules", {})
        self._fallback: dict[str, Any] = data.get("fallback_principles", {})

    # ── Lexicon accessors ──────────────────────────────────────────────────

    def lexicon(self, group: str) -> list[str]:
        """Get a lexicon term group as a flat list of strings."""
        return self._normalize_terms(self._lexicon.get(group, []))

    def lexicon_typed(self, group: str) -> list[dict[str, Any] | str]:
        """Get a lexicon term group with metadata preserved."""
        return self.typed_terms(self._lexicon.get(group, []))

    # ── Classifier term accessors ──────────────────────────────────────────

    def classifier_terms(self, group: str) -> list[str]:
        """Get a classifier term group as a flat list of strings."""
        return self._normalize_terms(self._classifier.get(group, []))

    def classifier_terms_typed(self, group: str) -> list[dict[str, Any] | str]:
        """Get a classifier term group with metadata preserved."""
        return self.typed_terms(self._classifier.get(group, []))

    # ── Intent rule accessors ──────────────────────────────────────────────

    def intent_rule(self, rule_name: str) -> dict[str, list[str]] | list[str]:
        """Get an intent rule. Returns dict of sub-groups or flat list."""
        raw = self._intent_rules.get(rule_name, {})
        if isinstance(raw, list):
            return self._normalize_terms(raw)
        if isinstance(raw, dict):
            return {k: self._normalize_terms(v) for k, v in raw.items()}
        return {}

    def intent_terms(self, rule_name: str) -> list[str]:
        """Get an intent rule as a flat list (for rules that are always flat)."""
        raw = self._intent_rules.get(rule_name, [])
        if isinstance(raw, list):
            return self._normalize_terms(raw)
        # If it's a dict, flatten all sub-group values into one list
        if isinstance(raw, dict):
            result: list[str] = []
            for v in raw.values():
                result.extend(self._normalize_terms(v))
            return result
        return []

    # ── Threshold accessors ────────────────────────────────────────────────

    def threshold(self, name: str, default: float = 0.0) -> float:
        """Get a threshold value by name."""
        return float(self.thresholds.get(name, default))

    def conf(self, name: str, default: float = 0.55) -> float:
        """Get a confidence value by name."""
        return float(self.confidence.get(name, default))

    # ── Fallback principle accessors ────────────────────────────────────────

    def fallback_terms(self, group: str) -> list[str]:
        """Get a fallback principle term group."""
        raw = self._fallback.get(group, [])
        if isinstance(raw, list):
            return self._normalize_terms(raw)
        return []

    def fallback_boost(self, name: str, default: float = 0.0) -> float:
        """Get a fallback confidence boost value."""
        return float(self._fallback.get(name, default))

    # ── Utilities ──────────────────────────────────────────────────────────

    @staticmethod
    def _normalize_terms(raw_list: list) -> list[str]:
        """Normalize term entries: plain strings pass through, dicts extract 'term'.

        Legacy accessor - flattens to strings for backward compatibility.
        Use typed_terms() to preserve metadata.
        """
        result: list[str] = []
        for item in raw_list:
            if isinstance(item, str):
                result.append(item)
            elif isinstance(item, dict) and "term" in item:
                result.append(item["term"])
            else:
                result.append(str(item))
        return result

    @staticmethod
    def typed_terms(raw_list: list) -> list[dict[str, Any] | str]:
        """Get terms with metadata preserved.

        Returns list where each item is either:
        - A plain string term
        - A dict with 'term' and optional 'weight', 'requires_context', etc.
        """
        result: list[dict[str, Any] | str] = []
        for item in raw_list:
            if isinstance(item, str):
                result.append(item)
            elif isinstance(item, dict):
                result.append(item)
            else:
                result.append(str(item))
        return result


# ── Module-level singleton ─────────────────────────────────────────────────

_config: RoutingConfig | None = None


def load_routing_config(force_reload: bool = False) -> RoutingConfig:
    """Load or return cached routing config."""
    global _config
    if _config is not None and not force_reload:
        return _config
    with open(_ROUTING_RULES_PATH, "r", encoding="utf-8") as f:
        data = yaml.safe_load(f)
    _config = RoutingConfig(data or {})
    return _config


def reload() -> RoutingConfig:
    """Force reload config from disk. Useful for development/testing."""
    return load_routing_config(force_reload=True)
