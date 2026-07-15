"""Write-time event classification for PCLTM.

PCLTM must not infer persona state from content. The persona state machine is the
single source of truth for memory classification. Domain words such as A-share,
ETF, gateway, QQBot, relationship, or cron remain ordinary text/query terms.
"""

from __future__ import annotations

import re
from dataclasses import dataclass


_SECRET_RE = re.compile(r"(?i)\b[A-Z0-9_]*(?:SECRET|TOKEN|API[_-]?KEY|PASSWORD|PASS)[A-Z0-9_]*\s*=")
_STATE_MODES = {
    "work",
    "system_maintenance",
    "daily",
    "intimacy",
    "conflict",
    "repair",
    "sex_candidate",
    "sex",
}


@dataclass(frozen=True)
class Classification:
    category: str
    subcategory: str
    inject_policy: str
    sensitivity: str
    confidence: float
    classifier_version: str = "state-machine-only-v3"


class EventClassifier:
    """Classifier constrained to the current persona state-machine mode only."""

    version = "state-machine-only-v3"

    def classify(
        self,
        *,
        role: str,
        source: str,
        content: str,
        persona_mode: str | None = None,
        sensitivity: str = "normal",
    ) -> Classification:
        mode = self._normalize_mode(persona_mode)

        if source == "cron":
            return Classification("unknown", "unknown", "no_memory", sensitivity, 1.0, self.version)
        if _SECRET_RE.search(content):
            return Classification(mode, mode, "retrieve_only", "secret" if sensitivity == "normal" else sensitivity, 1.0, self.version)
        if mode == "unknown":
            return Classification("unknown", "unknown", "retrieve_only", sensitivity, 0.2, self.version)
        if role == "user" and source == "chat":
            return Classification(mode, mode, "candidate_only", sensitivity, 0.95, self.version)
        return Classification(mode, mode, "retrieve_only", sensitivity, 0.9, self.version)

    @staticmethod
    def _normalize_mode(persona_mode: str | None) -> str:
        mode = (persona_mode or "").lower()
        if mode in _STATE_MODES:
            return mode
        return "unknown"
