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

_MEMORY_COMMAND_RE = re.compile(
    r"^\[(?P<action>memory|replace|forget)(?::(?P<key>[a-z0-9][a-z0-9._-]{0,127}))?\]"
    r"(?P<value>.*)$",
    re.IGNORECASE | re.DOTALL,
)

_QUESTION_OR_UNCERTAIN_RE = re.compile(
    r"[？?]|(?:也许|可能|大概|或许|不确定|是不是|要不要|你觉得|我觉得呢)"
)
_TRANSIENT_OR_PROCESS_RE = re.compile(
    r"(?:今天|今晚|现在|当前|刚才|这次|本次|临时|测试|通过了|失败了|"
    r"ASYNC DELEGATION|active task list|tool result|context compression)",
    re.IGNORECASE,
)
_STABLE_PREFERENCE_RE = re.compile(
    r"^(?:我|本人)(?:长期|一直|一贯|始终|以后|今后|通常|默认)"
    r"(?:都|会|更)?(?:偏好|喜欢|不喜欢|讨厌|希望|习惯)(?P<value>.+)$"
)
_IDENTITY_FACT_RE = re.compile(
    r"^(?:我的)?(?P<field>名字|姓名|称呼|生日|出生日期|职业|家乡|常住地|居住地)"
    r"(?:是|为|叫)(?P<value>.+)$"
)
_LONG_TERM_CONVENTION_RE = re.compile(
    r"^(?:以后|今后|从今以后|长期|始终|默认)(?P<value>.+)"
    r"(?:不要|不得|必须|一律|统一|都要|都不|默认)(?P<tail>.*)$"
)


@dataclass(frozen=True)
class StableMemoryAssertion:
    kind: str
    target_file: str
    content: str
    semantic_key: str


def _stable_key(value: str) -> str:
    import hashlib

    normalized = re.sub(r"\s+", "", value).casefold()
    normalized = re.sub(r"[，,、:：;；。.!！]", "", normalized)
    return hashlib.sha256(normalized.encode("utf-8")).hexdigest()[:24]


def parse_stable_memory_assertion(content: str) -> StableMemoryAssertion | None:
    """Recognize only explicit, stable natural-language memory assertions."""
    normalized = str(content or "").strip()
    if not normalized or len(normalized) > 500:
        return None
    normalized = normalized.rstrip("。.!！").strip()
    if (
        not normalized
        or _QUESTION_OR_UNCERTAIN_RE.search(normalized)
        or _TRANSIENT_OR_PROCESS_RE.search(normalized)
        or normalized.startswith("[")
    ):
        return None
    match = _STABLE_PREFERENCE_RE.fullmatch(normalized)
    if match is not None and match.group("value").strip():
        return StableMemoryAssertion(
            "user_preference", "user", normalized,
            "natural-preference:" + _stable_key(match.group("value")),
        )
    match = _IDENTITY_FACT_RE.fullmatch(normalized)
    if match is not None and match.group("value").strip():
        field = match.group("field")
        return StableMemoryAssertion(
            "identity_fact", "user", normalized, f"identity:{field}",
        )
    match = _LONG_TERM_CONVENTION_RE.fullmatch(normalized)
    if match is not None:
        value = (match.group("value") + match.group("tail")).strip()
        if value:
            return StableMemoryAssertion(
                "system_convention", "memory", normalized,
                "natural-convention:" + _stable_key(value),
            )
    return None


def parse_memory_command(content: str) -> tuple[str, str | None, str] | None:
    """Parse the bounded durable-memory protocol.

    ``memory`` may omit a key for backwards-compatible content addressing.
    ``replace`` and ``forget`` require a stable semantic key.  Forget carries no
    value; memory/replace require one.  Invalid forms are ordinary conversation.
    """
    match = _MEMORY_COMMAND_RE.fullmatch(str(content or "").strip())
    if match is None:
        return None
    action = match.group("action").casefold()
    key = match.group("key")
    value = match.group("value").strip()
    if action in {"replace", "forget"} and not key:
        return None
    if action == "forget":
        return (action, key, "") if not value else None
    return (action, key, value) if value else None


def is_durable_memory_assertion(content: str) -> bool:
    """Return whether content opts into automatic durable-memory admission.

    This deliberately tiny contract is lexical and model-free: only a message
    whose first non-whitespace token is ``[memory]`` is eligible.  Mode
    confidence never participates in this decision.
    """
    return parse_memory_command(content) is not None


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
        if role == "user" and source in {"chat", "hermes_state_db"} and (
            is_durable_memory_assertion(content)
            or parse_stable_memory_assertion(content) is not None
        ):
            return Classification(mode, mode, "candidate_only", sensitivity, 0.95, self.version)
        return Classification(mode, mode, "retrieve_only", sensitivity, 0.9, self.version)

    @staticmethod
    def _normalize_mode(persona_mode: str | None) -> str:
        mode = (persona_mode or "").lower()
        if mode in _STATE_MODES:
            return mode
        return "unknown"
