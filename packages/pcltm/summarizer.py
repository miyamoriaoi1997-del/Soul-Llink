"""Summary generation helpers with lossless, deterministic fallback."""

from __future__ import annotations

import re
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

from .secret_policy import redact_secrets


_REASONING_BLOCK_RE = re.compile(
    r"(?is)\s*<(?:thinking|reasoning|analysis)>.*?</(?:thinking|reasoning|analysis)>\s*"
)
def strip_reasoning_blocks(text: str) -> str:
    """Remove provider/private reasoning wrappers from text before summarization."""
    return " ".join(_REASONING_BLOCK_RE.sub(" ", text).split())


def redact_secret_assignments(text: str) -> str:
    return redact_secrets(text)


@dataclass
class FallbackSummarizer:
    """Try detailed, aggressive, then deterministic summary generation."""

    detailed: Callable[[list[dict[str, Any]], int], str] | None = None
    aggressive: Callable[[list[dict[str, Any]], int], str] | None = None

    def summarize(self, events: list[dict[str, Any]], max_chars: int) -> str:
        for candidate in (self.detailed, self.aggressive):
            if candidate is None:
                continue
            try:
                text = strip_reasoning_blocks(str(candidate(events, max_chars))).strip()
            except Exception:
                continue
            text = redact_secret_assignments(text)
            if text and len(text) <= max_chars:
                return text
        return deterministic_summary(events, max_chars=max_chars)


def deterministic_summary(events: list[dict[str, Any]], *, max_chars: int) -> str:
    """Return a bounded no-LLM summary that preserves evidence pointers."""
    parts = []
    for event in events:
        role = event.get("role") or "unknown"
        source = event.get("source") or "unknown"
        mode = event.get("persona_mode") or "unknown"
        content = redact_secret_assignments(strip_reasoning_blocks(str(event.get("content") or "")))
        excerpt = _head_tail(content, max(24, max_chars // max(1, len(events)) - 32))
        parts.append(f"#{event.get('event_id')} {role}/{source}/{mode}: {excerpt}")
    text = " | ".join(parts) or "empty summary"
    return _head_tail(text, max_chars)


def _head_tail(text: str, max_chars: int) -> str:
    if len(text) <= max_chars:
        return text
    if max_chars <= 1:
        return text[:max_chars]
    if max_chars <= 12:
        return text[: max_chars - 1] + "…"
    head_len = max_chars // 2 - 1
    tail_len = max_chars - head_len - 1
    return text[:head_len] + "…" + text[-tail_len:]
