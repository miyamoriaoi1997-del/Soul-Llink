"""Pure rendering helpers for typed PCLTM continuity capsules."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any


def _content_text(content: Any) -> str:
    if content is None:
        return ""
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts: list[str] = []
        for item in content:
            if isinstance(item, str):
                parts.append(item)
            elif isinstance(item, Mapping) and isinstance(item.get("text"), str):
                parts.append(str(item["text"]))
        return "\n".join(parts)
    return str(content)


def _truncate(text: str, limit: int) -> str:
    normalized = " ".join(str(text or "").split())
    return normalized if len(normalized) <= limit else normalized[: max(0, limit - 1)] + "…"


def continuity_line(message: Mapping[str, Any]) -> str:
    """Render one bounded continuity item without retaining ordinary system prompts."""
    role = str(message.get("role") or "unknown").lower()
    text = _content_text(message.get("content"))
    if not text:
        return ""
    text = _truncate(text, 900)
    if role == "tool":
        name = str(message.get("name") or message.get("tool_call_id") or "tool")
        return f"- tool[{name}]: {text}"
    if role == "system":
        if "[PCLTM" not in text and "<pcltm_context>" not in text:
            return ""
        return f"- system_memory: {_truncate(text, 700)}"
    return f"- {role}: {text}"


__all__ = ["continuity_line"]
