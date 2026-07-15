from __future__ import annotations

import json
from collections.abc import Callable, Mapping
from typing import Any

SEARCH_SCHEMA = {
    "name": "soullink_memory_search",
    "description": "Search SoulLink/PCLTM long-term memory. Returns prompt-safe references/excerpts; use soullink_memory_open for full body.",
    "parameters": {
        "type": "object",
        "properties": {
            "query": {"type": "string", "description": "Search query."},
            "mode": {"type": "string", "description": "Optional persona mode such as work, daily, sex, or cron."},
            "limit": {"type": "integer", "description": "Maximum results, default 8."},
        },
        "required": ["query"],
    },
}

OPEN_SCHEMA = {
    "name": "soullink_memory_open",
    "description": "Open one SoulLink/PCLTM memory returned by soullink_memory_search.",
    "parameters": {
        "type": "object",
        "properties": {
            "memory_id": {"type": "string", "description": "Memory id returned by search."},
            "body_limit": {"type": "integer", "description": "Maximum body characters, default 4000."},
        },
        "required": ["memory_id"],
    },
}

REMEMBER_SCHEMA = {
    "name": "soullink_memory_remember",
    "description": "Write an explicit user preference or durable memory into SoulLink/PCLTM long-term memory.",
    "parameters": {
        "type": "object",
        "properties": {
            "content": {"type": "string", "description": "Memory content to persist."},
            "target": {"type": "string", "enum": ["user", "memory"], "description": "user for USER.md-style preferences; memory for MEMORY.md-style notes."},
        },
        "required": ["content"],
    },
}


class PCLTMMemoryTools:
    """Host-neutral schemas and dispatch for governed PCLTM memory operations."""

    def __init__(
        self,
        *,
        search: Callable[..., Any],
        open_memory: Callable[..., Any],
        remember: Callable[..., Any],
    ) -> None:
        self._search = search
        self._open_memory = open_memory
        self._remember = remember

    def schemas(self) -> list[dict[str, Any]]:
        return [SEARCH_SCHEMA, OPEN_SCHEMA, REMEMBER_SCHEMA]

    def call(self, tool_name: str, args: Mapping[str, Any]) -> str:
        if tool_name == SEARCH_SCHEMA["name"]:
            query = str(args.get("query") or "").strip()
            if not query:
                return self._error("query is required")
            limit = self._bounded_int(args.get("limit"), default=8, minimum=1, maximum=100)
            if limit is None:
                return self._error("limit must be an integer between 1 and 100")
            results = self._search(
                query=query,
                mode=args.get("mode"),
                layers=("pinned", "episodic"),
                limit=limit,
            )
            return self._json({"success": True, "results": results})

        if tool_name == OPEN_SCHEMA["name"]:
            memory_id = str(args.get("memory_id") or "").strip()
            if not memory_id:
                return self._error("memory_id is required")
            body_limit = self._bounded_int(args.get("body_limit"), default=4000, minimum=1, maximum=20000)
            if body_limit is None:
                return self._error("body_limit must be an integer between 1 and 20000")
            opened = self._open_memory(
                memory_id=memory_id,
                body_limit=body_limit,
            )
            return self._json({"success": True, "memory": opened})

        if tool_name == REMEMBER_SCHEMA["name"]:
            content = str(args.get("content") or "").strip()
            if not content:
                return self._error("content is required")
            target = str(args.get("target") or "memory")
            if target not in {"user", "memory"}:
                return self._error("target must be user or memory")
            ok = self._remember(target=target, action="add", content=content)
            return self._json({"success": bool(ok), "target": target})

        return self._error(f"unknown tool: {tool_name}")

    @staticmethod
    def _bounded_int(value: Any, *, default: int, minimum: int, maximum: int) -> int | None:
        if value is None:
            return default
        if type(value) is not int:
            return None
        parsed = value
        return parsed if minimum <= parsed <= maximum else None

    @staticmethod
    def _json(payload: Mapping[str, Any]) -> str:
        return json.dumps(dict(payload), ensure_ascii=False, default=str)

    @classmethod
    def _error(cls, message: str) -> str:
        return cls._json({"success": False, "error": message})
