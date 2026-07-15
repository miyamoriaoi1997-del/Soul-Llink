from __future__ import annotations

import json
from collections.abc import Callable, Iterable, Mapping
from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True, slots=True)
class ToolSpec:
    name: str
    description: str
    parameters: Mapping[str, Any]
    handler: Callable[[Mapping[str, Any]], Any]

    def __post_init__(self) -> None:
        if not self.name.strip():
            raise ValueError("tool name is required")
        if not callable(self.handler):
            raise TypeError("tool handler must be callable")

    def schema(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "description": self.description,
            "parameters": dict(self.parameters),
        }


class ToolCatalog:
    def __init__(self, tools: Iterable[ToolSpec] = ()) -> None:
        self._tools: dict[str, ToolSpec] = {}
        for tool in tools:
            if tool.name in self._tools:
                raise ValueError(f"duplicate tool: {tool.name}")
            self._tools[tool.name] = tool

    def schemas(self) -> list[dict[str, Any]]:
        return [tool.schema() for tool in self._tools.values()]

    def call(self, name: str, arguments: Mapping[str, Any]) -> str:
        try:
            tool = self._tools[name]
        except KeyError:
            raise KeyError(f"unknown tool: {name}") from None
        result = tool.handler(arguments)
        return json.dumps(result, ensure_ascii=False, default=str)
