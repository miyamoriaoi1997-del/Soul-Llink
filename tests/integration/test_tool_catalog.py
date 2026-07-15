from __future__ import annotations

import json

import pytest

from soul_link.integration import ToolCatalog, ToolSpec


def test_tool_catalog_exposes_openai_schemas_and_dispatches_json_result() -> None:
    catalog = ToolCatalog(
        [
            ToolSpec(
                name="memory_search",
                description="Search memory",
                parameters={
                    "type": "object",
                    "properties": {"query": {"type": "string"}},
                    "required": ["query"],
                },
                handler=lambda args: {"ok": True, "query": args["query"]},
            )
        ]
    )

    assert catalog.schemas() == [
        {
            "name": "memory_search",
            "description": "Search memory",
            "parameters": {
                "type": "object",
                "properties": {"query": {"type": "string"}},
                "required": ["query"],
            },
        }
    ]
    assert json.loads(catalog.call("memory_search", {"query": "alpha"})) == {
        "ok": True,
        "query": "alpha",
    }


def test_tool_catalog_rejects_duplicate_and_unknown_tools() -> None:
    spec = ToolSpec(name="status", description="Status", parameters={"type": "object"}, handler=lambda _: {})
    with pytest.raises(ValueError, match="duplicate"):
        ToolCatalog([spec, spec])

    catalog = ToolCatalog([spec])
    with pytest.raises(KeyError, match="unknown"):
        catalog.call("missing", {})
