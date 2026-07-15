from __future__ import annotations

import json
import sys

from pcltm.host_tools import PCLTMMemoryTools


def test_pcltm_memory_tools_are_host_neutral_and_dispatch_injected_backend() -> None:
    forbidden_before = {name for name in sys.modules if name.split(".", 1)[0] in {"agent", "gateway", "hermes_cli"}}
    calls = []
    tools = PCLTMMemoryTools(
        search=lambda **kwargs: calls.append(("search", kwargs)) or [{"memory_id": "m1"}],
        open_memory=lambda **kwargs: calls.append(("open", kwargs)) or {"memory_id": "m1", "body": "text"},
        remember=lambda **kwargs: calls.append(("remember", kwargs)) or True,
    )

    assert [schema["name"] for schema in tools.schemas()] == [
        "soullink_memory_search",
        "soullink_memory_open",
        "soullink_memory_remember",
    ]
    assert json.loads(tools.call("soullink_memory_search", {"query": "alpha", "limit": 3})) == {
        "success": True,
        "results": [{"memory_id": "m1"}],
    }
    assert json.loads(tools.call("soullink_memory_open", {"memory_id": "m1"})) == {
        "success": True,
        "memory": {"memory_id": "m1", "body": "text"},
    }
    assert json.loads(tools.call("soullink_memory_remember", {"content": "stable", "target": "user"})) == {
        "success": True,
        "target": "user",
    }
    assert calls == [
        ("search", {"query": "alpha", "mode": None, "layers": ("pinned", "episodic"), "limit": 3}),
        ("open", {"memory_id": "m1", "body_limit": 4000}),
        ("remember", {"target": "user", "action": "add", "content": "stable"}),
    ]
    forbidden_after = {name for name in sys.modules if name.split(".", 1)[0] in {"agent", "gateway", "hermes_cli"}}
    assert forbidden_after == forbidden_before


def test_pcltm_memory_tools_validate_required_fields_and_unknown_names() -> None:
    tools = PCLTMMemoryTools(search=lambda **_: [], open_memory=lambda **_: {}, remember=lambda **_: True)

    assert json.loads(tools.call("soullink_memory_search", {}))["error"] == "query is required"
    assert json.loads(tools.call("soullink_memory_search", {"query": "x", "limit": "bad"}))["error"] == "limit must be an integer between 1 and 100"
    assert json.loads(tools.call("soullink_memory_open", {}))["error"] == "memory_id is required"
    assert json.loads(tools.call("soullink_memory_open", {"memory_id": "m1", "body_limit": -1}))["error"] == "body_limit must be an integer between 1 and 20000"
    assert json.loads(tools.call("soullink_memory_open", {"memory_id": "m1", "body_limit": "bad"}))["error"] == "body_limit must be an integer between 1 and 20000"
    assert json.loads(tools.call("soullink_memory_open", {"memory_id": "m1", "body_limit": 1e400}))["error"] == "body_limit must be an integer between 1 and 20000"
    assert json.loads(tools.call("soullink_memory_open", {"memory_id": "m1", "body_limit": 3.5}))["error"] == "body_limit must be an integer between 1 and 20000"
    assert json.loads(tools.call("soullink_memory_remember", {}))["error"] == "content is required"
    assert json.loads(tools.call("soullink_memory_remember", {"content": "x", "target": "other"}))["error"] == "target must be user or memory"
    assert json.loads(tools.call("missing", {}))["error"] == "unknown tool: missing"
