from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import pcltm.memory_adapter as memory_adapter
import pcltm.transcript_search as transcript_search


def _load_provider_class():
    plugin_path = Path(__file__).resolve().parents[1] / "adapters/hermes/memory_provider/__init__.py"
    spec = importlib.util.spec_from_file_location("soullink_tool_adapter_under_test", plugin_path)
    module = importlib.util.module_from_spec(spec)
    assert spec and spec.loader
    spec.loader.exec_module(module)
    return module.SoulLinkMemoryProvider


def test_hermes_provider_binds_shared_pcltm_memory_tools(monkeypatch) -> None:
    calls = []
    monkeypatch.setattr(memory_adapter, "search_archival_memories", lambda **kwargs: calls.append(("search", kwargs)) or [])
    monkeypatch.setattr(memory_adapter, "open_archival_memory", lambda **kwargs: calls.append(("open", kwargs)) or {"memory_id": "m1"})
    monkeypatch.setattr(memory_adapter, "sync_memory_tool_write", lambda **kwargs: calls.append(("remember", kwargs)) or True)
    monkeypatch.setattr(transcript_search, "search_exact_evidence", lambda store, query, limit: calls.append(("exact", {"query": query, "limit": limit})) or [])
    provider = _load_provider_class()()

    assert [schema["name"] for schema in provider.get_tool_schemas()] == [
        "soullink_memory_search",
        "soullink_memory_recall_exact",
        "soullink_memory_open",
        "soullink_memory_remember",
        "soullink_identity_status",
    ]
    assert json.loads(provider.handle_tool_call("soullink_memory_search", {"query": "alpha"}))["success"] is True
    exact = json.loads(provider.handle_tool_call("soullink_memory_recall_exact", {"query": "alpha"}))
    assert exact["success"] is True
    assert exact["results"] == []
    assert json.loads(provider.handle_tool_call("soullink_memory_open", {"memory_id": "m1"}))["success"] is True
    assert json.loads(provider.handle_tool_call("soullink_memory_remember", {"content": "stable"}))["success"] is True
    assert calls == [
        ("search", {"query": "alpha", "mode": None, "layers": ("pinned", "episodic"), "limit": 8}),
        ("exact", {"query": "alpha", "limit": 8}),
        ("open", {"memory_id": "m1", "body_limit": 4000}),
        ("remember", {"target": "memory", "action": "add", "content": "stable"}),
    ]
