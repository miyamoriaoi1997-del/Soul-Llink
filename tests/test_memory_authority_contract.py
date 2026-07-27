from __future__ import annotations

import json

from pcltm.host_tools import PCLTMMemoryTools
from pcltm.memory_authority import UNAVAILABLE_RESULT, unavailable_result
from soul_link.hermes_plugin.memory_provider import SoulLinkMemoryProvider


def test_new_session_prompt_exposes_soullink_memory_authority_first() -> None:
    block = SoulLinkMemoryProvider().system_prompt_block()

    assert block.startswith("SoulLink/PCLTM memory authority contract:")
    assert "canonical authority" in block
    assert "cross-session recall" in block
    assert "durable writes" in block
    assert "Recall questions: call soullink_memory_search, soullink_memory_recall_exact, or soullink_memory_open first." in block
    assert "Persistent facts or derived memories: call soullink_memory_remember." in block
    assert "Hermes built-in memory and session_search are legacy/non-authoritative for this profile; never use them as fallback." in block
    assert "If SoulLink is unavailable, report unavailable; do not silently fall back." in block


def test_memory_tool_schemas_make_soullink_priority_and_non_fallback_explicit() -> None:
    tools = PCLTMMemoryTools(
        search=lambda **_: [],
        open_memory=lambda **_: {},
        remember=lambda **_: True,
        recall_exact=lambda **_: [],
    )

    schemas = tools.schemas()
    assert [schema["name"] for schema in schemas] == [
        "soullink_memory_search",
        "soullink_memory_recall_exact",
        "soullink_memory_open",
        "soullink_memory_remember",
    ]
    descriptions = " ".join(schema["description"] for schema in schemas)
    assert "preferred first authority" in descriptions.lower()
    assert "no hermes memory fallback" in descriptions.lower()


def test_unavailable_envelope_is_immutable_and_each_result_is_fresh() -> None:
    try:
        UNAVAILABLE_RESULT["status"] = "corrupted"  # type: ignore[index]
    except TypeError:
        pass
    else:
        raise AssertionError("shared unavailable contract must be immutable")

    first = unavailable_result()
    second = unavailable_result()
    first["status"] = "changed"
    assert second["status"] == "unavailable"
    assert first is not second


def test_memory_tools_fail_closed_as_unavailable_without_native_fallback() -> None:
    def unavailable(**_: object) -> object:
        raise RuntimeError("PCLTM backend is offline")

    tools = PCLTMMemoryTools(
        search=unavailable,
        open_memory=unavailable,
        remember=unavailable,
        recall_exact=unavailable,
    )

    for tool_name, args in (
        ("soullink_memory_search", {"query": "preference"}),
        ("soullink_memory_recall_exact", {"query": "exact"}),
        ("soullink_memory_open", {"memory_id": "m1"}),
        ("soullink_memory_remember", {"content": "durable fact"}),
    ):
        result = json.loads(tools.call(tool_name, args))
        assert result == {
            "success": False,
            "status": "unavailable",
            "authority": "soullink/pcltm",
            "fallback": "forbidden",
        }
