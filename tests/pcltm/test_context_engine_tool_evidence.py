from __future__ import annotations

from pcltm.context_engine import PCLTMContextEngine


def test_context_engine_compacts_valid_large_tool_result_into_evidence_capsule() -> None:
    messages = [
        {"role": "user", "content": "运行测试并汇报"},
        {
            "role": "assistant",
            "content": "",
            "tool_calls": [{"id": "pytest-call", "function": {"name": "terminal", "arguments": "{}"}}],
        },
        {
            "role": "tool",
            "tool_call_id": "pytest-call",
            "name": "terminal",
            "content": "TOKEN=super-secret-value\n" + ("noise\n" * 1000) + "317 passed in 9.10s\n",
        },
        {"role": "assistant", "content": "测试通过。"},
        {"role": "user", "content": "继续下一步"},
    ]

    context = PCLTMContextEngine(mode="work", tail_limit=20).build_shadow_context(messages)
    rendered = context.render()

    assert "tool_evidence_capsule" in rendered
    assert "317 passed" in rendered
    assert "super-secret-value" not in rendered
    assert "[REDACTED_SECRET]" in rendered
    assert "noise\nnoise\nnoise" not in rendered
    assert context.debug_sidecars["tool_evidence"]["tool_events"] == 1
    assert context.debug_sidecars["tool_evidence"]["capsules"] == 1
    tool_items = [item for item in context.items if item.role == "tool"]
    assert len(tool_items) == 1
    assert len(tool_items[0].content) < 700


def test_context_engine_legacy_memory_probe_is_bodyless_retired_status() -> None:
    context = PCLTMContextEngine(
        mode="work", debug_memory_probe=True,
    ).build_shadow_context([{"role": "user", "content": "probe sentinel body"}])

    probe = context.debug_sidecars["memory_selection_probe"]
    assert probe == {
        "status": "retired",
        "bodyless": True,
        "reason": "legacy_memory_selection_probe_not_runtime_authority",
    }
    assert "selected" not in probe
    assert "load_entries_baseline" not in probe
