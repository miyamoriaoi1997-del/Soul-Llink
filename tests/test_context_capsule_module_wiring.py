from __future__ import annotations

from soul_link.hermes_plugin import context_engine
from soul_link.hermes_plugin import continuity_capsules, tool_capsules


def test_context_engine_uses_extracted_continuity_renderer() -> None:
    assert context_engine._continuity_line_impl is continuity_capsules.continuity_line


def test_context_engine_uses_extracted_tool_capsule_helpers() -> None:
    assert context_engine._tool_capsule_kind_impl is tool_capsules.tool_capsule_kind
    assert context_engine._render_tool_capsule_impl is tool_capsules.render_tool_capsule
    assert context_engine._tool_capsule_indicates_error_impl is tool_capsules.tool_capsule_indicates_error
