from __future__ import annotations

from soul_link.hermes_plugin.continuity_capsules import continuity_line


def test_continuity_line_formats_dialogue_and_tools() -> None:
    assert continuity_line({"role": "user", "content": "  continue   P1  "}) == "- user: continue P1"
    assert continuity_line({"role": "tool", "name": "pytest", "content": "30 passed"}) == "- tool[pytest]: 30 passed"


def test_continuity_line_only_retains_typed_system_memory() -> None:
    assert continuity_line({"role": "system", "content": "ordinary host prompt"}) == ""
    assert continuity_line({"role": "system", "content": "[PCLTM state] active"}) == "- system_memory: [PCLTM state] active"
