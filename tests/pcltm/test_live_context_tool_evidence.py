from __future__ import annotations

from pcltm.live_context_evidence import build_tool_evidence_capsules
from pcltm.live_context_governor import ContextBudgetPolicy, govern_prompt_context


def test_build_tool_evidence_capsules_redacts_hashes_and_prioritizes_failures() -> None:
    capsules, telemetry = build_tool_evidence_capsules(
        [
            {
                "tool": "terminal",
                "command": "pytest tests -q",
                "exit_code": 1,
                "output": "TOKEN=super-secret-value\n" + ("noise\n" * 200) + "FAILED tests/test_x.py::test_y\nTraceback: boom\n",
                "affected_files": ["tests/test_x.py"],
            }
        ],
        max_items=4,
        max_total_chars=400,
    )

    assert len(capsules) == 1
    rendered = capsules[0].render(max_chars=400)
    assert "super-secret-value" not in rendered
    assert "[REDACTED_SECRET]" in rendered
    assert "FAILED" in rendered
    assert "Traceback" in rendered
    assert capsules[0].evidence_hash.startswith("sha256:")
    assert telemetry["tool_events"] == 1
    assert telemetry["capsules"] == 1
    assert telemetry["omitted_tool_events"] == 0
    assert telemetry["within_budget"] is True


def test_build_tool_evidence_capsules_omits_events_over_budget() -> None:
    events = [
        {"tool": "terminal", "command": f"cmd {idx}", "exit_code": 0, "output": f"{idx} passed\n" + ("x" * 500)}
        for idx in range(8)
    ]

    capsules, telemetry = build_tool_evidence_capsules(events, max_items=6, max_total_chars=260)

    assert 1 <= len(capsules) < len(events)
    assert telemetry["tool_events"] == 8
    assert telemetry["capsules"] == len(capsules)
    assert telemetry["omitted_tool_events"] > 0
    assert telemetry["within_budget"] is True
    assert telemetry["rendered_chars"] <= 260


def test_tool_evidence_capsules_feed_live_context_governor_without_bloat() -> None:
    capsules, evidence_telemetry = build_tool_evidence_capsules(
        [
            {"tool": "read_file", "command": "read_file huge.log", "exit_code": 0, "output": "line\n" * 1000},
            {"tool": "terminal", "command": "pytest -q", "exit_code": 0, "output": "312 passed in 6.91s"},
        ],
        max_items=4,
        max_total_chars=320,
    )

    governed = govern_prompt_context(
        "<pcltm_context>\nselected memory\n</pcltm_context>",
        policy=ContextBudgetPolicy(total_chars=700, evidence_chars=320, memory_chars=220),
        tool_evidence=capsules,
        outer_tag="pcltm_context",
    )

    assert governed.telemetry["within_budget"] is True
    assert governed.telemetry["capsules"]["tool_evidence"] == len(capsules)
    assert evidence_telemetry["within_budget"] is True
    assert "tool_evidence_capsule" in governed.rendered
    assert "312 passed" in governed.rendered
    assert len(governed.rendered) <= 700
