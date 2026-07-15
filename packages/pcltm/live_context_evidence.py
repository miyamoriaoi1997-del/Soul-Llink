"""Tool-output evidence compaction for PCLTM live context governance."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

from .live_context_governor import ToolEvidenceCapsule


def build_tool_evidence_capsules(
    tool_events: Sequence[Mapping[str, Any]],
    *,
    max_items: int = 6,
    max_total_chars: int = 1200,
) -> tuple[list[ToolEvidenceCapsule], dict[str, Any]]:
    """Build prompt-safe evidence capsules from raw tool events under a hard render budget."""
    if max_items <= 0 or max_total_chars <= 0:
        return [], {
            "tool_events": len(tool_events),
            "capsules": 0,
            "omitted_tool_events": len(tool_events),
            "rendered_chars": 0,
            "max_total_chars": max_total_chars,
            "within_budget": True,
            "hashes": [],
        }

    capsules: list[ToolEvidenceCapsule] = []
    rendered_chars = 0
    omitted = 0
    for event in tool_events:
        if len(capsules) >= max_items:
            omitted += 1
            continue
        capsule = _capsule_from_event(event)
        remaining = max_total_chars - rendered_chars
        rendered = capsule.render(max_chars=max(0, remaining))
        if not rendered or len(rendered) > remaining:
            omitted += 1
            continue
        capsules.append(capsule)
        rendered_chars += len(rendered)

    telemetry = {
        "tool_events": len(tool_events),
        "capsules": len(capsules),
        "omitted_tool_events": omitted,
        "rendered_chars": rendered_chars,
        "max_total_chars": max_total_chars,
        "within_budget": rendered_chars <= max_total_chars,
        "hashes": [capsule.evidence_hash for capsule in capsules],
    }
    return capsules, telemetry


def _capsule_from_event(event: Mapping[str, Any]) -> ToolEvidenceCapsule:
    tool = str(event.get("tool") or event.get("name") or "tool")
    command = str(event.get("command") or event.get("input") or tool)
    output = str(event.get("output") or event.get("result") or event.get("content") or "")
    exit_code = int(event.get("exit_code") or event.get("status_code") or 0)
    affected = event.get("affected_files") or event.get("files") or ()
    if isinstance(affected, (str, bytes)):
        affected_files = (str(affected),)
    else:
        affected_files = tuple(str(item) for item in affected or ())
    return ToolEvidenceCapsule.from_tool_output(
        command=f"{tool}: {command}" if command != tool else tool,
        exit_code=exit_code,
        output=output,
        affected_files=affected_files,
    )
