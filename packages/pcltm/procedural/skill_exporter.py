"""Export PCLTM procedural memory into Hermes skill operations.

PCLTM decides what is worth retaining as procedural knowledge. Hermes skills
carry the reusable procedure. This module returns skill_manage-compatible
operation payloads instead of mutating the Hermes skill library directly, so the
caller can review, confirm, or execute them with the existing Hermes tool.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum
from typing import Mapping

from .procedural_memory import ProceduralMemory, SkillUpdateAction


class ExistingSkillStatus(StrEnum):
    """Caller-supplied status of the matching Hermes skill."""

    MISSING = "missing"
    CURRENT = "current"
    OUTDATED = "outdated"
    WRONG = "wrong"


@dataclass(frozen=True)
class SkillExportPlan:
    """A proposed Hermes skill operation derived from procedural memory."""

    action: SkillUpdateAction
    memory: ProceduralMemory
    reason: str
    content: str | None = None
    old_string: str | None = None
    new_string: str | None = None


class SkillExporter:
    """Plan Hermes skill creation or patching from PCLTM procedural records."""

    def plan(
        self,
        memory: ProceduralMemory,
        *,
        existing_skill: str | None = None,
        existing_status: ExistingSkillStatus = ExistingSkillStatus.MISSING,
    ) -> SkillExportPlan:
        """Return the skill_manage operation that should happen next.

        Rules aligned with Hermes skill governance:
        * PCLTM records what should be preserved; Hermes skills carry the flow.
        * Non-exportable/task-log-shaped records are skipped.
        * Missing skills are created.
        * Outdated or wrong skills are patched, not duplicated.
        * Current skills are left alone.
        """

        if not memory.is_exportable:
            return SkillExportPlan(
                action=SkillUpdateAction.SKIP,
                memory=memory,
                reason="procedural memory is not exportable or looks like a task log",
            )

        rendered = self.render_skill(memory)
        if existing_status == ExistingSkillStatus.MISSING or not existing_skill:
            return SkillExportPlan(
                action=SkillUpdateAction.CREATE,
                memory=memory,
                reason="no existing Hermes skill; create reusable procedure",
                content=rendered,
            )

        if existing_status == ExistingSkillStatus.CURRENT:
            return SkillExportPlan(
                action=SkillUpdateAction.SKIP,
                memory=memory,
                reason="existing Hermes skill is current",
            )

        if existing_status in {ExistingSkillStatus.OUTDATED, ExistingSkillStatus.WRONG}:
            anchor = self._patch_anchor(existing_skill)
            return SkillExportPlan(
                action=SkillUpdateAction.PATCH,
                memory=memory,
                reason=f"existing Hermes skill is {existing_status.value}; patch with reusable procedure",
                old_string=anchor,
                new_string=self._render_body(memory),
            )

        return SkillExportPlan(
            action=SkillUpdateAction.SKIP,
            memory=memory,
            reason="unrecognized existing skill status",
        )

    def render_skill(self, memory: ProceduralMemory) -> str:
        """Render a complete Hermes SKILL.md document."""

        body = self._render_body(memory)
        return (
            "---\n"
            f"name: {memory.skill_name}\n"
            f"description: Use when {memory.trigger_conditions[0]}\n"
            f"category: {memory.category}\n"
            "---\n\n"
            f"# {memory.skill_name}\n\n"
            f"{body}"
        )

    def _render_body(self, memory: ProceduralMemory) -> str:
        lines: list[str] = []
        lines.append("## Relationship to PCLTM")
        lines.append("PCLTM procedural memory records that this workflow should be retained; this Hermes skill carries the reusable procedure. Do not add one-off task progress here.")
        lines.append("")
        lines.append("## Trigger conditions")
        lines.extend(f"- {item}" for item in memory.trigger_conditions)
        lines.append("")
        lines.append("## Procedure")
        lines.extend(f"{index}. {item}" for index, item in enumerate(memory.procedure, start=1))
        lines.append("")
        lines.append("## Verification steps")
        lines.extend(f"- {item}" for item in memory.verification_steps)
        lines.append("")
        lines.append("## Pitfalls")
        if memory.pitfalls:
            lines.extend(f"- {item}" for item in memory.pitfalls)
        else:
            lines.append("- Do not store task progress, status reports, PR numbers, commits, or phase-completion notes as skill content.")
        lines.append("")
        lines.append("## Source sessions")
        if memory.source_sessions:
            lines.extend(f"- {item}" for item in memory.source_sessions)
        else:
            lines.append("- Not recorded")
        lines.append("")
        return "\n".join(lines)

    def _patch_anchor(self, existing_skill: str) -> str:
        """Choose a stable old_string for skill_manage patch operations."""

        marker = "## Relationship to PCLTM"
        if marker in existing_skill:
            return existing_skill[existing_skill.index(marker) :].rstrip()
        stripped = existing_skill.strip()
        return stripped if stripped else marker


def operation_for_skill_manage(plan: SkillExportPlan) -> Mapping[str, object] | None:
    """Convert an export plan to a skill_manage-compatible payload."""

    if plan.action == SkillUpdateAction.SKIP:
        return None
    if plan.action == SkillUpdateAction.CREATE:
        if plan.content is None:
            raise ValueError("create plan requires content")
        return {
            "action": "create",
            "name": plan.memory.skill_name,
            "category": plan.memory.category,
            "content": plan.content,
        }
    if plan.action == SkillUpdateAction.PATCH:
        if plan.old_string is None or plan.new_string is None:
            raise ValueError("patch plan requires old_string and new_string")
        return {
            "action": "patch",
            "name": plan.memory.skill_name,
            "old_string": plan.old_string,
            "new_string": plan.new_string,
        }
    raise ValueError(f"unsupported skill export action: {plan.action}")
