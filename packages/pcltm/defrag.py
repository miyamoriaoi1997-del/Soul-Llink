"""Read-only memory defragmentation governance support for MemFS.

Safe memory cleanup planning without automatic production changes.
"""

from __future__ import annotations

import warnings
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml

from pcltm.memfs_store import MemFSStore
from pcltm.memfs_types import MemoryFileFrontmatter, MemoryTypePolicy


@dataclass
class DefragAction:
    """A single proposed action."""

    action_type: str  # "merge", "split", "move", "delete", "archive"
    source_path: str  # relative path
    target_path: str = ""  # for merge/move/split/archive
    reason: str = ""
    risk_level: str = "low"
    requires_human_review: bool = False
    authority_boundary: str = "memfs_defrag"
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "action_type": self.action_type,
            "source_path": self.source_path,
            "target_path": self.target_path,
            "reason": self.reason,
            "risk_level": self.risk_level,
            "requires_human_review": self.requires_human_review,
            "authority_boundary": self.authority_boundary,
            "metadata": dict(self.metadata),
        }


@dataclass
class DefragPlan:
    """A defragmentation plan (dry-run result)."""

    actions: list[DefragAction] = field(default_factory=list)
    total_files_analyzed: int = 0
    duplicate_count: int = 0
    oversized_count: int = 0
    stale_task_count: int = 0


@dataclass(frozen=True)
class DefragGovernanceReport:
    """Stable JSON contract for read-only defrag governance."""

    schema_version: int = 1
    dry_run: bool = True
    authority_boundary: str = "read_only_defrag_governance"
    actions: tuple[DefragAction, ...] = ()
    total_files_analyzed: int = 0
    merged_count: int = 0
    archived_count: int = 0
    decayed_count: int = 0
    conflict_count: int = 0
    stale_task_count: int = 0
    high_risk_changes: int = 0

    @classmethod
    def from_plan(cls, plan: DefragPlan) -> "DefragGovernanceReport":
        actions = tuple(plan.actions)
        return cls(
            actions=actions,
            total_files_analyzed=plan.total_files_analyzed,
            merged_count=sum(1 for action in actions if action.action_type == "merge"),
            archived_count=sum(1 for action in actions if action.action_type == "archive"),
            decayed_count=sum(1 for action in actions if action.action_type == "decay"),
            conflict_count=sum(1 for action in actions if action.action_type == "conflict"),
            stale_task_count=plan.stale_task_count,
            high_risk_changes=sum(1 for action in actions if action.risk_level == "high"),
        )

    @property
    def needs_review(self) -> bool:
        return any(action.requires_human_review for action in self.actions)

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "dry_run": self.dry_run,
            "authority_boundary": self.authority_boundary,
            "total_files_analyzed": self.total_files_analyzed,
            "merged_count": self.merged_count,
            "archived_count": self.archived_count,
            "decayed_count": self.decayed_count,
            "conflict_count": self.conflict_count,
            "stale_task_count": self.stale_task_count,
            "high_risk_changes": self.high_risk_changes,
            "needs_review": self.needs_review,
            "actions": [action.to_dict() for action in self.actions],
        }


class MemFSDefragger:
    """Analyze and optionally apply memory defragmentation."""

    def __init__(self, store: MemFSStore, *, size_threshold: int = 3000):
        if size_threshold < 1:
            raise ValueError("size_threshold must be >= 1")
        self.store = store
        self.size_threshold = size_threshold

    def analyze(self) -> DefragPlan:
        """Scan all memory files and produce a dry-run plan. Never modifies files."""
        plan = DefragPlan()
        bodies: dict[str, str] = {}

        for item in self.store.list_tree():
            try:
                frontmatter, body = self.store.read_file(item.path)
            except (OSError, ValueError, yaml.YAMLError) as exc:
                warnings.warn(f"skipping unreadable MemFS file {item.path}: {exc}", RuntimeWarning)
                continue

            plan.total_files_analyzed += 1
            normalized_body = self._normalize_body(body)
            risk_level, requires_review, boundary = self._risk_for(frontmatter, item.path)

            if normalized_body:
                original_path = bodies.get(normalized_body)
                if original_path is None:
                    bodies[normalized_body] = item.path
                else:
                    plan.duplicate_count += 1
                    plan.actions.append(
                        DefragAction(
                            action_type="merge",
                            source_path=original_path,
                            target_path=item.path,
                            reason="duplicate body content",
                            risk_level=risk_level,
                            requires_human_review=requires_review,
                            authority_boundary=boundary,
                            metadata={"memory_type": frontmatter.memory_type},
                        )
                    )

            if self._is_stale_temporary_task(frontmatter):
                plan.stale_task_count += 1
                plan.actions.append(
                    DefragAction(
                        action_type="archive",
                        source_path=item.path,
                        target_path=f"archived/{item.path}",
                        reason="temporary task memory has short ttl and should not remain active",
                        risk_level="medium" if requires_review else "low",
                        requires_human_review=True,
                        authority_boundary=boundary,
                        metadata={"memory_type": frontmatter.memory_type, "ttl": frontmatter.ttl},
                    )
                )

            if len(body) > self.size_threshold:
                plan.oversized_count += 1
                plan.actions.append(
                    DefragAction(
                        action_type="split",
                        source_path=item.path,
                        target_path=self._split_target_path(item.path),
                        reason=f"body length {len(body)} exceeds threshold {self.size_threshold}",
                        risk_level=risk_level,
                        requires_human_review=requires_review,
                        authority_boundary=boundary,
                        metadata={"memory_type": frontmatter.memory_type},
                    )
                )

        return plan

    def analyze_report(self) -> DefragGovernanceReport:
        """Return a stable read-only governance report for defrag analysis."""
        return DefragGovernanceReport.from_plan(self.analyze())

    def apply(self, plan: DefragPlan) -> None:
        """Apply a defrag plan. Modifies files in the store."""
        for action in plan.actions:
            try:
                if action.action_type == "merge":
                    # Merge is currently conservative: keep source, remove duplicate target.
                    if action.target_path:
                        self._delete_path(action.target_path)
                elif action.action_type == "split":
                    self._apply_split(action)
                elif action.action_type == "delete":
                    self._delete_path(action.source_path)
                elif action.action_type == "move":
                    self._apply_move(action)
                else:
                    warnings.warn(f"unknown defrag action type: {action.action_type}", RuntimeWarning)
            except Exception as exc:  # Defensive batch behavior: one bad action must not stop others.
                warnings.warn(
                    f"failed to apply defrag action {action.action_type} "
                    f"for {action.source_path}: {exc}",
                    RuntimeWarning,
                )
                continue

    def _apply_split(self, action: DefragAction) -> None:
        target_path = action.target_path or self._split_target_path(action.source_path)
        frontmatter, body = self.store.read_file(action.source_path)
        split_body = body[: self.size_threshold]
        remaining_body = body[self.size_threshold :]

        source = self._safe_path(action.source_path)
        target = self._safe_path(target_path)
        target.parent.mkdir(parents=True, exist_ok=True)

        target.write_text(self._render_memory_file(frontmatter, split_body), encoding="utf-8")
        source.write_text(self._render_memory_file(frontmatter, remaining_body), encoding="utf-8")

    def _apply_move(self, action: DefragAction) -> None:
        if not action.target_path:
            raise ValueError("move action requires target_path")
        source = self._safe_path(action.source_path)
        target = self._safe_path(action.target_path)
        target.parent.mkdir(parents=True, exist_ok=True)
        source.replace(target)

    def _delete_path(self, relative_path: str) -> None:
        path = self._safe_path(relative_path)
        if path.exists():
            path.unlink()

    def _safe_path(self, relative_path: str) -> Path:
        candidate = Path(relative_path)
        if candidate.is_absolute() or ".." in candidate.parts:
            raise ValueError(f"unsafe MemFS relative path: {relative_path!r}")
        root = self.store.root.resolve()
        resolved = (root / candidate).resolve()
        try:
            resolved.relative_to(root)
        except ValueError as exc:
            raise ValueError(f"path escapes MemFS root: {relative_path!r}") from exc
        return resolved

    def _split_target_path(self, relative_path: str) -> str:
        path = Path(relative_path)
        stem = path.stem
        suffix = path.suffix or ".md"
        return path.with_name(f"{stem}.part1{suffix}").as_posix()

    def _risk_for(self, frontmatter: MemoryFileFrontmatter, relative_path: str) -> tuple[str, bool, str]:
        policy = MemoryTypePolicy.for_type(frontmatter.memory_type)
        if relative_path.startswith("system/") or frontmatter.authority == "system" or policy.authority == "high":
            return "high", True, "system_memory" if relative_path.startswith("system/") else "high_authority_memory"
        if policy.review_required:
            return "medium", True, "review_required_memory"
        return "low", False, "memfs_defrag"

    def _is_stale_temporary_task(self, frontmatter: MemoryFileFrontmatter) -> bool:
        return frontmatter.memory_type == "TemporaryTaskState" and (frontmatter.ttl == "short" or frontmatter.type_policy.ttl == "short")

    def _normalize_body(self, body: str) -> str:
        return body.strip()

    def _render_memory_file(self, frontmatter: MemoryFileFrontmatter, body: str) -> str:
        data: dict[str, Any] = {
            "description": frontmatter.description,
            "authority": frontmatter.authority,
            "mode_scope": list(frontmatter.mode_scope),
            "buckets": list(frontmatter.buckets),
            "source": frontmatter.source,
            "last_reviewed": frontmatter.last_reviewed,
        }
        data.update(frontmatter.extra)
        yaml_text = yaml.safe_dump(data, sort_keys=False, allow_unicode=True).strip()
        return f"---\n{yaml_text}\n---\n{body}"


__all__ = ["DefragAction", "DefragGovernanceReport", "DefragPlan", "MemFSDefragger"]
