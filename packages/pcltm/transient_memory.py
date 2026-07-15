"""Focused writers for transient working-memory and evidence MemFS entries."""
from __future__ import annotations

import hashlib
import time
from pathlib import Path

from .memfs_store import MemFSStore
from .memfs_types import MemoryFileFrontmatter
from .secret_policy import redact_secrets


def _safe_title(value: str, fallback: str) -> str:
    return " ".join(str(value or fallback).split())[:120] or fallback


def _safe_body(value: str, limit: int) -> str:
    body = redact_secrets("\n".join(str(value or "").splitlines()).strip())
    if len(body) > limit:
        return body[: limit - 1] + "…"
    return body


def _bucket_values(
    buckets: list[str] | tuple[str, ...] | None,
    fallback: str,
) -> list[str]:
    return [str(bucket) for bucket in (buckets or [fallback]) if str(bucket)] or [fallback]


def write_current_task_state(
    *,
    title: str,
    body: str,
    mode: str,
    task_id: str | None,
    buckets: list[str] | tuple[str, ...] | None,
    root: Path,
) -> dict:
    safe_title = _safe_title(title, "current task")
    safe_body = _safe_body(body, 1600)
    if not safe_body:
        return {"ok": False, "error": "current_task_body_required"}

    relative_path = "transient/current-task.md"
    frontmatter = MemoryFileFrontmatter(
        description=safe_title,
        authority="transient",
        mode_scope=(str(mode or "work"),),
        buckets=tuple(_bucket_values(buckets, "current_task")),
        source="current_task_state",
        memory_type="TemporaryTaskState",
        lifecycle_state="active",
        ttl="short",
        injection_policy="transient_only",
        metadata={
            "task_id": task_id or "current",
            "reference_only": False,
            "overwrite_policy": "replace_current_task",
        },
    )
    MemFSStore(root=root).write_file(relative_path, frontmatter, safe_body)
    return {
        "ok": True,
        "memory_id": relative_path,
        "layer": "transient",
        "body_chars": len(safe_body),
        "overwrite_policy": "replace_current_task",
    }


def write_evidence_capsule(
    *,
    title: str,
    body: str,
    mode: str,
    buckets: list[str] | tuple[str, ...] | None,
    source_tool: str,
    evidence_id: str | None,
    root: Path,
) -> dict:
    safe_title = _safe_title(title, "evidence capsule")
    safe_body = _safe_body(body, 1800)
    tool_value = str(source_tool or "tool")[:80]
    slug_source = evidence_id or f"{safe_title}-{tool_value}-{time.time_ns()}"
    slug = hashlib.sha256(slug_source.encode("utf-8", errors="ignore")).hexdigest()[:16]
    relative_path = f"transient/evidence-{slug}.md"
    frontmatter = MemoryFileFrontmatter(
        description=safe_title,
        authority="transient",
        mode_scope=(str(mode or "work"),),
        buckets=tuple(_bucket_values(buckets, "tool_evidence")),
        source=f"tool_evidence:{tool_value}",
        metadata={
            "evidence_id": evidence_id or slug,
            "source_tool": tool_value,
            "reference_only": True,
        },
    )
    MemFSStore(root=root).write_file(relative_path, frontmatter, safe_body)
    return {
        "ok": True,
        "memory_id": relative_path,
        "layer": "transient",
        "reference_only": True,
        "body_chars": len(safe_body),
    }


__all__ = ["write_current_task_state", "write_evidence_capsule"]
