"""Read-only sleep-time reflection governance support for MemFS.

Reflection writes candidate memory drafts into episodic/ or transient/,
never directly into system/ unless explicitly allowed.

This module is intentionally deterministic: it builds reviewable draft files from
structured events and performs no LLM calls.  Older ``ReflectionCandidate`` and
``ReflectionBuilder`` helpers are kept for compatibility with the existing PCLTM
reflection candidate workflow.
"""

from __future__ import annotations

import hashlib
import re
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any

import yaml

from .memfs_store import MemFSStore
from .memfs_types import MemoryFileFrontmatter

DRAFT_TARGET_LAYERS = {"episodic", "transient", "system"}


@dataclass(frozen=True)
class MemoryDraft:
    """A proposed memory file to be written."""

    target_layer: str  # "episodic" or "transient" or "system"
    relative_path: str  # e.g. "episodic/2026/2026-05-task-review.md"
    frontmatter: MemoryFileFrontmatter
    body: str
    source: str = "reflection"
    provenance: str = ""  # describes where this draft came from

    def __post_init__(self) -> None:
        if self.target_layer not in DRAFT_TARGET_LAYERS:
            raise ValueError(
                f"invalid target_layer {self.target_layer!r}; "
                f"expected one of {sorted(DRAFT_TARGET_LAYERS)}"
            )
        if not self.relative_path or not self.relative_path.strip():
            raise ValueError("relative_path is required")
        if not isinstance(self.frontmatter, MemoryFileFrontmatter):
            raise TypeError("frontmatter must be a MemoryFileFrontmatter")
        if not isinstance(self.body, str):
            raise TypeError("body must be a string")

        # Reconstruct the frontmatter object to force dataclass validation even
        # if a subclass or mutated object is supplied.
        MemoryFileFrontmatter(
            description=self.frontmatter.description,
            authority=self.frontmatter.authority,
            mode_scope=tuple(self.frontmatter.mode_scope),
            buckets=tuple(self.frontmatter.buckets),
            source=self.frontmatter.source,
            last_reviewed=self.frontmatter.last_reviewed,
            extra=dict(self.frontmatter.extra),
        )


class ReflectionCandidateBuilder:
    """Build memory drafts from structured event records (no LLM calls yet)."""

    def build_from_events(self, events: list[dict]) -> list[MemoryDraft]:
        """Convert event dicts into MemoryDraft objects."""
        drafts: list[MemoryDraft] = []
        for index, event in enumerate(events):
            if not isinstance(event, dict):
                raise TypeError("events must be dictionaries")
            drafts.append(self._build_event_draft(event, index))
        return drafts

    def _build_event_draft(self, event: dict, index: int) -> MemoryDraft:
        event_type = str(event.get("type") or "event").strip() or "event"
        user_message = str(event.get("user_message") or event.get("message") or "").strip()
        mode = str(event.get("mode") or "default").strip() or "default"
        timestamp = str(event.get("timestamp") or self._now_iso()).strip()
        dt = self._parse_timestamp(timestamp)
        date = dt.date().isoformat()
        year = f"{dt.year:04d}"
        slug = self._slug(event_type)
        digest = hashlib.sha256(
            repr(sorted(event.items())).encode("utf-8", errors="replace")
        ).hexdigest()[:8]

        relative_path = f"episodic/{year}/{date}-{slug}-{digest}.md"
        description = self._description(event_type, user_message)
        frontmatter = MemoryFileFrontmatter(
            description=description,
            authority="episodic",
            mode_scope=(mode,),
            buckets=("reflection_candidate", event_type),
            source="reflection",
            last_reviewed=date,
            extra={
                "event_type": event_type,
                "draft_status": "candidate",
                "provenance": "structured_event",
            },
        )
        body_lines = [
            f"# Reflection candidate: {event_type}",
            "",
            f"- timestamp: {timestamp}",
            f"- mode: {mode}",
        ]
        if user_message:
            body_lines.extend(["- user_message:", f"  {user_message}"])
        remaining = {
            str(k): v
            for k, v in event.items()
            if k not in {"type", "user_message", "message", "mode", "timestamp"}
        }
        if remaining:
            body_lines.extend(["", "## Event data", "", yaml.safe_dump(remaining, allow_unicode=True, sort_keys=True).strip()])
        body = "\n".join(body_lines).rstrip() + "\n"
        return MemoryDraft(
            target_layer="episodic",
            relative_path=relative_path,
            frontmatter=frontmatter,
            body=body,
            source="reflection",
            provenance=f"event[{index}]:{event_type}",
        )

    @staticmethod
    def _now_iso() -> str:
        return datetime.now(timezone.utc).replace(microsecond=0).isoformat()

    @staticmethod
    def _parse_timestamp(timestamp: str) -> datetime:
        try:
            normalized = timestamp.replace("Z", "+00:00")
            return datetime.fromisoformat(normalized)
        except ValueError:
            return datetime.now(timezone.utc)

    @staticmethod
    def _slug(value: str) -> str:
        slug = re.sub(r"[^a-zA-Z0-9]+", "-", value.lower()).strip("-")
        return slug or "event"

    @staticmethod
    def _description(event_type: str, user_message: str) -> str:
        prefix = f"Reflection candidate from {event_type}"
        if not user_message:
            return prefix
        compact = " ".join(user_message.split())
        if len(compact) > 80:
            compact = compact[:79].rstrip() + "…"
        return f"{prefix}: {compact}"


class ReflectionWriter:
    """Write approved memory drafts to MemFS store."""

    def __init__(self, store: MemFSStore):
        self.store = store

    def write_draft(self, draft: MemoryDraft, *, allow_system: bool = False) -> bool:
        """Write a single draft. Reject system/ writes unless allow_system=True."""
        if draft.target_layer == "system" and not allow_system:
            return False
        if draft.relative_path.split("/", 1)[0] != draft.target_layer:
            raise ValueError("draft relative_path must begin with target_layer")

        path = self.store._safe_resolve(draft.relative_path)  # noqa: SLF001 - store owns root safety.
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(self._render_draft(draft), encoding="utf-8")
        return True

    def write_all(self, drafts: list[MemoryDraft], *, allow_system: bool = False) -> dict:
        """Write multiple drafts. Returns {"written": N, "rejected": N, "errors": N}."""
        stats = {"written": 0, "rejected": 0, "errors": 0}
        for draft in drafts:
            try:
                if self.write_draft(draft, allow_system=allow_system):
                    stats["written"] += 1
                else:
                    stats["rejected"] += 1
            except Exception:
                stats["errors"] += 1
        return stats

    def _render_draft(self, draft: MemoryDraft) -> str:
        frontmatter = self._frontmatter_mapping(draft)
        yaml_text = yaml.safe_dump(
            frontmatter,
            allow_unicode=True,
            sort_keys=False,
            default_flow_style=False,
        ).strip()
        body = draft.body.rstrip() + "\n" if draft.body else ""
        return f"---\n{yaml_text}\n---\n\n{body}"

    @staticmethod
    def _frontmatter_mapping(draft: MemoryDraft) -> dict[str, Any]:
        fm = draft.frontmatter
        data: dict[str, Any] = {
            "description": fm.description,
            "authority": fm.authority,
            "mode_scope": list(fm.mode_scope),
            "buckets": list(fm.buckets),
            "source": draft.source or fm.source,
            "last_reviewed": fm.last_reviewed,
        }
        data.update(fm.extra)
        if draft.provenance:
            data.setdefault("provenance", draft.provenance)
        return data


@dataclass(frozen=True)
class ReflectionCandidate:
    candidate_id: str
    summary: str
    mode_scope: tuple[str, ...]
    source_record_ids: tuple[int, ...] = ()
    source_node_ids: tuple[int, ...] = ()
    tags: tuple[str, ...] = ()
    confidence: float = 0.7
    inject_policy: str = "retrieve_only"
    status: str = "pending"
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_memory_record(self, *, target_file: str = "MEMORY.md") -> dict[str, Any]:
        return {
            "candidate_id": self.candidate_id,
            "kind": "reflection",
            "target_file": target_file,
            "content": self.summary,
            "confidence": self.confidence,
            "sensitivity": "normal",
            "source_event_ids": [],
            "source_node_ids": list(self.source_node_ids),
            "status": self.status,
            "reviewer": None,
            "decision_reason": "reflection candidate; not active until approved/promoted",
            "metadata": {
                "category": "reflection",
                "mode_scope": list(self.mode_scope),
                "tags": list(self.tags),
                "inject_policy": self.inject_policy,
                "source_record_ids": list(self.source_record_ids),
                "source_node_ids": list(self.source_node_ids),
                **self.metadata,
            },
        }


class ReflectionBuilder:
    """Create bounded reflection candidates from already-approved evidence.

    This builder is intentionally deterministic. It avoids LLM rewriting and
    does not mutate the store by itself.
    """

    def __init__(self, *, max_summary_chars: int = 280):
        self.max_summary_chars = max_summary_chars

    def build_from_records(
        self,
        records: list[dict[str, Any]],
        *,
        mode_scope: tuple[str, ...],
        tags: tuple[str, ...] = (),
        prefix: str = "Reflection",
    ) -> ReflectionCandidate | None:
        if not records:
            return None
        normalized_scope = tuple(sorted({str(mode).lower() for mode in mode_scope if mode}))
        if not normalized_scope:
            return None
        source_ids = tuple(int(record.get("record_id") or 0) for record in records if record.get("record_id"))
        evidence = "；".join((record.get("content") or "").strip() for record in records if (record.get("content") or "").strip())
        if not evidence:
            return None
        summary = f"{prefix}: {evidence}"
        if len(summary) > self.max_summary_chars:
            summary = summary[: self.max_summary_chars - 1].rstrip() + "…"
        candidate_id = self._candidate_id(summary, normalized_scope, source_ids)
        return ReflectionCandidate(
            candidate_id=candidate_id,
            summary=summary,
            mode_scope=normalized_scope,
            source_record_ids=source_ids,
            tags=tuple(sorted({str(tag).lower() for tag in tags if tag})),
            metadata={"source": "reflection_builder"},
        )


    @staticmethod
    def _extract_actionable_summary(text: str) -> str:
        markers = ("## Active Task", "## Key Decisions", "## Remaining Work", "## Critical Context")
        for marker in markers:
            idx = text.find(marker)
            if idx >= 0:
                return " ".join(text[idx:].split())
        return " ".join(text.split())

    @staticmethod
    def _candidate_id(summary: str, mode_scope: tuple[str, ...], source_ids: tuple[int, ...]) -> str:
        material = "|".join([summary, ",".join(mode_scope), ",".join(map(str, source_ids))])
        return "reflection:" + hashlib.sha256(material.encode("utf-8")).hexdigest()[:16]


__all__ = [
    "MemoryDraft",
    "ReflectionBuilder",
    "ReflectionCandidate",
    "ReflectionCandidateBuilder",
    "ReflectionWriter",
]