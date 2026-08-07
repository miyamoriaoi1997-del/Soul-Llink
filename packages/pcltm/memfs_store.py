"""Filesystem-backed MemFS memory layer store.

This module provides a small, deterministic local view over a MemFS-style
memory repository.  It intentionally performs no LLM calls and treats invalid
memory files as warnings so a single malformed file cannot break runtime reads.
"""

from __future__ import annotations

import os
import re
import shutil
import subprocess
import tempfile
import warnings
from pathlib import Path
from typing import Any

import yaml

from .memfs_types import MemoryFileFrontmatter, MemoryLayerItem, MemoryLayerView
from .secret_policy import redact_secrets


MEMFS_LAYERS = ("system", "pinned", "episodic", "transient")
MEMFS_DIRECTORIES = (*MEMFS_LAYERS, "skills")


def atomic_write_text(path: Path, text: str, *, encoding: str = "utf-8") -> None:
    """Durably replace a text file without exposing a partially written target."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w",
            encoding=encoding,
            newline="",
            dir=path.parent,
            prefix=f".{path.name}.",
            suffix=".tmp",
            delete=False,
        ) as stream:
            temporary = Path(stream.name)
            stream.write(text)
            stream.flush()
            os.fsync(stream.fileno())
        temporary.replace(path)
        temporary = None
    finally:
        if temporary is not None:
            try:
                temporary.unlink()
            except FileNotFoundError:
                pass


class MemFSStore:
    """Local filesystem-backed MemFS view layer."""

    def __init__(
        self,
        root: Path,
        *,
        query_alias_groups: tuple[tuple[str, ...], ...] = (),
    ):
        self.root = Path(root)
        self.query_alias_groups = query_alias_groups

    def init(self) -> None:
        """Create the expected MemFS directory layout without deleting data."""
        self.root.mkdir(parents=True, exist_ok=True)
        for name in MEMFS_DIRECTORIES:
            (self.root / name).mkdir(parents=True, exist_ok=True)

    def list_tree(self) -> list[MemoryLayerItem]:
        """Return progressive-disclosure tree items: path + metadata only."""
        items: list[MemoryLayerItem] = []
        for path in self._iter_memory_files():
            rel = self._relative_path(path)
            parsed = self._read_valid_file(rel, warn=True)
            if parsed is None:
                continue
            frontmatter, body = parsed
            items.append(
                MemoryLayerItem(
                    path=rel,
                    id=rel,
                    description=frontmatter.description,
                    body="",
                    authority=frontmatter.authority,
                    buckets=frontmatter.buckets,
                    mode_scope=frontmatter.mode_scope,
                    char_count=len(body),
                    char_limit=frontmatter.char_limit or len(body),
                    read_only=frontmatter.read_only,
                    metadata={**frontmatter.metadata, **frontmatter.extra},
                    updated_at=frontmatter.updated_at or frontmatter.last_reviewed,
                    score=0.0,
                    memory_type=frontmatter.memory_type,
                    lifecycle_state=frontmatter.lifecycle_state,
                    ttl=frontmatter.ttl,
                    conflict_policy=frontmatter.conflict_policy,
                    injection_policy=frontmatter.injection_policy,
                    evidence_refs=frontmatter.evidence_refs,
                )
            )
        return sorted(items, key=lambda item: item.path)

    def load_layer(
        self,
        layer: str,
        *,
        mode: str | None = None,
        query: str | None = None,
        budget_chars: int = 2000,
        buckets: set[str] | None = None,
    ) -> MemoryLayerView:
        """Load a full-body view for one memory layer under a character budget."""
        if layer not in MEMFS_LAYERS:
            raise ValueError(f"invalid layer {layer!r}; expected one of {list(MEMFS_LAYERS)}")
        if budget_chars < 0:
            raise ValueError("budget_chars must be >= 0")

        candidates = self._filter_by_buckets(self._load_layer_candidates(layer), buckets)

        if layer == "system":
            ordered = sorted(candidates, key=lambda item: item.path)
        elif layer == "pinned":
            filtered = [item for item in candidates if self._mode_matches(item.mode_scope, mode)]
            scored = [self._with_score(item, query) for item in filtered]
            ordered = sorted(scored, key=lambda item: (-item.score, item.path))
        elif layer == "episodic":
            filtered = [item for item in candidates if self._mode_matches(item.mode_scope, mode)]
            scored = [self._with_score(item, query) for item in filtered]
            ordered = sorted(
                scored,
                key=lambda item: (-item.score, *self._recency_sort_key(item.path), item.path),
            )
        else:  # transient: current files only; bucket filtering still applies.
            ordered = sorted(candidates, key=lambda item: item.path)

        selected: list[MemoryLayerItem] = []
        omitted_count = 0
        used_chars = 0
        for item in ordered:
            if used_chars + item.char_count > budget_chars:
                omitted_count += 1
                continue
            selected.append(item)
            used_chars += item.char_count

        return MemoryLayerView(
            layer=layer,
            items=selected,
            omitted_count=omitted_count,
            budget_chars=budget_chars,
            used_chars=used_chars,
        )

    def read_file(self, relative_path: str) -> tuple[MemoryFileFrontmatter, str]:
        """Safely read and parse a memory file relative to the MemFS root."""
        path = self._safe_resolve(relative_path)
        text = path.read_text(encoding="utf-8")
        return self._parse_memory_text(text, relative_path)

    def search(
        self,
        query: str | None,
        *,
        layers: tuple[str, ...] | list[str] = ("episodic",),
        mode: str | None = None,
        buckets: set[str] | None = None,
        limit: int = 8,
        excerpt_chars: int = 240,
    ) -> list[dict[str, Any]]:
        """Retired: unbound MemFS roots are not a recall authority."""
        return []

    def open_memory(self, memory_id: str, *, body_limit: int = 4000) -> dict[str, Any]:
        """Retired: governed claim open is the only durable body surface."""
        raise RuntimeError("legacy_memfs_open_retired")

    def ensure_git_repo(self) -> bool:
        """Initialize a git repo at self.root if one doesn't exist. Return True if repo is ready."""
        if shutil.which("git") is None:
            warnings.warn("git binary not available; MemFS versioning disabled", RuntimeWarning)
            return False

        try:
            self.root.mkdir(parents=True, exist_ok=True)
            git_dir = self.root / ".git"
            if not git_dir.is_dir():
                subprocess.run(
                    ["git", "init"],
                    cwd=self.root,
                    check=True,
                    stdout=subprocess.DEVNULL,
                    stderr=subprocess.PIPE,
                    text=True,
                )
            return git_dir.is_dir()
        except (OSError, subprocess.SubprocessError) as exc:
            warnings.warn(f"failed to initialize MemFS git repo at {self.root}: {exc}", RuntimeWarning)
            return False

    def has_uncommitted_changes(self) -> bool:
        """Check if there are uncommitted changes in the memory repo. Return False on any error."""
        if shutil.which("git") is None:
            return False

        try:
            unstaged = subprocess.run(
                ["git", "diff", "--quiet"],
                cwd=self.root,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
                text=True,
            )
            if unstaged.returncode == 1:
                return True
            if unstaged.returncode != 0:
                return False

            staged = subprocess.run(
                ["git", "diff", "--cached", "--quiet"],
                cwd=self.root,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
                text=True,
            )
            if staged.returncode == 1:
                return True
            status = subprocess.run(
                ["git", "status", "--porcelain"],
                cwd=self.root,
                stdout=subprocess.PIPE,
                stderr=subprocess.DEVNULL,
                text=True,
            )
            if status.returncode != 0:
                return False
            if status.stdout.strip():
                return True

            return False
        except (OSError, subprocess.SubprocessError):
            return False

    def commit_changes(self, message: str) -> bool:
        """Commit all changed memory files. Return True on success."""
        if shutil.which("git") is None:
            return False

        try:
            subprocess.run(
                ["git", "add", "-A"],
                cwd=self.root,
                check=True,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.PIPE,
                text=True,
            )
            subprocess.run(
                ["git", "commit", "-m", message],
                cwd=self.root,
                check=True,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.PIPE,
                text=True,
            )
            return True
        except (OSError, subprocess.SubprocessError):
            return False

    def _load_layer_candidates(self, layer: str) -> list[MemoryLayerItem]:
        layer_root = self.root / layer
        if not layer_root.exists():
            return []
        items: list[MemoryLayerItem] = []
        for path in sorted(layer_root.rglob("*.md")):
            if not path.is_file():
                continue
            rel = self._relative_path(path)
            parsed = self._read_valid_file(rel, warn=True)
            if parsed is None:
                continue
            frontmatter, body = parsed
            items.append(
                MemoryLayerItem(
                    path=rel,
                    id=rel,
                    description=frontmatter.description,
                    body=redact_secrets(body),
                    authority=frontmatter.authority,
                    buckets=frontmatter.buckets,
                    mode_scope=frontmatter.mode_scope,
                    char_count=len(body),
                    char_limit=frontmatter.char_limit or len(body),
                    read_only=frontmatter.read_only,
                    metadata={**frontmatter.metadata, **frontmatter.extra},
                    updated_at=frontmatter.updated_at or frontmatter.last_reviewed,
                    score=0.0,
                    memory_type=frontmatter.memory_type,
                    lifecycle_state=frontmatter.lifecycle_state,
                    ttl=frontmatter.ttl,
                    conflict_policy=frontmatter.conflict_policy,
                    injection_policy=frontmatter.injection_policy,
                    evidence_refs=frontmatter.evidence_refs,
                )
            )
        return items

    def _iter_memory_files(self) -> list[Path]:
        if not self.root.exists():
            return []
        return sorted(path for path in self.root.rglob("*.md") if path.is_file())

    def _read_valid_file(
        self, relative_path: str, *, warn: bool = False
    ) -> tuple[MemoryFileFrontmatter, str] | None:
        try:
            return self.read_file(relative_path)
        except (OSError, ValueError, yaml.YAMLError) as exc:
            if warn:
                warnings.warn(f"skipping invalid MemFS file {relative_path}: {exc}", RuntimeWarning)
            return None

    def write_file(self, relative_path: str, frontmatter: MemoryFileFrontmatter, body: str) -> None:
        """Safely write a MemFS markdown file with YAML frontmatter."""
        path = self._safe_resolve(relative_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        normalized_body = str(body or "").rstrip()
        if frontmatter.char_limit is not None and len(normalized_body) > frontmatter.char_limit:
            raise ValueError(
                f"body exceeds char_limit ({len(normalized_body)} > {frontmatter.char_limit})"
            )
        data = dict(frontmatter.extra)
        data.update({
            "description": frontmatter.description,
            "authority": frontmatter.authority,
            "mode_scope": list(frontmatter.mode_scope),
            "buckets": list(frontmatter.buckets),
            "source": frontmatter.source,
            "last_reviewed": frontmatter.last_reviewed,
            "char_limit": frontmatter.char_limit,
            "read_only": frontmatter.read_only,
            "metadata": dict(frontmatter.metadata),
            "updated_at": frontmatter.updated_at,
            "memory_type": frontmatter.memory_type,
            "lifecycle_state": frontmatter.lifecycle_state,
            "ttl": frontmatter.ttl,
            "conflict_policy": frontmatter.conflict_policy,
            "injection_policy": frontmatter.injection_policy,
            "evidence_refs": [dict(ref) for ref in frontmatter.evidence_refs],
        })
        text = (
            "---\n"
            + yaml.safe_dump(data, allow_unicode=True, sort_keys=False)
            + "---\n"
            + normalized_body
            + "\n"
        )
        atomic_write_text(path, text)

    def _safe_resolve(self, relative_path: str) -> Path:
        candidate = Path(relative_path)
        if candidate.is_absolute() or ".." in candidate.parts:
            raise ValueError(f"unsafe MemFS relative path: {relative_path!r}")

        root = self.root.resolve()
        resolved = (root / candidate).resolve()
        try:
            resolved.relative_to(root)
        except ValueError as exc:
            raise ValueError(f"path escapes MemFS root: {relative_path!r}") from exc
        return resolved

    def _relative_path(self, path: Path) -> str:
        return path.resolve().relative_to(self.root.resolve()).as_posix()

    def _parse_memory_text(
        self, text: str, relative_path: str
    ) -> tuple[MemoryFileFrontmatter, str]:
        lines = text.splitlines()
        if not lines or lines[0].strip() != "---":
            raise ValueError("missing YAML frontmatter")

        end_index: int | None = None
        for index in range(1, len(lines)):
            if lines[index].strip() == "---":
                end_index = index
                break
        if end_index is None:
            raise ValueError("unterminated YAML frontmatter")

        raw_frontmatter = "\n".join(lines[1:end_index])
        loaded = yaml.safe_load(raw_frontmatter)
        if not isinstance(loaded, dict):
            raise ValueError("frontmatter must be a YAML mapping")

        frontmatter = self._frontmatter_from_mapping(loaded)
        body = "\n".join(lines[end_index + 1 :])
        if text.endswith("\n") and body:
            body += "\n"
        return frontmatter, body

    def _frontmatter_from_mapping(self, data: dict[str, Any]) -> MemoryFileFrontmatter:
        known = {
            "description",
            "authority",
            "mode_scope",
            "buckets",
            "source",
            "last_reviewed",
            "char_limit",
            "read_only",
            "metadata",
            "updated_at",
            "memory_type",
            "lifecycle_state",
            "ttl",
            "conflict_policy",
            "injection_policy",
            "evidence_refs",
        }
        raw_metadata = data.get("metadata", {})
        if not isinstance(raw_metadata, dict):
            raw_metadata = {"value": raw_metadata}
        raw_memory_type = str(
            data.get("memory_type")
            or raw_metadata.get("memory_type")
            or raw_metadata.get("type")
            or "UserPreference"
        )
        # Governed claim vocabulary predates the Letta-style MemFS view types.
        # Normalize only at this read boundary; preserve the original value in
        # ``extra`` so the filesystem projection remains an auditable view of
        # the canonical claim rather than a second authority.
        memory_type_aliases = {
            "preference": "UserPreference",
            "user_preference": "UserPreference",
            "memory_note": "WorkflowConvention",
            "system_convention": "WorkflowConvention",
            "architecture_current": "RuntimeInvariant",
            "relationship": "RelationshipAnchor",
            "runtime_invariant": "RuntimeInvariant",
            "persona_boundary": "PersonaBoundary",
            "tool_quirk": "ToolQuirk",
            "project_path": "ProjectPath",
            "workflow_convention": "WorkflowConvention",
            "risk_policy": "RiskPolicy",
            "temporary_task_state": "TemporaryTaskState",
        }
        normalized_memory_type = memory_type_aliases.get(raw_memory_type, raw_memory_type)
        kwargs = {
            "description": str(data.get("description", "")),
            "authority": str(data.get("authority", "pinned")),
            "mode_scope": self._string_tuple(data.get("mode_scope", ("daily", "work", "sex"))),
            "buckets": self._string_tuple(data.get("buckets", ())),
            "source": str(data.get("source", "pcltm")),
            "last_reviewed": str(data.get("last_reviewed", "")),
            "char_limit": data.get("char_limit"),
            "read_only": bool(data.get("read_only", False)),
            "metadata": raw_metadata,
            "updated_at": str(data.get("updated_at", "")),
            "memory_type": normalized_memory_type,
            "lifecycle_state": str(data.get("lifecycle_state") or raw_metadata.get("lifecycle_state") or raw_metadata.get("status") or "active"),
            "ttl": str(data.get("ttl") or raw_metadata.get("ttl") or "none"),
            "conflict_policy": str(data.get("conflict_policy") or raw_metadata.get("conflict_policy") or ""),
            "injection_policy": str(data.get("injection_policy") or raw_metadata.get("injection_policy") or raw_metadata.get("inject_policy") or ""),
            "evidence_refs": tuple(data.get("evidence_refs") or raw_metadata.get("evidence_refs") or ()),
            "extra": {
                **{k: v for k, v in data.items() if k not in known},
                **(
                    {"governed_memory_type": raw_memory_type}
                    if normalized_memory_type != raw_memory_type
                    else {}
                ),
            },
        }
        return MemoryFileFrontmatter(**kwargs)

    def _string_tuple(self, value: Any) -> tuple[str, ...]:
        if value is None:
            return ()
        if isinstance(value, str):
            return (value,)
        if isinstance(value, (list, tuple, set)):
            return tuple(str(item) for item in value)
        return (str(value),)

    def _mode_matches(self, mode_scope: tuple[str, ...], mode: str | None) -> bool:
        if mode is None:
            return True
        return mode in mode_scope or "default" in mode_scope

    def _filter_by_buckets(
        self,
        items: list[MemoryLayerItem],
        buckets: set[str] | None,
    ) -> list[MemoryLayerItem]:
        """Keep items that intersect the selected context buckets.

        System files are allowed through even without a bucket match because
        they carry top-level operating contracts.  Other layers are trimmed by
        the MemorySelection budget profile so Letta-style memory selection is
        coupled to the persona/context router instead of blindly rendering all
        files in a mode.
        """
        if not buckets:
            return items
        selected: list[MemoryLayerItem] = []
        for item in items:
            if item.authority == "system" or set(item.buckets) & buckets:
                selected.append(item)
        return selected

    def _excerpt(self, text: str, limit: int) -> str:
        clean = " ".join((text or "").split())
        if limit < 0 or len(clean) <= limit:
            return clean
        return clean[: max(0, limit - 1)].rstrip() + "…"

    def _with_score(self, item: MemoryLayerItem, query: str | None) -> MemoryLayerItem:
        score = self._query_score(item, query)
        return MemoryLayerItem(
            path=item.path,
            id=item.id or item.path,
            description=item.description,
            body=item.body,
            authority=item.authority,
            buckets=item.buckets,
            mode_scope=item.mode_scope,
            char_count=item.char_count,
            char_limit=item.char_limit,
            read_only=item.read_only,
            metadata=dict(item.metadata),
            updated_at=item.updated_at,
            score=score,
            memory_type=item.memory_type,
            lifecycle_state=item.lifecycle_state,
            ttl=item.ttl,
            conflict_policy=item.conflict_policy,
            injection_policy=item.injection_policy,
            evidence_refs=item.evidence_refs,
        )

    def _query_score(self, item: MemoryLayerItem, query: str | None) -> float:
        terms = self._query_terms(query)
        if not terms:
            return 0.0
        bucket_text = " ".join(item.buckets).lower()
        metadata_text = " ".join((
            item.path,
            item.description,
            *sorted(self._fact_projection_terms(item.metadata)),
        )).lower()
        body_text = item.body.lower()
        score = 0.0
        for term in terms:
            if term in bucket_text:
                score += 3.0
            if term in metadata_text:
                score += 2.0
            if term in body_text:
                score += 1.0
        return score

    def _query_terms(self, query: str | None) -> list[str]:
        if not query:
            return []
        terms = [term for term in re.split(r"\W+", query.lower()) if term]
        compact = "".join(query.lower().split())
        for group in self.query_alias_groups:
            normalized = tuple(str(term).strip().lower() for term in group if str(term).strip())
            if any(term in compact for term in normalized):
                terms.extend(normalized)
        for size in (2, 3, 4):
            for i in range(0, max(0, len(compact) - size + 1)):
                chunk = compact[i : i + size]
                if any("\u4e00" <= ch <= "\u9fff" for ch in chunk):
                    terms.append(chunk)
        return list(dict.fromkeys(terms))

    @staticmethod
    def _fact_projection_terms(metadata: dict[str, Any]) -> set[str]:
        """Project explicit structured fact values without changing memory bodies."""
        terms: set[str] = set()
        facts = metadata.get("facts")
        if not isinstance(facts, dict):
            return terms
        stack: list[Any] = list(facts.values())
        while stack:
            value = stack.pop()
            if isinstance(value, dict):
                stack.extend(value.values())
            elif isinstance(value, (list, tuple, set)):
                stack.extend(value)
            elif value is not None:
                text = str(value).strip().lower()
                if text:
                    terms.add(text)
        return terms

    def _recency_sort_key(self, path: str) -> tuple[int, int]:
        year, month, _ = self._recency_key(path)
        return (-year, -month)

    def _recency_key(self, path: str) -> tuple[int, int, str]:
        # Sort descending by this key.  Supports episodic/YYYY/MM/*.md and
        # episodic/YYYY/YYYY-MM-topic.md layouts.
        year = 0
        month = 0
        match = re.search(r"(?:^|/)(20\d{2})(?:/|-)(0[1-9]|1[0-2])", path)
        if match:
            year = int(match.group(1))
            month = int(match.group(2))
        else:
            match = re.search(r"(?:^|/)(20\d{2})(?:/|$)", path)
            if match:
                year = int(match.group(1))
        return (year, month, path)


__all__ = ["MEMFS_DIRECTORIES", "MEMFS_LAYERS", "MemFSStore"]
