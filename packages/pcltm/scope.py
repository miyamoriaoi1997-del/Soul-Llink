"""Stable scope keys for SoulLink/PCLTM memory isolation.

The helpers here are intentionally storage-neutral.  They give SQLite rows,
MemFS frontmatter, semantic facts, and future derived indexes the same compact
scope vocabulary without forcing a specific persistence backend.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Iterable

from .memory_object import MemoryObjectScope

_SCOPE_ORDER = (
    ("profile", "profile_id"),
    ("app", "app_id"),
    ("project", "project_id"),
    ("persona", "persona_id"),
    ("user", "user_id"),
)


def _slug(value: object) -> str:
    text = str(value or "").strip().lower()
    text = re.sub(r"[^a-z0-9]+", "-", text)
    text = text.strip("-")
    return text or "default"


def _normalized_modes(mode_scope: Iterable[str] | None) -> tuple[str, ...]:
    if not mode_scope:
        return ()
    return tuple(sorted({_slug(mode) for mode in mode_scope if str(mode or "").strip()}))


@dataclass(frozen=True)
class MemoryScope:
    """Explicit multidimensional memory scope.

    Empty fields are omitted from the rendered key.  This lets callers create a
    narrow key such as ``project:foo/modes:work`` or a full production key that
    includes profile/app/project/persona/user/mode dimensions.
    """

    profile_id: str | None = None
    app_id: str | None = None
    project_id: str | None = None
    persona_id: str | None = None
    user_id: str | None = None
    mode_scope: tuple[str, ...] = ()

    @property
    def key(self) -> str:
        return build_scope_key(
            profile_id=self.profile_id,
            app_id=self.app_id,
            project_id=self.project_id,
            persona_id=self.persona_id,
            user_id=self.user_id,
            mode_scope=self.mode_scope,
        )

    def to_metadata(self) -> dict[str, object]:
        return {
            "scope_key": self.key,
            "profile_id": self.profile_id or "",
            "app_id": self.app_id or "",
            "project_id": self.project_id or "",
            "persona_id": self.persona_id or "",
            "user_id": self.user_id or "",
            "mode_scope": list(self.mode_scope),
        }


def build_scope_key(
    *,
    profile_id: str | None = None,
    app_id: str | None = None,
    project_id: str | None = None,
    persona_id: str | None = None,
    user_id: str | None = None,
    mode_scope: Iterable[str] | None = None,
) -> str:
    """Build a deterministic scope key from optional dimensions."""

    values = {
        "profile_id": profile_id,
        "app_id": app_id,
        "project_id": project_id,
        "persona_id": persona_id,
        "user_id": user_id,
    }
    parts: list[str] = []
    for label, field_name in _SCOPE_ORDER:
        raw = values[field_name]
        if raw is not None and str(raw).strip():
            parts.append(f"{label}:{_slug(raw)}")
    modes = _normalized_modes(mode_scope)
    if modes:
        parts.append("modes:" + "+".join(modes))
    return "/".join(parts) or "global:default"


def scoped_canonical_key(scope: MemoryScope | str, object_scope: MemoryObjectScope | str, name: str) -> str:
    """Attach a scope key to a memory object's local canonical name."""

    scope_key = scope.key if isinstance(scope, MemoryScope) else str(scope or "global:default")
    object_scope_value = object_scope.value if isinstance(object_scope, MemoryObjectScope) else str(object_scope)
    return f"{scope_key}/{_slug(object_scope_value)}/{_slug(name)}"
