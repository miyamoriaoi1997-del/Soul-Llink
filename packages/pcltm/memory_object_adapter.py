from __future__ import annotations

import re
from collections.abc import Mapping
from typing import Any

from .memory_object import (
    InjectionPolicy,
    MemoryObject,
    MemoryObjectScope,
    MemoryObjectStatus,
    MemoryObjectType,
    StateAffinity,
)


_IDENTITY_HINTS = {"identity", "persona", "anchor", "pinned", "user"}
_RELATIONSHIP_HINTS = {"relationship", "teacher", "sensei", "affection"}
_PREFERENCE_HINTS = {"preference", "pref", "style"}
_PROJECT_HINTS = {"project", "repo", "runtime", "hermes", "pcltm", "soul-link"}
_PROCEDURAL_HINTS = {"procedure", "workflow", "skill", "howto", "runbook"}
_STATE_HINTS = {"state", "emotion", "mode"}
_TOOL_HINTS = {"tool", "evidence", "terminal", "browser"}
_CONFLICT_HINTS = {"conflict", "quarantine", "pending_conflict"}
_RETIRED_HINTS = {"retired", "stale", "superseded"}


class MemoryObjectAdapter:
    """Read-only adapter from existing PCLTM/MemFS shapes to MemoryObject."""

    def from_mapping(self, record: Mapping[str, Any]) -> MemoryObject:
        """Convert a legacy mapping without mutating it or existing runtime paths."""

        metadata = _as_mapping(record.get("metadata"))
        merged = {**metadata, **record}
        canonical_key = _first_text(
            merged,
            "canonical_key",
            "memory_id",
            "record_id",
            "id",
            default=_stable_key(record),
        )
        content = _first_text(
            merged,
            "content",
            "value",
            "body",
            "text",
            "description",
            default=canonical_key,
        )
        target_file = _first_text(merged, "target_file", "file", default="")
        layer = _first_text(merged, "layer", default="")
        buckets = _string_tuple(merged.get("buckets"))
        tags = _string_tuple(merged.get("tags")) or buckets
        hints = _hint_set(target_file, layer, buckets, tags, content, canonical_key)
        status = _status_from(merged)
        object_type = _explicit_object_type_from(merged) or _object_type_from(hints, target_file, layer, status)
        injection_policy = _injection_policy_from(merged, object_type, status, layer, target_file)

        return MemoryObject(
            canonical_key=canonical_key,
            object_type=object_type,
            content=content,
            scope=_scope_from(merged),
            status=status,
            injection_policy=injection_policy,
            source=_optional_text(merged.get("source")) or _optional_text(merged.get("path")),
            confidence=_float_unit(merged.get("confidence"), 1.0),
            stability=_float_unit(merged.get("stability"), _default_stability(object_type, layer)),
            emotional_weight=_float_unit(merged.get("emotional_weight"), 0.0),
            budget_weight=_positive_float(merged.get("budget_weight"), 1.0),
            state_affinity=StateAffinity(
                modes=_string_tuple(merged.get("mode_scope") or merged.get("modes")),
                emotion_axes=_string_tuple(merged.get("emotion_axes")),
                min_intensity=_optional_text(merged.get("min_intensity")),
            ),
            tags=tags,
            conflict_keys=_string_tuple(merged.get("conflict_keys") or merged.get("conflict_set")),
            metadata={k: v for k, v in metadata.items()},
        )

    def from_memfs_item(self, item: Any) -> MemoryObject:
        """Convert a MemFS dataclass-like item using public attributes only."""

        if isinstance(item, Mapping):
            return self.from_mapping(item)
        data = {
            "memory_id": getattr(item, "memory_id", None),
            "path": getattr(item, "path", None),
            "layer": getattr(item, "layer", None),
            "description": getattr(item, "description", None),
            "authority": getattr(item, "authority", None),
            "buckets": getattr(item, "buckets", None),
            "mode_scope": getattr(item, "mode_scope", None),
            "metadata": getattr(item, "metadata", None),
        }
        return self.from_mapping(data)


def adapt_memory_object(record: Mapping[str, Any] | Any) -> MemoryObject:
    """Convenience wrapper for one-off read-only conversions."""

    adapter = MemoryObjectAdapter()
    if isinstance(record, Mapping):
        return adapter.from_mapping(record)
    return adapter.from_memfs_item(record)


def _status_from(data: Mapping[str, Any]) -> MemoryObjectStatus:
    raw = _first_text(data, "status", "approval_status", default="pending").lower()
    if raw in {"approved", "active", "accepted"}:
        return MemoryObjectStatus.APPROVED
    if raw in {"quarantined", "quarantine", "blocked", "conflict"}:
        return MemoryObjectStatus.QUARANTINED
    if raw in {"retired", "stale", "superseded", "deleted"}:
        return MemoryObjectStatus.RETIRED
    return MemoryObjectStatus.PENDING


def _explicit_object_type_from(record: Mapping[str, Any]) -> MemoryObjectType | None:
    raw = record.get("object_type") or record.get("memory_type") or record.get("type")
    if not isinstance(raw, str):
        return None
    value = raw.strip().lower()
    aliases = {
        "identity": MemoryObjectType.IDENTITY,
        "identity_memory": MemoryObjectType.IDENTITY,
        "relationship": MemoryObjectType.RELATIONSHIP,
        "relationship_memory": MemoryObjectType.RELATIONSHIP,
        "preference": MemoryObjectType.PREFERENCE,
        "preference_memory": MemoryObjectType.PREFERENCE,
        "project": MemoryObjectType.PROJECT,
        "project_memory": MemoryObjectType.PROJECT,
        "procedural": MemoryObjectType.PROCEDURAL,
        "procedural_memory": MemoryObjectType.PROCEDURAL,
        "episodic": MemoryObjectType.EPISODIC,
        "episodic_memory": MemoryObjectType.EPISODIC,
        "state": MemoryObjectType.STATE_TRACE,
        "state_trace": MemoryObjectType.STATE_TRACE,
        "state_trace_memory": MemoryObjectType.STATE_TRACE,
        "tool": MemoryObjectType.TOOL_EVIDENCE,
        "tool_evidence": MemoryObjectType.TOOL_EVIDENCE,
        "tool_evidence_memory": MemoryObjectType.TOOL_EVIDENCE,
        "conflict": MemoryObjectType.CONFLICT,
        "conflict_memory": MemoryObjectType.CONFLICT,
        "retired": MemoryObjectType.RETIRED,
        "retired_memory": MemoryObjectType.RETIRED,
    }
    return aliases.get(value)


def _object_type_from(
    hints: set[str],
    target_file: str,
    layer: str,
    status: MemoryObjectStatus,
) -> MemoryObjectType:
    if status is MemoryObjectStatus.RETIRED or hints & _RETIRED_HINTS:
        return MemoryObjectType.RETIRED
    if hints & _CONFLICT_HINTS:
        return MemoryObjectType.CONFLICT
    if target_file == "USER.md" or layer == "pinned" or hints & _IDENTITY_HINTS:
        return MemoryObjectType.IDENTITY
    if hints & _RELATIONSHIP_HINTS:
        return MemoryObjectType.RELATIONSHIP
    if hints & _PREFERENCE_HINTS:
        return MemoryObjectType.PREFERENCE
    if hints & _PROCEDURAL_HINTS:
        return MemoryObjectType.PROCEDURAL
    if hints & _STATE_HINTS:
        return MemoryObjectType.STATE_TRACE
    if hints & _TOOL_HINTS:
        return MemoryObjectType.TOOL_EVIDENCE
    if hints & _PROJECT_HINTS:
        return MemoryObjectType.PROJECT
    return MemoryObjectType.EPISODIC


def _injection_policy_from(
    data: Mapping[str, Any],
    object_type: MemoryObjectType,
    status: MemoryObjectStatus,
    layer: str,
    target_file: str,
) -> InjectionPolicy:
    raw = _optional_text(data.get("injection_policy"))
    if raw:
        policy = InjectionPolicy(raw)
    elif status is MemoryObjectStatus.RETIRED:
        policy = InjectionPolicy.NEVER
    elif object_type is MemoryObjectType.IDENTITY or layer == "pinned" or target_file == "USER.md":
        policy = InjectionPolicy.PINNED
    elif object_type in {MemoryObjectType.TOOL_EVIDENCE, MemoryObjectType.STATE_TRACE}:
        policy = InjectionPolicy.EVIDENCE_ONLY
    else:
        policy = InjectionPolicy.SELECTIVE
    if status is MemoryObjectStatus.RETIRED:
        return InjectionPolicy.NEVER
    if status is MemoryObjectStatus.QUARANTINED and policy is InjectionPolicy.PINNED:
        return InjectionPolicy.EVIDENCE_ONLY
    return policy


def _scope_from(data: Mapping[str, Any]) -> MemoryObjectScope:
    raw = _first_text(data, "scope", default="user").lower()
    if raw in {item.value for item in MemoryObjectScope}:
        return MemoryObjectScope(raw)
    return MemoryObjectScope.USER


def _default_stability(object_type: MemoryObjectType, layer: str) -> float:
    if object_type is MemoryObjectType.IDENTITY or layer == "pinned":
        return 1.0
    if object_type is MemoryObjectType.EPISODIC:
        return 0.35
    return 0.5


def _hint_set(*values: Any) -> set[str]:
    hints: set[str] = set()
    for value in values:
        for item in _string_tuple(value):
            hints.update(token for token in re.split(r"[^a-z0-9_-]+", item.lower()) if token)
    return hints


def _first_text(data: Mapping[str, Any], *keys: str, default: str) -> str:
    for key in keys:
        value = data.get(key)
        text = _optional_text(value)
        if text:
            return text
    return default


def _optional_text(value: Any) -> str | None:
    if value is None:
        return None
    text = str(value).strip()
    return text or None


def _string_tuple(value: Any) -> tuple[str, ...]:
    if value is None:
        return ()
    if isinstance(value, str):
        return tuple(part.strip() for part in value.split(",") if part.strip())
    if isinstance(value, Mapping):
        return tuple(str(key) for key in value.keys())
    try:
        return tuple(str(item).strip() for item in value if str(item).strip())
    except TypeError:
        return (str(value).strip(),) if str(value).strip() else ()


def _as_mapping(value: Any) -> Mapping[str, Any]:
    return value if isinstance(value, Mapping) else {}


def _float_unit(value: Any, default: float) -> float:
    try:
        parsed = float(value)
    except (TypeError, ValueError):
        return default
    return min(1.0, max(0.0, parsed))


def _positive_float(value: Any, default: float) -> float:
    try:
        parsed = float(value)
    except (TypeError, ValueError):
        return default
    return parsed if parsed > 0 else default


def _stable_key(record: Mapping[str, Any]) -> str:
    seed = _optional_text(record.get("path")) or _optional_text(record.get("description")) or "memory-object"
    return re.sub(r"[^a-zA-Z0-9_.:-]+", "-", seed).strip("-") or "memory-object"
