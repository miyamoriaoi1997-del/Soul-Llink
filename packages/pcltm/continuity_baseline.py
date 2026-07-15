"""Pure normalization of existing SoulLink continuity artifacts.

This module is a baseline *producer*, not a replay runner.  It accepts artifacts
that the current runtime has already produced and copies them into a stable,
JSON-safe envelope.  It performs no retrieval, inference, database access, or
runtime switching.
"""

from __future__ import annotations

import json
import math
from dataclasses import dataclass
from enum import Enum
from types import MappingProxyType
from typing import Any, Mapping, Sequence

AUTHORITY_BOUNDARY = "read_only_artifact_normalization"
OBJECT_TYPE = "soul_link_continuity_baseline"


def _non_empty_text(value: Any, field_name: str) -> str:
    if not isinstance(value, str):
        raise ValueError(f"{field_name} must be a string")
    text = value.strip()
    if not text:
        raise ValueError(f"{field_name} must be non-empty")
    return text


def _artifact_mapping(value: Any, field_name: str) -> Mapping[str, Any]:
    if isinstance(value, Mapping):
        return value
    for method_name in ("to_dict", "to_payload", "as_prompt_anchor"):
        method = getattr(value, method_name, None)
        if callable(method):
            produced = method()
            if isinstance(produced, Mapping):
                return produced
    raise ValueError(f"{field_name} must be a mapping or expose a mapping serializer")


def _json_safe_copy(value: Any) -> Any:
    if isinstance(value, Enum):
        return _json_safe_copy(value.value)
    if isinstance(value, Mapping):
        output: dict[str, Any] = {}
        for key, item in value.items():
            if not isinstance(key, str):
                raise ValueError("artifact mapping keys must be strings")
            if key in output:
                raise ValueError(f"artifact contains duplicate mapping key: {key}")
            output[key] = _json_safe_copy(item)
        return output
    if isinstance(value, (list, tuple)):
        return [_json_safe_copy(item) for item in value]
    if isinstance(value, float) and not math.isfinite(value):
        raise ValueError("artifact contains a non-finite float")
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    raise ValueError(f"artifact contains unsupported JSON value: {type(value).__name__}")


def _unique_refs(values: Sequence[str]) -> tuple[str, ...]:
    if isinstance(values, (str, bytes)):
        raise ValueError("evidence_refs must be a sequence of strings")
    output: list[str] = []
    for value in values:
        if not isinstance(value, str):
            raise ValueError("evidence_refs must contain only strings")
        item = value.strip()
        if item and item not in output:
            output.append(item)
    if not output:
        raise ValueError("evidence_refs must contain at least one reference")
    return tuple(output)


@dataclass(frozen=True)
class ContinuityBaselineArtifact:
    baseline_id: str
    case_id: str
    identity: Mapping[str, Any]
    conversation: Mapping[str, Any]
    active_dialogue: Mapping[str, Any]
    summary_chain: Mapping[str, Any]
    evidence_refs: tuple[str, ...]
    schema_version: int = 1
    object_type: str = OBJECT_TYPE
    authority_boundary: str = AUTHORITY_BOUNDARY

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "object_type": self.object_type,
            "authority_boundary": self.authority_boundary,
            "baseline_id": self.baseline_id,
            "case_id": self.case_id,
            "identity": _json_safe_copy(self.identity),
            "conversation": _json_safe_copy(self.conversation),
            "active_dialogue": _json_safe_copy(self.active_dialogue),
            "summary_chain": _json_safe_copy(self.summary_chain),
            "evidence_refs": list(self.evidence_refs),
        }

    def to_json(self) -> str:
        return json.dumps(
            self.to_dict(), ensure_ascii=False, indent=2, sort_keys=True, allow_nan=False
        ) + "\n"


def build_continuity_baseline(
    *,
    baseline_id: str,
    case_id: str,
    identity_anchor: Any,
    conversation_snapshot: Any,
    active_dialogue_state: Any,
    session_summary_chain: Any,
    evidence_refs: Sequence[str],
) -> ContinuityBaselineArtifact:
    """Copy existing runtime artifacts into a deterministic baseline envelope."""
    identity = _json_safe_copy(_artifact_mapping(identity_anchor, "identity_anchor"))
    conversation = _json_safe_copy(
        _artifact_mapping(conversation_snapshot, "conversation_snapshot")
    )
    active_dialogue = _json_safe_copy(
        _artifact_mapping(active_dialogue_state, "active_dialogue_state")
    )
    summary_chain = _json_safe_copy(
        _artifact_mapping(session_summary_chain, "session_summary_chain")
    )
    identity_value = identity.get("identity")
    if not isinstance(identity_value, str) or not identity_value.strip():
        raise ValueError("identity_anchor.identity must be a non-empty string")
    if identity.get("read_only") is not True:
        raise ValueError("identity_anchor.read_only must be true")
    for field_name, artifact in (
        ("conversation_snapshot", conversation),
        ("active_dialogue_state", active_dialogue),
        ("session_summary_chain", summary_chain),
    ):
        if not artifact:
            raise ValueError(f"{field_name} must be non-empty")

    return ContinuityBaselineArtifact(
        baseline_id=_non_empty_text(baseline_id, "baseline_id"),
        case_id=_non_empty_text(case_id, "case_id"),
        identity=MappingProxyType(identity),
        conversation=MappingProxyType(conversation),
        active_dialogue=MappingProxyType(active_dialogue),
        summary_chain=MappingProxyType(summary_chain),
        evidence_refs=_unique_refs(evidence_refs),
    )
