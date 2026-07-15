"""Versioned, JSON-safe contracts exposed by the read-only monitor."""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Mapping, Sequence

SCHEMA_VERSION = 1
_FORBIDDEN_FIELD_NAMES = frozenset(
    {
        "api_key",
        "authorization",
        "content",
        "memory_text",
        "prompt",
        "tool_output",
    }
)
_VALID_SEVERITIES = frozenset({"info", "warning", "error"})


def _iso_timestamp(value: datetime | str | None) -> str:
    if value is None:
        value = datetime.now(timezone.utc)
    if isinstance(value, datetime):
        if value.tzinfo is None:
            value = value.replace(tzinfo=timezone.utc)
        return value.isoformat()
    return str(value)


def _validate_safe(value: Any, *, depth: int = 0) -> None:
    if depth > 16:
        raise ValueError("monitoring value exceeds maximum nesting depth")
    if value is None or isinstance(value, (bool, int, float, str)):
        return
    if isinstance(value, Mapping):
        for raw_key, child in value.items():
            key = str(raw_key).strip().lower()
            if key in _FORBIDDEN_FIELD_NAMES:
                raise ValueError(f"forbidden monitoring field: {key}")
            _validate_safe(child, depth=depth + 1)
        return
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray)):
        for child in value:
            _validate_safe(child, depth=depth + 1)
        return
    raise TypeError(f"monitoring value is not JSON-safe: {type(value).__name__}")


@dataclass(frozen=True)
class Issue:
    severity: str
    code: str
    source: str
    message: str
    timestamp: datetime | str
    remediation: str

    def __post_init__(self) -> None:
        if self.severity not in _VALID_SEVERITIES:
            raise ValueError(f"invalid issue severity: {self.severity}")
        _validate_safe(self.to_dict())

    def to_dict(self) -> dict[str, str]:
        return {
            "severity": self.severity,
            "code": self.code,
            "source": self.source,
            "message": self.message,
            "timestamp": _iso_timestamp(self.timestamp),
            "remediation": self.remediation,
        }


@dataclass(frozen=True)
class Snapshot:
    generated_at: datetime | str | None = None
    duration_ms: int = 0
    runtime: Mapping[str, Any] = field(default_factory=dict)
    context: Mapping[str, Any] = field(default_factory=dict)
    memory: Mapping[str, Any] = field(default_factory=dict)
    persona: Mapping[str, Any] = field(default_factory=dict)
    router: Mapping[str, Any] = field(default_factory=dict)
    issues: Sequence[Issue] = field(default_factory=tuple)

    def __post_init__(self) -> None:
        for section in (
            self.runtime,
            self.context,
            self.memory,
            self.persona,
            self.router,
        ):
            _validate_safe(section)
        for issue in self.issues:
            if not isinstance(issue, Issue):
                raise TypeError("snapshot issues must be Issue instances")

    @classmethod
    def empty(
        cls,
        *,
        generated_at: datetime | str | None = None,
        duration_ms: int = 0,
    ) -> "Snapshot":
        return cls(generated_at=generated_at, duration_ms=duration_ms)

    @property
    def ok(self) -> bool:
        return not any(issue.severity == "error" for issue in self.issues)

    def to_dict(self) -> dict[str, Any]:
        payload = {
            "schema_version": SCHEMA_VERSION,
            "ok": self.ok,
            "generated_at": _iso_timestamp(self.generated_at),
            "duration_ms": int(self.duration_ms),
            "runtime": dict(self.runtime),
            "context": dict(self.context),
            "memory": dict(self.memory),
            "persona": dict(self.persona),
            "router": dict(self.router),
            "issues": [issue.to_dict() for issue in self.issues],
        }
        _validate_safe(payload)
        return payload
