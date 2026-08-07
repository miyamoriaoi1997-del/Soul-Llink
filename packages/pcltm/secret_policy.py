"""Secret-safe memory write and render policy for PCLTM.

This module is intentionally independent from the store/adapter layers so every
memory write/read surface can share one boundary:

- durable memory may store connection metadata and credential references;
- durable memory must not store raw secret values;
- model-visible output must redact secret-like strings defensively.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from typing import Any, Literal

SecretAction = Literal["allow", "sanitize", "reject"]
SecretSensitivity = Literal["normal", "private", "restricted", "secret"]


@dataclass(frozen=True)
class SecretPolicyDecision:
    allowed: bool
    action: SecretAction
    sanitized_content: str | None
    sensitivity: SecretSensitivity
    reason: str
    detected_categories: list[str] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)


_SECRET_PATTERNS: tuple[tuple[str, re.Pattern[str]], ...] = (
    (
        "secret_assignment",
        re.compile(
            r"(?i)\b[A-Z0-9_]*(?:SECRET|TOKEN|API[_-]?KEY|PASSWORD|PASS|COOKIE|SESSION)[A-Z0-9_]*\s*=\s*[^\s,;]+"
        ),
    ),
    ("bearer_token", re.compile(r"(?i)\bBearer\s+[A-Za-z0-9._\-]{12,}")),
    ("openai_key_like", re.compile(r"\bsk-[A-Za-z0-9_-]{16,}\b")),
    ("github_token_like", re.compile(r"\bghp_[A-Za-z0-9_]{20,}\b")),
    ("github_pat_like", re.compile(r"\bgithub_pat_[A-Za-z0-9_]{20,}\b")),
    ("slack_token_like", re.compile(r"\bxox[baprs]-[A-Za-z0-9-]{12,}\b")),
    ("jwt_like", re.compile(r"\beyJ[A-Za-z0-9_-]{8,}\.[A-Za-z0-9_-]{8,}\.[A-Za-z0-9_-]{8,}\b")),
    ("pem_private_key", re.compile(r"-----BEGIN [A-Z ]*PRIVATE KEY-----[\s\S]*?-----END [A-Z ]*PRIVATE KEY-----")),
    ("password_bearing_url", re.compile(r"\b[a-z][a-z0-9+.-]*://([^:/@\s]+):([^@\s]+)@([^\s]+)", re.I)),
)

_SECRET_VALUE_PLACEHOLDER = "[REDACTED_SECRET]"
_REJECTED_SECRET_PLACEHOLDER = "[REJECTED_SECRET_MEMORY: raw secret value omitted]"

_STRUCTURED_SECRET_KEYS = frozenset(
    {
        "token",
        "access_token",
        "refresh_token",
        "id_token",
        "api_key",
        "apikey",
        "client_secret",
        "private_key",
        "password",
        "passwd",
        "authorization",
        "cookie",
        "session",
        "session_id",
    }
)

_ENV_REF_RE = re.compile(r"\b[A-Z][A-Z0-9_]{2,}(?:SECRET|TOKEN|API_KEY|APIKEY|PASSWORD|PASS|COOKIE|SESSION|DATABASE_URL)[A-Z0-9_]*\b")
_SECRET_REF_RE = re.compile(r"\b(?:secret|op|bw|vault)://[^\s,;]+", re.I)
_HOST_RE = re.compile(r"\b(?:\d{1,3}\.){3}\d{1,3}\b|\b[a-zA-Z0-9][a-zA-Z0-9.-]+\.[a-zA-Z]{2,}\b")
_USER_RE = re.compile(r"(?i)(?:\buser(?:name)?\s*=\s*|\b用户\s*|\bssh\s+)([a-zA-Z0-9._-]{1,64})(?:@|\b)")
_PORT_RE = re.compile(r"(?i)\bport\s*=\s*(\d{1,5})\b|\b端口\s*(\d{1,5})\b")
_PATH_RE = re.compile(r"(?<!\w)(/(?:root|srv|opt|var|home|mnt|data|www|app)[^\s,;，。]*)")


def _is_structured_secret_key(key: object) -> bool:
    normalized = re.sub(r"[-\s]+", "_", str(key).strip().lower())
    return normalized in _STRUCTURED_SECRET_KEYS or normalized.endswith(
        ("_token", "_secret", "_password", "_passwd", "_private_key", "_api_key")
    )


def _transform_structured_secrets(value: Any) -> tuple[Any, bool]:
    found = False
    if isinstance(value, dict):
        transformed: dict[Any, Any] = {}
        for key, child in value.items():
            if _is_structured_secret_key(key):
                transformed[key] = _SECRET_VALUE_PLACEHOLDER
                found = True
            else:
                transformed[key], child_found = _transform_structured_secrets(child)
                found = found or child_found
        return transformed, found
    if isinstance(value, list):
        transformed_list = []
        for child in value:
            transformed_child, child_found = _transform_structured_secrets(child)
            transformed_list.append(transformed_child)
            found = found or child_found
        return transformed_list, found
    if isinstance(value, str):
        transformed_value = value
        for category, pattern in _SECRET_PATTERNS:
            if category == "password_bearing_url":
                transformed_value = pattern.sub(
                    lambda match: match.group(0).replace(
                        f":{match.group(2)}@", f":{_SECRET_VALUE_PLACEHOLDER}@",
                    ),
                    transformed_value,
                )
            else:
                transformed_value = pattern.sub(_SECRET_VALUE_PLACEHOLDER, transformed_value)
        return transformed_value, transformed_value != value
    return value, False


def _parse_structured_text(text: str) -> Any | None:
    try:
        value = json.loads(text)
    except (json.JSONDecodeError, TypeError):
        return None
    return value if isinstance(value, (dict, list)) else None


def detect_secret_categories(text: str) -> list[str]:
    """Return categories of secret-like values found in text."""
    found: list[str] = []
    for category, pattern in _SECRET_PATTERNS:
        if pattern.search(text or ""):
            found.append(category)
    structured = _parse_structured_text(text or "")
    if structured is not None and _transform_structured_secrets(structured)[1]:
        found.append("structured_secret_field")
    return found


def contains_secret_value(text: str) -> bool:
    return bool(detect_secret_categories(text))


def redact_secrets(text: str) -> str:
    """Mask secret-like values before content reaches model-visible surfaces."""
    if not text:
        return text
    structured = _parse_structured_text(text)
    if structured is not None:
        transformed, found = _transform_structured_secrets(structured)
        if found:
            return json.dumps(transformed, ensure_ascii=False)
    redacted = text
    for category, pattern in _SECRET_PATTERNS:
        if category == "password_bearing_url":
            redacted = pattern.sub(lambda m: m.group(0).replace(f":{m.group(2)}@", f":{_SECRET_VALUE_PLACEHOLDER}@"), redacted)
        else:
            redacted = pattern.sub(_SECRET_VALUE_PLACEHOLDER, redacted)
    return redacted


def _unique(values: list[str]) -> list[str]:
    seen: set[str] = set()
    result: list[str] = []
    for value in values:
        if value and value not in seen:
            seen.add(value)
            result.append(value)
    return result


def extract_safe_connection_metadata(text: str) -> str | None:
    """Extract useful non-secret operational metadata from a mixed secret input."""
    parts: list[str] = []
    hosts = _unique([m.group(0) for m in _HOST_RE.finditer(text)])
    users = _unique([m.group(1) for m in _USER_RE.finditer(text) if m.group(1)])
    ports = _unique([next(g for g in m.groups() if g) for m in _PORT_RE.finditer(text) if any(m.groups())])
    paths = _unique([m.group(1) for m in _PATH_RE.finditer(text)])
    env_refs = _unique([m.group(0) for m in _ENV_REF_RE.finditer(text)])
    secret_refs = _unique([m.group(0) for m in _SECRET_REF_RE.finditer(text)])

    if hosts:
        parts.append("hosts=" + ",".join(hosts[:4]))
    if users:
        parts.append("users=" + ",".join(users[:3]))
    if ports:
        parts.append("ports=" + ",".join(ports[:3]))
    if paths:
        parts.append("paths=" + ",".join(paths[:4]))
    if env_refs:
        parts.append("env_refs=" + ",".join(env_refs[:5]))
    if secret_refs:
        parts.append("secret_refs=" + ",".join(secret_refs[:5]))

    if not parts:
        return None
    return "Connection/credential metadata: " + "; ".join(parts) + ". Secret values are not stored; ask the user for the missing secret when needed."


def _is_reference_only(text: str) -> bool:
    if contains_secret_value(text):
        return False
    return bool(_ENV_REF_RE.search(text) or _SECRET_REF_RE.search(text))


def evaluate_memory_write(content: str, *, target_file: str) -> SecretPolicyDecision:
    """Decide whether and how a memory write can be persisted safely.

    The function never returns raw secret values in sanitized_content for sanitize
    or reject decisions.
    """
    text = (content or "").strip()
    categories = detect_secret_categories(text)
    base_meta: dict[str, Any] = {"secret_policy_version": "secret-safe-memory-v1"}

    if not categories:
        if _is_reference_only(text):
            return SecretPolicyDecision(
                allowed=True,
                action="allow",
                sanitized_content=text,
                sensitivity="normal",
                reason="credential reference only; no raw secret value detected",
                detected_categories=[],
                metadata={
                    **base_meta,
                    "secret_boundary": True,
                    "credential_reference_only": True,
                    "contains_secret_value": False,
                },
            )
        return SecretPolicyDecision(
            allowed=True,
            action="allow",
            sanitized_content=text,
            sensitivity="normal",
            reason="no secret-like value detected",
            detected_categories=[],
            metadata={**base_meta, "contains_secret_value": False},
        )

    safe_metadata = extract_safe_connection_metadata(text)
    if safe_metadata:
        return SecretPolicyDecision(
            allowed=True,
            action="sanitize",
            sanitized_content=safe_metadata,
            sensitivity="normal",
            reason="secret value removed; safe connection/reference metadata retained",
            detected_categories=categories,
            metadata={
                **base_meta,
                "secret_boundary": True,
                "sanitized_from_secret": True,
                "removed_secret_categories": categories,
                "contains_secret_value": False,
            },
        )

    return SecretPolicyDecision(
        allowed=False,
        action="reject",
        sanitized_content=_REJECTED_SECRET_PLACEHOLDER,
        sensitivity="secret",
        reason="raw secret value cannot be stored in durable memory",
        detected_categories=categories,
        metadata={
            **base_meta,
            "secret_boundary": True,
            "rejected_raw_secret": True,
            "removed_secret_categories": categories,
            "contains_secret_value": False,
        },
    )
