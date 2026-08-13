"""Pure rendering helpers for bounded, redacted tool-result capsules."""

from __future__ import annotations

import ast
import json
import re
from collections.abc import Callable, Mapping
from typing import Any

Redactor = Callable[[str], str]


def tool_capsule_kind(tool_name: str, text: str) -> str:
    normalized = (tool_name or "").strip().lower()
    stripped = (text or "").lstrip()
    if normalized in {"skill_view", "skills_view"} or (
        '"readiness_status"' in stripped
        and '"linked_files"' in stripped
        and '"usage_hint"' in stripped
        and '"content"' in stripped
    ):
        return "skill"
    if normalized in {"read_file", "search_files"} or '"total_lines"' in stripped or '"matches"' in stripped:
        return "file"
    if normalized in {"terminal", "execute_code", "process"} or '"exit_code"' in stripped or '"stdout"' in stripped:
        return "terminal"
    if normalized in {"web_extract", "web_search", "browser_console", "browser_snapshot"} or '"url"' in stripped:
        return "web"
    if stripped.startswith("{") or stripped.startswith("["):
        return "json"
    return "generic"


def _parse_structured(raw_content: Any, text: str) -> Any:
    if isinstance(raw_content, (Mapping, list)):
        return raw_content
    for value in (raw_content, text):
        if isinstance(value, str) and value.strip().startswith(("{", "[")):
            try:
                return json.loads(value.strip())
            except json.JSONDecodeError:
                try:
                    return ast.literal_eval(value.strip())
                except (SyntaxError, ValueError):
                    pass
    return None


def _fallback_redact(text: str) -> str:
    secret_pattern = (
        r"(?i)(authorization\s*[:=]\s*bearer\s+)[A-Za-z0-9._~+/-]{8,}|"
        r"\bsk-[A-Za-z0-9_-]{16,}\b|\bghp_[A-Za-z0-9_]{20,}\b|"
        r"\bgithub_pat_[A-Za-z0-9_]{20,}\b|\bxox[baprs]-[A-Za-z0-9-]{12,}\b"
    )
    return re.sub(
        secret_pattern,
        lambda match: match.group(1) + "[REDACTED]" if match.group(1) else "[REDACTED]",
        text,
    )


def _truncate(text: str, limit: int) -> str:
    text = " ".join(text.split())
    return text if len(text) <= limit else text[: max(0, limit - 1)] + "…"


def _edge_excerpt(text: str, limit: int) -> str:
    normalized = " ".join(text.split())
    if len(normalized) <= limit:
        return normalized
    tail_limit = max(80, limit // 3)
    return normalized[: max(0, limit - tail_limit - 3)] + " … " + normalized[-tail_limit:]


def _json_excerpt(text: str, limit: int, raw_content: Any, redact: Redactor) -> str:
    parsed = _parse_structured(raw_content, text)
    if not isinstance(parsed, dict):
        return _truncate(redact(text), limit)
    safe_keys = (
        "path",
        "url",
        "title",
        "description",
        "total_lines",
        "exit_code",
        "error",
        "status",
        "count",
        "memory_id",
        "line",
        "offset",
        "limit",
        "command",
    )
    safe = {key: parsed[key] for key in safe_keys if key in parsed}
    for key in ("content", "output"):
        if isinstance(parsed.get(key), str):
            safe[f"{key}_excerpt"] = _edge_excerpt(redact(parsed[key]), 300)
    return redact(json.dumps(safe, ensure_ascii=False)) if safe else _truncate(redact(text), limit)


def render_tool_capsule(
    *,
    tool_name: str,
    tool_call_id: str,
    text: str,
    raw_content: Any = None,
    kind: str,
    preserve_edges: bool,
    redact: Redactor | None = None,
) -> str:
    redactor = redact or _fallback_redact
    text = redactor(text)
    labels = {
        "file": "[File tool evidence capsule for active context]",
        "terminal": "[Terminal tool evidence capsule for active context]",
        "web": "[Web/browser tool evidence capsule for active context]",
        "json": "[Structured tool evidence capsule for active context]",
        "generic": "[Tool result truncated for active context]",
    }
    head_limit = 900 if preserve_edges else 700
    head = (
        _json_excerpt(text, head_limit, raw_content, redactor)
        if kind in {"json", "file", "terminal", "web"}
        else _truncate(text, head_limit)
    )
    tail = _truncate(text[-120:], 120) if preserve_edges and kind == "generic" else ""
    lines = [
        labels.get(kind, labels["generic"]),
        f"tool={tool_name}",
        f"tool_call_id={tool_call_id}",
        f"capsule_kind={kind}",
        f"original_chars={len(text)}",
        f"dropped_chars={max(0, len(text) - len(head) - len(tail))}",
        "head_excerpt:",
        head,
    ]
    if tail:
        lines.extend(("tail_excerpt:", tail))
    lines.append(
        "Full result remains available in the session transcript/tool logs; "
        "use explicit file/session/PCLTM retrieval if more detail is needed."
    )
    return "\n".join(lines)


def tool_capsule_indicates_error(kind: str, capsule_text: str, original_text: str = "") -> bool:
    parsed = _parse_structured(None, original_text)
    if isinstance(parsed, dict):
        if parsed.get("exit_code") is not None:
            try:
                return int(parsed["exit_code"]) != 0
            except (TypeError, ValueError):
                return str(parsed["exit_code"]).strip() not in {"", "0", "None", "none"}
        if any(
            isinstance(parsed.get(key), str) and parsed[key].strip()
            for key in ("error", "exception", "traceback")
        ):
            return True
    lowered = f"{capsule_text}\n{original_text}".lower()
    if kind == "terminal" and ('"exit_code": 0' in lowered or "exit_code=0" in lowered):
        return False
    return any(
        marker in lowered
        for marker in ("traceback", "exception", "error", "failed", "failure", "non-zero", "exit_code")
    )
