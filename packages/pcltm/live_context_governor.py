"""Live context budget governance for PCLTM prompt surfaces.

This module governs *prompt-time views*, not durable memory.  It is designed to
preserve task continuity under a hard budget by converting bulky context into
sealed continuation and evidence capsules, then truncating low-priority memory
text first.
"""

from __future__ import annotations

import hashlib
import json
import re
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Iterable, Mapping, Sequence

from .secret_policy import redact_secrets


@dataclass(frozen=True)
class ContextBudgetPolicy:
    """Character-budget policy for a governed PCLTM prompt view."""

    total_chars: int = 3600
    continuation_chars: int = 900
    evidence_chars: int = 700
    memory_chars: int = 1600
    telemetry_chars: int = 300

    def __post_init__(self) -> None:
        for name in ("total_chars", "continuation_chars", "evidence_chars", "memory_chars", "telemetry_chars"):
            value = int(getattr(self, name))
            if value < 0:
                raise ValueError(f"{name} must be >= 0")
            object.__setattr__(self, name, value)
        if self.total_chars <= 0:
            raise ValueError("total_chars must be positive")


@dataclass(frozen=True)
class ContinuationCapsule:
    """Sealed short state that preserves continuity without replaying transcript."""

    conversation_goal: str = ""
    current_task: str = ""
    completed: tuple[str, ...] = ()
    open_threads: tuple[str, ...] = ()
    constraints: tuple[str, ...] = ()
    latest_verified_state: Mapping[str, str] = field(default_factory=dict)
    recovery_handles: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        object.__setattr__(self, "conversation_goal", _clean(self.conversation_goal, 220))
        object.__setattr__(self, "current_task", _clean(self.current_task, 260))
        object.__setattr__(self, "completed", _clean_tuple(self.completed, max_items=6, max_chars=120))
        object.__setattr__(self, "open_threads", _clean_tuple(self.open_threads, max_items=6, max_chars=140))
        object.__setattr__(self, "constraints", _clean_tuple(self.constraints, max_items=6, max_chars=140))
        object.__setattr__(
            self,
            "latest_verified_state",
            {str(k)[:80]: _clean(v, 140) for k, v in dict(self.latest_verified_state or {}).items()},
        )
        object.__setattr__(self, "recovery_handles", _clean_tuple(self.recovery_handles, max_items=6, max_chars=160))

    @property
    def is_empty(self) -> bool:
        return not any(
            (
                self.conversation_goal,
                self.current_task,
                self.completed,
                self.open_threads,
                self.constraints,
                self.latest_verified_state,
                self.recovery_handles,
            )
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "conversation_goal": self.conversation_goal,
            "current_task": self.current_task,
            "completed": list(self.completed),
            "open_threads": list(self.open_threads),
            "constraints": list(self.constraints),
            "latest_verified_state": dict(self.latest_verified_state),
            "recovery_handles": list(self.recovery_handles),
        }

    @classmethod
    def from_dict(cls, data: Mapping[str, Any] | None) -> "ContinuationCapsule":
        data = dict(data or {})
        return cls(
            conversation_goal=str(data.get("conversation_goal") or ""),
            current_task=str(data.get("current_task") or ""),
            completed=tuple(str(item) for item in data.get("completed") or ()),
            open_threads=tuple(str(item) for item in data.get("open_threads") or ()),
            constraints=tuple(str(item) for item in data.get("constraints") or ()),
            latest_verified_state={str(k): str(v) for k, v in dict(data.get("latest_verified_state") or {}).items()},
            recovery_handles=tuple(str(item) for item in data.get("recovery_handles") or ()),
        )

    def render(self, *, max_chars: int = 900) -> str:
        if max_chars <= 0 or self.is_empty:
            return ""
        lines = [
            "【continuation_capsule】",
            "  semantics: sealed_state_not_transcript",
        ]
        if self.current_task:
            lines.append(f"  current_task: {self.current_task}")
        if self.latest_verified_state:
            verified = "; ".join(f"{k}={v}" for k, v in self.latest_verified_state.items())
            lines.append(f"  latest_verified_state: {verified}")
        if self.conversation_goal:
            lines.append(f"  conversation_goal: {self.conversation_goal}")
        if self.completed:
            lines.append("  completed: " + " | ".join(self.completed))
        if self.open_threads:
            lines.append("  open_threads: " + " | ".join(self.open_threads))
        if self.constraints:
            lines.append("  constraints: " + " | ".join(self.constraints))
        if self.recovery_handles:
            lines.append("  recovery_handles: " + " | ".join(self.recovery_handles))
        required = ("current_task", self.current_task, "latest_verified_state", *self.latest_verified_state.keys())
        return _truncate_preserving_markers("\n".join(lines), max_chars, required=required)


@dataclass(frozen=True)
class ToolEvidenceCapsule:
    """Prompt-safe summary of bulky tool output."""

    command: str
    exit_code: int
    short_result: str
    evidence_hash: str
    affected_files: tuple[str, ...] = ()

    @classmethod
    def from_tool_output(
        cls,
        *,
        command: str,
        exit_code: int,
        output: str,
        affected_files: Iterable[str] = (),
    ) -> "ToolEvidenceCapsule":
        redacted = redact_secrets(output or "")
        digest = hashlib.sha256((command + "\0" + str(exit_code) + "\0" + redacted).encode("utf-8", errors="replace")).hexdigest()
        return cls(
            command=_clean(command, 180),
            exit_code=int(exit_code),
            short_result=_summarize_tool_output(redacted),
            evidence_hash="sha256:" + digest[:16],
            affected_files=_clean_tuple(tuple(affected_files), max_items=8, max_chars=160),
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "command": self.command,
            "exit_code": self.exit_code,
            "short_result": self.short_result,
            "evidence_hash": self.evidence_hash,
            "affected_files": list(self.affected_files),
        }

    def render(self, *, max_chars: int = 700) -> str:
        if max_chars <= 0:
            return ""
        lines = [
            "【tool_evidence_capsule】",
            f"  command: {self.command}",
            f"  exit_code: {self.exit_code}",
            f"  result: {self.short_result}",
            f"  evidence_hash: {self.evidence_hash}",
        ]
        if self.affected_files:
            lines.append("  affected_files: " + " | ".join(self.affected_files))
        return _truncate_preserving_markers("\n".join(lines), max_chars, required=(self.evidence_hash, self.short_result))


class RecallIntent(str, Enum):
    CONTEXT_DIAGNOSTICS = "context_diagnostics"
    MEMORY_RETRIEVAL_DIAGNOSTICS = "memory_retrieval_diagnostics"
    GIT_WORKFLOW = "git_workflow"
    CODING = "coding"
    RUNTIME_MAINTENANCE = "runtime_maintenance"
    RELATIONSHIP = "relationship"
    DEFAULT = "default"


@dataclass(frozen=True)
class RecallIntentDecision:
    intent: RecallIntent
    allowed_buckets: frozenset[str]
    allow_user_preferences: bool = True
    reason: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {
            "intent": self.intent.value,
            "allowed_buckets": sorted(self.allowed_buckets),
            "allow_user_preferences": self.allow_user_preferences,
            "reason": self.reason,
        }


@dataclass(frozen=True)
class RecallContinuityEvidence:
    """Session-scoped intent evidence from an active turn boundary."""

    prior_intent: RecallIntent
    confidence: float
    source: str
    session_id: str

    def __post_init__(self) -> None:
        confidence = float(self.confidence)
        if not 0.0 <= confidence <= 1.0:
            raise ValueError("continuity confidence must be between 0 and 1")
        object.__setattr__(self, "confidence", confidence)

    def is_usable(self, *, session_id: str | None = None) -> bool:
        return (
            self.prior_intent is not RecallIntent.DEFAULT
            and self.confidence >= 0.8
            and self.source in {"session_turn", "active_task", "continuity_capsule"}
            and bool(self.session_id)
            and (not session_id or self.session_id == session_id)
        )


@dataclass(frozen=True)
class GovernedPromptContext:
    rendered: str
    telemetry: dict[str, Any]


def has_memory_retrieval_signal(text: str | None) -> bool:
    normalized = (text or "").lower()
    return _contains_any(normalized, ("长期记忆", "记忆召回", "召回", "检索", "recall", "retrieval"))


def has_recall_quality_signal(text: str | None) -> bool:
    normalized = (text or "").lower()
    return _contains_any(
        normalized,
        ("精准", "精确", "正确", "优化", "诊断", "准确", "相关性", "相关", "精度", "precision", "relevance", "accuracy"),
    )


def has_elliptical_followup_signal(text: str | None) -> bool:
    normalized = (text or "").lower()
    return (
        _contains_any(normalized, ("这个", "那个", "现在", "也就是说", "this", "that", "now"))
        and _contains_any(normalized, ("优化", "改进", "预期", "符合", "达到", "效果", "expected", "working"))
        and _contains_any(normalized, ("吗", "么", "?", "？"))
    )


def classify_recall_intent(
    query: str | None,
    *,
    continuity_evidence: RecallContinuityEvidence | None = None,
    session_id: str | None = None,
) -> RecallIntentDecision:
    text = (query or "").lower()
    if has_memory_retrieval_signal(text) and has_recall_quality_signal(text):
        return RecallIntentDecision(
            intent=RecallIntent.MEMORY_RETRIEVAL_DIAGNOSTICS,
            allowed_buckets=frozenset({"memory_retrieval", "runtime_boundary", "current_task", "tool_evidence"}),
            allow_user_preferences=True,
            reason="query asks to diagnose or improve long-term memory retrieval precision",
        )

    if _contains_any(text, ("上下文", "context", "预算", "budget", "剪裁", "compaction", "链路", "pcltm")):
        return RecallIntentDecision(
            intent=RecallIntent.CONTEXT_DIAGNOSTICS,
            allowed_buckets=frozenset({"runtime_boundary", "current_task", "continuity_capsule", "tool_evidence", "context_budget"}),
            allow_user_preferences=False,
            reason="query asks about live context/PCLTM budget chain",
        )

    if _contains_any(text, ("git", "commit", "push", "rebase", "fetch", "远端", "分支", "提交")):
        return RecallIntentDecision(
            intent=RecallIntent.GIT_WORKFLOW,
            allowed_buckets=frozenset({"workflow", "runtime_boundary", "project_path", "current_task"}),
            allow_user_preferences=True,
            reason="query asks about git workflow",
        )

    if _contains_any(text, ("代码", "测试", "pytest", "实现", "修复", "文件", "repo", "仓库")):
        return RecallIntentDecision(
            intent=RecallIntent.CODING,
            allowed_buckets=frozenset({"workflow", "project_path", "runtime_boundary", "current_task", "tool_evidence"}),
            allow_user_preferences=True,
            reason="query asks about coding work",
        )
    if _contains_any(text, ("soullink", "hermes", "runtime", "provider", "治理", "doctor", "index")):
        return RecallIntentDecision(
            intent=RecallIntent.RUNTIME_MAINTENANCE,
            allowed_buckets=frozenset({"runtime_boundary", "project_path", "current_task", "tool_evidence"}),
            allow_user_preferences=True,
            reason="query asks about runtime maintenance",
        )
    if _contains_any(text, ("关系", "抱", "亲", "爱你", "喜欢你", "老婆", "想你", "累")):
        return RecallIntentDecision(
            intent=RecallIntent.RELATIONSHIP,
            allowed_buckets=frozenset({"relationship", "emotion_boundary", "user_preference"}),
            allow_user_preferences=True,
            reason="query asks for relationship/daily support",
        )
    if has_elliptical_followup_signal(text):
        if continuity_evidence and continuity_evidence.is_usable(session_id=session_id):
            if continuity_evidence.prior_intent is RecallIntent.MEMORY_RETRIEVAL_DIAGNOSTICS:
                return RecallIntentDecision(
                    intent=RecallIntent.MEMORY_RETRIEVAL_DIAGNOSTICS,
                    allowed_buckets=frozenset({"memory_retrieval", "runtime_boundary", "current_task", "tool_evidence"}),
                    allow_user_preferences=True,
                    reason="inherited memory retrieval diagnostics from session continuity",
                )
        return RecallIntentDecision(
            intent=RecallIntent.DEFAULT,
            allowed_buckets=frozenset(),
            allow_user_preferences=False,
            reason="ambiguous follow-up lacks session continuity evidence",
        )

    return RecallIntentDecision(
        intent=RecallIntent.DEFAULT,
        allowed_buckets=frozenset({"user_preference", "runtime_boundary", "current_task", "generic"}),
        allow_user_preferences=True,
        reason="default recall intent",
    )


def govern_prompt_context(
    raw_context: str,
    *,
    policy: ContextBudgetPolicy | None = None,
    continuation_capsule: ContinuationCapsule | None = None,
    tool_evidence: Sequence[ToolEvidenceCapsule] = (),
    recall_intent: RecallIntentDecision | None = None,
    outer_tag: str = "pcltm_live_context",
) -> GovernedPromptContext:
    """Render a hard-capped prompt context while preserving continuity state."""

    policy = policy or ContextBudgetPolicy()
    safe_outer_tag = _safe_outer_tag(outer_tag)
    actions: list[str] = []
    parts: list[str] = [f"<{safe_outer_tag}>"]
    continuation_rendered = ""
    if continuation_capsule and not continuation_capsule.is_empty:
        continuation_rendered = continuation_capsule.render(max_chars=policy.continuation_chars)
        if continuation_rendered:
            parts.append(continuation_rendered)
    evidence_budget = policy.evidence_chars
    evidence_rendered: list[str] = []
    for capsule in tool_evidence:
        remaining = max(evidence_budget - sum(len(item) for item in evidence_rendered), 0)
        if remaining <= 0:
            actions.append("omitted_tool_evidence")
            break
        rendered = capsule.render(max_chars=remaining)
        if rendered:
            evidence_rendered.append(rendered)
    parts.extend(evidence_rendered)
    if recall_intent:
        parts.append("【recall_intent】" + json.dumps(recall_intent.to_dict(), ensure_ascii=False, sort_keys=True))

    footer = f"\n</{safe_outer_tag}>"
    memory_header = "【governed_memory_view】"
    base = "\n".join(parts)
    memory_overhead = len("\n") + len(memory_header) + len("\n")
    remaining_for_memory = max(
        min(policy.memory_chars, policy.total_chars - len(base) - len(footer) - memory_overhead),
        0,
    )
    memory = _clean_raw_context(raw_context, strip_outer_tag=safe_outer_tag)
    omitted_chars = 0
    if len(memory) > remaining_for_memory:
        omitted_chars = len(memory) - remaining_for_memory
        memory = _truncate_with_omission(memory, remaining_for_memory, omitted_chars) if remaining_for_memory > 0 else ""
        actions.append("omitted")
    if memory:
        parts.append(memory_header)
        parts.append(memory)
    rendered = "\n".join(parts) + footer
    if len(rendered) > policy.total_chars:
        actions.append("hard_cap")
        rendered = _hard_cap(rendered, policy.total_chars)
    telemetry = {
        "within_budget": len(rendered) <= policy.total_chars,
        "total_chars": len(rendered),
        "limit_chars": policy.total_chars,
        "omitted_chars": omitted_chars,
        "actions": actions,
        "capsules": {
            "continuation": 1 if continuation_rendered else 0,
            "tool_evidence": len(evidence_rendered),
        },
    }
    return GovernedPromptContext(rendered=rendered, telemetry=telemetry)


def _safe_outer_tag(value: str) -> str:
    tag = (value or "pcltm_live_context").strip()
    if not re.fullmatch(r"[A-Za-z][A-Za-z0-9_:-]{0,80}", tag):
        return "pcltm_live_context"
    return tag


def _clean(value: object, max_chars: int) -> str:
    text = "\n".join(str(value or "").splitlines()).strip()
    text = redact_secrets(text)
    if max_chars >= 0 and len(text) > max_chars:
        return text[: max(0, max_chars - 1)].rstrip() + "…"
    return text


def _clean_tuple(values: Iterable[object], *, max_items: int, max_chars: int) -> tuple[str, ...]:
    cleaned: list[str] = []
    for value in values or ():
        item = _clean(value, max_chars)
        if item and item not in cleaned:
            cleaned.append(item)
        if len(cleaned) >= max_items:
            break
    return tuple(cleaned)


def _contains_any(text: str, markers: Iterable[str]) -> bool:
    return any(marker.lower() in text for marker in markers)


def _summarize_tool_output(output: str) -> str:
    lines = [line.strip() for line in (output or "").splitlines() if line.strip()]
    interesting = []
    for line in lines:
        lower = line.lower()
        if any(term in lower for term in ("passed", "failed", "error", "traceback", "exit", "ok", "warning", "redacted")):
            interesting.append(line)
    if not interesting and lines:
        interesting = [lines[-1]]
    return _clean(" | ".join(interesting[-4:]), 360)


def _clean_raw_context(raw_context: str, *, strip_outer_tag: str | None = None) -> str:
    text = redact_secrets(raw_context or "").strip()
    tag = _safe_outer_tag(strip_outer_tag or "") if strip_outer_tag else ""
    if tag:
        open_tag = f"<{tag}>"
        close_tag = f"</{tag}>"
        if text.startswith(open_tag) and text.endswith(close_tag):
            text = text[len(open_tag) : -len(close_tag)].strip()
    return text


def _truncate_with_omission(text: str, limit: int, omitted_chars: int) -> str:
    if limit <= 0:
        return f"[omitted {omitted_chars} chars due to live context budget]"
    marker = f"\n[omitted {omitted_chars} chars due to live context budget]"
    if len(marker) >= limit:
        return marker[-limit:]
    return text[: limit - len(marker)].rstrip() + marker


def _truncate_preserving_markers(text: str, max_chars: int, required: Sequence[str] = ()) -> str:
    if len(text) <= max_chars:
        return text
    cleaned_required = [item for item in required if item]
    prefix = text[: max(0, max_chars - 1)].rstrip() + "…"
    missing = [item for item in cleaned_required if item not in prefix]
    if not missing:
        return prefix
    suffix = "\n" + "\n".join(_clean(item, 120) for item in missing)
    if len(suffix) >= max_chars:
        return suffix[-max_chars:]
    return prefix[: max_chars - len(suffix)].rstrip() + suffix


def _hard_cap(text: str, limit: int) -> str:
    marker = "\n[hard_cap_applied]"
    if len(text) <= limit:
        return text
    if len(marker) >= limit:
        return marker[-limit:]
    return text[: limit - len(marker)].rstrip() + marker
