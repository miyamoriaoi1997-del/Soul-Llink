"""Extract reusable procedural memory candidates from complex task traces.

The extractor is deliberately conservative. It should find repeatable
workflows, not archive task progress. One-off status such as "phase 5 done" or
"submitted PR #12" belongs in episodic/session logs, never in procedural
memory or Hermes skills.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from datetime import UTC, datetime
from typing import Iterable

from .procedural_memory import ProceduralMemory


_STEP_PREFIXES = (
    "step:",
    "steps:",
    "procedure:",
    "process:",
    "workflow:",
    "do:",
    "run:",
    "patch:",
    "implement:",
    "verify:",
    "步骤:",
    "流程:",
    "执行:",
    "验证:",
)
_VERIFY_PREFIXES = (
    "verify:",
    "verification:",
    "test:",
    "tests:",
    "check:",
    "验收:",
    "验证:",
    "测试:",
)
_PITFALL_PREFIXES = (
    "pitfall:",
    "avoid:",
    "risk:",
    "warning:",
    "do not:",
    "don't:",
    "风险:",
    "避免:",
    "不要:",
    "坑:",
)
_TRIGGER_PREFIXES = (
    "trigger:",
    "use when:",
    "when:",
    "applies when:",
    "触发:",
    "适用:",
)
_ONE_OFF_PATTERNS = (
    re.compile(r"\bphase\s*\d+\s+(done|complete|completed|finished)\b", re.I),
    re.compile(r"\b(pr|issue)\s*#\d+\b", re.I),
    re.compile(r"\bcommit\s+[0-9a-f]{7,40}\b", re.I),
    re.compile(r"\bsha\s+[0-9a-f]{7,40}\b", re.I),
    re.compile(r"\b(ticket|task)\s*[-#]?\d+\b", re.I),
    re.compile(r"阶段\s*\d+\s*(完成|完工|提交)"),
    re.compile(r"(今天|昨天|刚才|本轮|当前任务|阶段提交)"),
)
_REUSABLE_HINTS = (
    "reusable",
    "repeatable",
    "workflow",
    "procedure",
    "pattern",
    "skill",
    "verify",
    "pitfall",
    "use when",
    "trigger",
    "可复用",
    "复用",
    "流程",
    "步骤",
    "技能",
    "验证",
    "风险",
    "沉淀",
)


@dataclass(frozen=True)
class TaskTrace:
    """Minimal task evidence used to propose a reusable skill."""

    title: str
    messages: tuple[str, ...] = ()
    tool_calls: tuple[str, ...] = ()
    outcomes: tuple[str, ...] = ()
    source_session: str | None = None
    category: str = "general"

    def text(self) -> str:
        return "\n".join((self.title, *self.messages, *self.tool_calls, *self.outcomes))


@dataclass(frozen=True)
class SkillCandidate:
    """Result of extracting a possible procedural memory record."""

    memory: ProceduralMemory | None
    accepted: bool
    reasons: tuple[str, ...]


class SkillCandidateExtractor:
    """Conservative extractor from complex task traces to procedural memory."""

    def __init__(self, *, min_tool_calls: int = 5, min_messages: int = 3) -> None:
        self.min_tool_calls = min_tool_calls
        self.min_messages = min_messages

    def extract(self, trace: TaskTrace) -> SkillCandidate:
        """Extract a reusable procedure if the trace has durable skill signal."""

        text = trace.text()
        reasons: list[str] = []

        if self._looks_like_one_off_status(text):
            return SkillCandidate(None, False, ("trace looks like one-off task status",))

        if not self._is_complex(trace):
            return SkillCandidate(None, False, ("task is not complex enough to justify a skill",))
        reasons.append("task crossed complexity threshold")

        reusable_hint_count = sum(1 for hint in _REUSABLE_HINTS if hint in text.lower())
        if reusable_hint_count == 0:
            return SkillCandidate(None, False, ("no reusable workflow signal found",))
        reasons.append("reusable workflow signal found")

        trigger_conditions = self._extract_trigger_conditions(trace)
        procedure = self._extract_procedure(trace)
        verification_steps = self._extract_verification_steps(trace)
        pitfalls = self._extract_pitfalls(trace)

        if len(procedure) < 2:
            return SkillCandidate(None, False, ("not enough procedure steps",))
        if not verification_steps:
            return SkillCandidate(None, False, ("no verification step found",))

        confidence = self._confidence(
            trace,
            reusable_hint_count=reusable_hint_count,
            procedure=procedure,
            verification_steps=verification_steps,
            pitfalls=pitfalls,
        )
        memory = ProceduralMemory(
            skill_name=self._skill_name(trace.title),
            trigger_conditions=tuple(trigger_conditions),
            procedure=tuple(procedure),
            verification_steps=tuple(verification_steps),
            pitfalls=tuple(pitfalls),
            source_sessions=(trace.source_session,) if trace.source_session else (),
            last_updated=datetime.now(UTC),
            confidence=confidence,
            category=trace.category,
        )
        if not memory.is_exportable:
            return SkillCandidate(memory, False, ("candidate remained below export policy",))
        return SkillCandidate(memory, True, tuple(reasons))

    def _is_complex(self, trace: TaskTrace) -> bool:
        evidence_points = 0
        if len(trace.tool_calls) >= self.min_tool_calls:
            evidence_points += 2
        elif len(trace.tool_calls) >= 3:
            evidence_points += 1
        if len(trace.messages) >= self.min_messages:
            evidence_points += 1
        if len(trace.outcomes) >= 1:
            evidence_points += 1
        if len(set(trace.tool_calls)) >= 3:
            evidence_points += 1
        return evidence_points >= 3

    def _looks_like_one_off_status(self, text: str) -> bool:
        lowered = text.lower()
        if any(pattern.search(text) for pattern in _ONE_OFF_PATTERNS):
            return True
        status_words = sum(
            1
            for marker in ("done", "completed", "finished", "submitted", "已完成", "完工", "提交")
            if marker in lowered
        )
        reusable_words = sum(1 for marker in _REUSABLE_HINTS if marker in lowered)
        return status_words >= 2 and reusable_words == 0

    def _extract_trigger_conditions(self, trace: TaskTrace) -> list[str]:
        prefixed = _prefixed_lines(trace.text(), _TRIGGER_PREFIXES)
        if prefixed:
            return _dedupe(prefixed)
        title = trace.title.strip().rstrip(".")
        return [f"Use when handling tasks like: {title}"]

    def _extract_procedure(self, trace: TaskTrace) -> list[str]:
        prefixed = _prefixed_lines(trace.text(), _STEP_PREFIXES)
        candidates: list[str] = []
        candidates.extend(prefixed)
        if not candidates:
            for call in trace.tool_calls:
                normalized = call.strip()
                if not normalized:
                    continue
                candidates.append(f"Use {normalized} when it provides necessary evidence or changes")
        if len(candidates) < 2:
            for line in trace.messages:
                cleaned = _strip_bullet(line)
                if _looks_procedural_line(cleaned):
                    candidates.append(cleaned)
        return _dedupe(candidates)

    def _extract_verification_steps(self, trace: TaskTrace) -> list[str]:
        prefixed = _prefixed_lines(trace.text(), _VERIFY_PREFIXES)
        candidates = list(prefixed)
        for outcome in trace.outcomes:
            cleaned = _strip_bullet(outcome)
            if any(word in cleaned.lower() for word in ("pass", "verify", "test", "check", "验收", "验证", "测试")):
                candidates.append(cleaned)
        return _dedupe(candidates)

    def _extract_pitfalls(self, trace: TaskTrace) -> list[str]:
        return _dedupe(_prefixed_lines(trace.text(), _PITFALL_PREFIXES))

    def _skill_name(self, title: str) -> str:
        words = re.findall(r"[a-zA-Z0-9\u4e00-\u9fff]+", title.lower())
        if not words:
            return "procedural-workflow"
        return "-".join(words[:8])

    def _confidence(
        self,
        trace: TaskTrace,
        *,
        reusable_hint_count: int,
        procedure: Iterable[str],
        verification_steps: Iterable[str],
        pitfalls: Iterable[str],
    ) -> float:
        score = 0.35
        score += min(0.2, len(trace.tool_calls) * 0.03)
        score += min(0.15, reusable_hint_count * 0.03)
        score += min(0.15, len(tuple(procedure)) * 0.03)
        score += min(0.1, len(tuple(verification_steps)) * 0.05)
        score += min(0.05, len(tuple(pitfalls)) * 0.025)
        if trace.source_session:
            score += 0.05
        return min(1.0, score)


def _prefixed_lines(text: str, prefixes: tuple[str, ...]) -> list[str]:
    values: list[str] = []
    for raw_line in text.splitlines():
        line = _strip_bullet(raw_line)
        lowered = line.lower()
        for prefix in prefixes:
            if lowered.startswith(prefix):
                values.append(line[len(prefix) :].strip())
                break
    return [value for value in values if value]


def _strip_bullet(line: str) -> str:
    return re.sub(r"^\s*(?:[-*•]|\d+[.)])\s*", "", line).strip()


def _looks_procedural_line(line: str) -> bool:
    lowered = line.lower()
    return any(word in lowered for word in ("run ", "check ", "verify", "patch", "write", "test", "use ", "执行", "验证", "测试", "修复"))


def _dedupe(values: Iterable[str]) -> list[str]:
    seen: set[str] = set()
    result: list[str] = []
    for value in values:
        normalized = " ".join(str(value).strip().split())
        if not normalized:
            continue
        key = normalized.lower()
        if key in seen:
            continue
        seen.add(key)
        result.append(normalized)
    return result
