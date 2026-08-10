"""Write-time event classification for PCLTM.

PCLTM must not infer persona state from content. The persona state machine is the
single source of truth for memory classification. Domain words such as A-share,
ETF, gateway, QQBot, relationship, or cron remain ordinary text/query terms.
"""

from __future__ import annotations

import re
from dataclasses import dataclass

_SECRET_RE = re.compile(r"(?i)\b[A-Z0-9_]*(?:SECRET|TOKEN|API[_-]?KEY|PASSWORD|PASS)[A-Z0-9_]*\s*=")
_STATE_MODES = {
    "work",
    "system_maintenance",
    "daily",
    "intimacy",
    "conflict",
    "repair",
    "sex_candidate",
    "sex",
}

_MEMORY_COMMAND_RE = re.compile(
    r"^\[(?P<action>memory|replace|forget)(?::(?P<key>[a-z0-9][a-z0-9._-]{0,127}))?\]"
    r"(?P<value>.*)$",
    re.IGNORECASE | re.DOTALL,
)

_QUESTION_OR_UNCERTAIN_RE = re.compile(
    r"[？?]|(?:也许|可能|大概|或许|不确定|是不是|要不要|你觉得|我觉得呢)"
)
_TRANSIENT_OR_PROCESS_RE = re.compile(
    r"(?:今天|今晚|现在|当前|刚才|这次|本次|临时|通过了|失败了|"
    r"ASYNC DELEGATION|active task list|tool result|context compression)",
    re.IGNORECASE,
)
_STABLE_PREFERENCE_RE = re.compile(
    r"^(?:我|本人)(?P<stability>长期|一直|一贯|始终|以后|今后|通常|默认|向来|平时|一向)"
    r"(?:都|会|更)?(?:偏好|喜欢|更喜欢|不喜欢|讨厌|希望|习惯|倾向)(?P<value>.+)$"
)
_DIRECT_PREFERENCE_RE = re.compile(
    r"^(?:我|本人)(?:更)?(?:偏好|喜欢|不喜欢|讨厌|希望|习惯|倾向)(?P<value>.+)$"
)
_IDENTITY_FACT_RE = re.compile(
    r"^(?:我的)?(?P<field>名字|姓名|称呼|生日|出生日期|职业|家乡|常住地|居住地)"
    r"(?:是|为|叫)(?P<value>.+)$"
)
_LONG_TERM_CONVENTION_RE = re.compile(
    r"^(?:以后|今后|从今以后|长期|始终|默认)(?P<value>.+)"
    r"(?:不要|不得|必须|一律|统一|都要|都不|默认)(?P<tail>.*)$"
)

# Natural assertions are admitted by a bounded semantic feature layer rather
# than an ever-growing list of complete sentence templates.  The patterns
# below describe concepts and paraphrases; they never bypass candidate review.
_WRONG_SUBJECT_OR_QUOTE_RE = re.compile(
    r"^(?:我(?:的)?朋友|朋友|他|她|他们|她们|用户|对方)(?:说|觉得|喜欢|偏好|希望|讨厌|是|做|从事)|"
    r"^(?:用户|朋友|他|她)说[：:]"
)
_NEGATED_SELF_ASSERTION_RE = re.compile(
    r"(?:我(?:并)?不是说|我没说|不代表我|并非我的|不是我的偏好)"
)
_DISCOURSE_PREFIX_RE = re.compile(
    r"^(?:(?:你知道的|说真的|老实说|坦白说|其实|说实话|讲真的)[，,、：:\s]*)+"
)
_VOCATIVE_BEFORE_SELF_RE = re.compile(
    r"^(?:助手|助理|秘书|伙伴|Assistant)[，,、：:\s]*(?=我)"
)

_NATURAL_SEMANTIC_RULES: tuple[tuple[str, str, str, tuple[str, ...], float, float, float], ...] = (
    (
        "user_preference", "user", "preference:engineering:repository-cleanliness",
        (
            r"^我(?:喜欢|偏爱|更喜欢|偏好|更看重|重视|追求).*(?:代码仓库|仓库|repo(?:sitory)?).*(?:干净|整洁|清爽|无残留)",
            r"^我(?:喜欢|偏爱|更喜欢|偏好|更看重|重视|追求).*(?:干净|整洁|清爽|无残留).*(?:代码仓库|仓库|repo(?:sitory)?)",
            r"^(?:代码仓库|仓库|repo(?:sitory)?).*(?:必须|应当|应该|要|保持).*(?:干净|整洁|清爽|无残留)",
        ),
        0.20, 0.95, 0.96,
    ),
    (
        "user_preference", "user", "preference:response-style:directness",
        (r"回答.*(?:直接|开门见山).*(?:别|少|不要).*(?:铺垫|寒暄)",
         r"(?:更喜欢|更希望|倾向).*(?:直接(?:给|说)?(?:结论|答案)|开门见山).*(?:少|不要|别).*(?:铺垫|寒暄)"),
        0.20, 0.94, 0.90,
    ),
    (
        "user_preference", "user", "preference:response-style:evidence-over-reassurance",
        (r"比起.*(?:安慰|共情).*(?:更在意|更看重|更希望).*(?:查清|证据|事实|解决)",),
        0.15, 0.96, 0.94,
    ),
    (
        "user_preference", "user", "preference:response-style:conciseness",
        (r"(?:长篇|太长|冗长).*(?:一般|通常|总是)?.*(?:看不下去|不爱看|不喜欢|受不了)",
         r"我(?:受不了|不喜欢|讨厌).*(?:回答|回复).*(?:客套|废话|寒暄|铺垫)"),
        0.20, 0.91, 0.88,
    ),
    (
        "user_preference", "user", "preference:engineering:code-and-architecture-quality",
        (
            r"^我(?:喜欢|偏爱|更喜欢|更看重|重视|追求).*(?:代码|代码质量).*(?:架构|结构)",
            r"^我(?:喜欢|偏爱|更喜欢|更看重|重视|追求).*(?:架构|结构).*(?:代码|代码质量)",
        ),
        0.20, 0.96, 0.96,
    ),
    (
        "identity_fact", "user", "identity:职业",
        (r"^我(?:做|从事|干)(?P<occupation>.+?)(?:很多年|多年|有些年)(?:了)?$",
         r"^我是在.+?(?:做|从事|干)(?P<occupation>.+?)(?:的)?$"),
        0.25, 0.92, 0.90,
    ),
    (
        "system_convention", "memory", "convention:git:commit-confirmation",
        (r"(?:提交代码|git\s*commit|commit).*(?:先|之前).*(?:跟我说|问我|确认|告诉我)",),
        0.20, 0.95, 0.96,
    ),
    (
        "system_convention", "memory", "convention:change:test-before-report",
        (r"(?:修改|改动).*(?:以后|今后|都要|一律).*(?:测试|验证).*(?:再|之后).*(?:告诉我|报告|汇报)",),
        0.25, 0.94, 0.97,
    ),
    (
        "system_convention", "memory", "convention:destructive-action:confirm-first",
        (r"(?:以后|今后|往后|从今以后).*(?:删|删除|覆盖|清空).*(?:先|之前).*(?:问我|确认|告诉我)",),
        0.20, 0.95, 0.98,
    ),
)


@dataclass(frozen=True)
class StableMemoryAssertion:
    kind: str
    target_file: str
    content: str
    semantic_key: str
    stability_signal: str = "inferred"
    admission_tier: str = "pending_review"
    reason_codes: tuple[str, ...] = ()
    lexical_score: float = 1.0
    semantic_score: float = 0.55
    stability_score: float = 0.90
    future_value_score: float = 0.85

    @property
    def admission_confidence(self) -> float:
        """Weighted worthiness score; policy still forces natural claims to review."""
        score = (
            0.20 * self.lexical_score
            + 0.45 * self.semantic_score
            + 0.20 * self.stability_score
            + 0.15 * self.future_value_score
        )
        return max(0.0, min(float(score), 0.84))


def _stable_key(value: str) -> str:
    import hashlib

    normalized = re.sub(r"\s+", "", value).casefold()
    normalized = re.sub(r"[，,、:：;；。.!！]", "", normalized)
    return hashlib.sha256(normalized.encode("utf-8")).hexdigest()[:24]


def _assertion_core(value: str) -> str:
    """Remove bounded conversational framing without rewriting the assertion."""
    core = str(value or "").strip()
    previous = None
    while core and core != previous:
        previous = core
        core = _DISCOURSE_PREFIX_RE.sub("", core, count=1).strip()
        core = _VOCATIVE_BEFORE_SELF_RE.sub("", core, count=1).strip()
    return core


def parse_stable_memory_assertion(content: str) -> StableMemoryAssertion | None:
    """Recognize only explicit, stable natural-language memory assertions."""
    normalized = str(content or "").strip()
    if not normalized or len(normalized) > 500:
        return None
    normalized = normalized.rstrip("。.!！").strip()
    if (
        not normalized
        or _QUESTION_OR_UNCERTAIN_RE.search(normalized)
        or _TRANSIENT_OR_PROCESS_RE.search(normalized)
        or normalized.startswith("[")
        or _WRONG_SUBJECT_OR_QUOTE_RE.search(normalized)
        or _NEGATED_SELF_ASSERTION_RE.search(normalized)
    ):
        return None
    core = _assertion_core(normalized)
    if not core or _WRONG_SUBJECT_OR_QUOTE_RE.search(core):
        return None
    for (
        kind, target, semantic_key, patterns,
        lexical_score, semantic_score, future_value_score,
    ) in _NATURAL_SEMANTIC_RULES:
        if any(re.search(pattern, core, re.IGNORECASE) for pattern in patterns):
            return StableMemoryAssertion(
                kind, target, normalized, semantic_key,
                stability_signal="bounded_semantic_rule",
                admission_tier="auto_activate",
                reason_codes=("bounded_semantic_rule", "stable", "future_value"),
                lexical_score=lexical_score,
                semantic_score=semantic_score,
                stability_score=0.88,
                future_value_score=future_value_score,
            )
    match = _STABLE_PREFERENCE_RE.fullmatch(core)
    if match is not None and match.group("value").strip():
        return StableMemoryAssertion(
            "user_preference", "user", normalized,
            "natural-preference:" + _stable_key(match.group("value")),
            stability_signal="explicit_long_term",
            admission_tier="auto_activate",
            reason_codes=("first_person", "explicit_preference", "explicit_stability_marker"),
        )
    match = _DIRECT_PREFERENCE_RE.fullmatch(core)
    if match is not None and match.group("value").strip():
        return StableMemoryAssertion(
            "user_preference", "user", normalized,
            "natural-preference:" + _stable_key(match.group("value")),
            stability_signal="unmarked",
            admission_tier="pending_review",
            reason_codes=("first_person", "explicit_preference"),
        )
    match = _IDENTITY_FACT_RE.fullmatch(core)
    if match is not None and match.group("value").strip():
        field = match.group("field")
        return StableMemoryAssertion(
            "identity_fact", "user", normalized, f"identity:{field}",
            stability_signal="intrinsic",
            admission_tier="auto_activate",
            reason_codes=("first_person", "explicit_identity_fact"),
        )
    match = _LONG_TERM_CONVENTION_RE.fullmatch(core)
    if match is not None:
        value = (match.group("value") + match.group("tail")).strip()
        if value:
            return StableMemoryAssertion(
                "system_convention", "memory", normalized,
                "natural-convention:" + _stable_key(value),
                stability_signal="explicit_long_term",
                admission_tier="auto_activate",
                reason_codes=("explicit_convention", "explicit_stability_marker"),
            )
    return None


def parse_memory_command(content: str) -> tuple[str, str | None, str] | None:
    """Parse the bounded durable-memory protocol.

    ``memory`` may omit a key for backwards-compatible content addressing.
    ``replace`` and ``forget`` require a stable semantic key.  Forget carries no
    value; memory/replace require one.  Invalid forms are ordinary conversation.
    """
    match = _MEMORY_COMMAND_RE.fullmatch(str(content or "").strip())
    if match is None:
        return None
    action = match.group("action").casefold()
    key = match.group("key")
    value = match.group("value").strip()
    if action in {"replace", "forget"} and not key:
        return None
    if action == "forget":
        return (action, key, "") if not value else None
    return (action, key, value) if value else None


def is_durable_memory_assertion(content: str) -> bool:
    """Return whether content opts into automatic durable-memory admission.

    This deliberately tiny contract is lexical and model-free: only a message
    whose first non-whitespace token is ``[memory]`` is eligible.  Mode
    confidence never participates in this decision.
    """
    return parse_memory_command(content) is not None


@dataclass(frozen=True)
class Classification:
    category: str
    subcategory: str
    inject_policy: str
    sensitivity: str
    confidence: float
    classifier_version: str = "state-machine-only-v4"


class EventClassifier:
    """Classifier constrained to the current persona state-machine mode only."""

    version = "state-machine-only-v4"

    def classify(
        self,
        *,
        role: str,
        source: str,
        content: str,
        persona_mode: str | None = None,
        sensitivity: str = "normal",
    ) -> Classification:
        mode = self._normalize_mode(persona_mode)

        if source == "cron":
            return Classification("unknown", "unknown", "no_memory", sensitivity, 1.0, self.version)
        if _SECRET_RE.search(content):
            return Classification(mode, mode, "retrieve_only", "secret" if sensitivity == "normal" else sensitivity, 1.0, self.version)
        if mode == "unknown":
            return Classification("unknown", "unknown", "retrieve_only", sensitivity, 0.2, self.version)
        if role == "user" and source in {"chat", "hermes_state_db"} and (
            is_durable_memory_assertion(content)
            or parse_stable_memory_assertion(content) is not None
        ):
            return Classification(mode, mode, "candidate_only", sensitivity, 0.95, self.version)
        return Classification(mode, mode, "retrieve_only", sensitivity, 0.9, self.version)

    @staticmethod
    def _normalize_mode(persona_mode: str | None) -> str:
        mode = (persona_mode or "").lower()
        if mode in _STATE_MODES:
            return mode
        return "unknown"
