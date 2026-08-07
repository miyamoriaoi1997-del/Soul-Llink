"""PCLTM persona-memory adapter for production prompt and memory-tool writes.

This adapter is the bridge between Hermes' public memory interface and the
PCLTM database. It renders prompt-time memory as <pcltm_context>, exposes
approved PCLTM entries to MemoryStore's live tool view, and applies memory-tool
mutations to PCLTM memory_records. STATE.md remains owned by the emotion
runtime.
"""

from __future__ import annotations

import os
import sqlite3
import hashlib
import json
import logging
import re
from datetime import UTC, datetime
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Any, Iterable

import yaml

from pcltm.live_context_governor import (
    ContextBudgetPolicy,
    RecallContinuityEvidence,
    classify_recall_intent,
    govern_prompt_context,
    has_memory_retrieval_signal,
    has_recall_quality_signal,
)
from pcltm.secret_policy import evaluate_memory_write, redact_secrets

if TYPE_CHECKING:
    from pcltm.memfs_types import PromptMemoryView

ENTRY_DELIMITER = "\n§\n"
from .runtime_paths import DEFAULT_DB, DEFAULT_MEMFS_ROOT, SOUL_LINK_ROOT, resolve_db_path


def _safe_configured_memfs_root(raw_root: str | os.PathLike[str] | None) -> Path:
    """Return a production-safe MemFS root.

    Environment variables are process-global and easy to poison in service
    managers or tests.  Production imports therefore only honor
    HERMES_PCLTM_MEMFS_ROOT when it stays inside Soul-Link's managed var tree;
    tests can still monkeypatch the module-level MEMFS_ROOT directly.
    """

    if raw_root is None or str(raw_root).strip() == "":
        return DEFAULT_MEMFS_ROOT
    candidate = Path(raw_root).expanduser()
    resolved = candidate.resolve(strict=False)
    allowed_root = DEFAULT_MEMFS_ROOT.resolve(strict=False)
    try:
        resolved.relative_to(allowed_root)
    except ValueError:
        return DEFAULT_MEMFS_ROOT
    return resolved


MEMFS_ROOT = _safe_configured_memfs_root(os.getenv("HERMES_PCLTM_MEMFS_ROOT"))
RELATIONSHIP_MODES = {"intimacy", "daily", "conflict", "repair", "sex_candidate", "sex"}
WORK_MODES = {"system_maintenance", "work"}
DEFAULT_TOP_K = {"USER.md": 10, "MEMORY.md": 8}

# Module-level state for post-response citation tracking
_last_injected_ids: list[int] = []
_last_live_context_telemetry: dict[str, Any] = {}
_last_memory_selection_observation: dict[str, Any] = {}
MAX_ENTRY_CHARS = {"USER.md": 260, "MEMORY.md": 320}
LIVE_CONTEXT_MIN_TOTAL_CHARS = 900
LIVE_CONTEXT_MAX_TOTAL_CHARS = 3600


@dataclass(frozen=True)
class BucketRule:
    bucket: str
    metadata_any: tuple[str, ...] = ()
    text_any: tuple[str, ...] = ()
    target_file: str | None = None


@dataclass(frozen=True)
class ViewPolicy:
    """Data-driven prompt-context policy used by the production adapter."""

    bucket_rules: tuple[BucketRule, ...] = field(
        default_factory=lambda: (
            BucketRule("project_path", ("project_path", "path", "repo", "workdir"), ("/root/", "/a-share-quant", "路径：")),
            BucketRule("rollback", ("rollback", "backup", "checkpoint"), ("rollback", "backup", "checkpoint")),
            BucketRule("current_task", ("current_task", "task_context", "active_task")),
            BucketRule(
                "continuity_capsule",
                (
                    "current_task",
                    "task_context",
                    "active_task",
                    "continuity",
                    "continuity_capsule",
                    "previous_conversation",
                    "resume_requested",
                    "session_continuity",
                ),
                (
                    "continuity capsule",
                    "previous_conversation_state",
                    "session-continuity",
                    "active task",
                    "active_task",
                    "继续任务",
                ),
            ),
            BucketRule("investment", ("investment", "a-share", "finance"), ("A股", "ETF", "基金", "持仓", "电力")),
            BucketRule("relationship", ("relationship", "intimacy", "emotional_commitment")),
            BucketRule("user_preference", ("user_preference", "preference"), target_file="USER.md"),
        )
    )
    intimacy_markers: tuple[str, ...] = ("亲密", "老婆", "老公", "黑丝", "丝袜", "害怕", "忘记", "关系", "likes_assistant")
    runtime_markers: tuple[str, ...] = (
        "gateway",
        "qqbot",
        "runtime",
        "router",
        "cron",
        "hermes",
        "state.md",
        "pcltm",
        "active prompt",
        "system_prompt",
        "prompt",
        "重启",
        "验收",
        "自检",
        "维护",
        "生产",
        "路径",
    )
    investment_markers: tuple[str, ...] = ("A股", "ETF", "基金", "/a-share-quant", "持仓", "电力")
    query_terms: tuple[str, ...] = (
        "PCLTM",
        "STATE",
        "USER",
        "MEMORY",
        "A股",
        "ETF",
        "基金",
        "持仓",
        "电力",
                "老婆",
        "情绪",
        "记忆",
        "gateway",
        "Hermes",
        "router",
        "QQBot",
        "continuity capsule",
        "previous_conversation_state",
        "session-continuity",
        "active task",
        "active_task",
        "继续任务",
    )
    # Optional policy data, supplied by the deployment/evaluation boundary.
    # Keep the adapter mechanism generic rather than embedding domain phrases.
    query_alias_groups: tuple[tuple[str, ...], ...] = ()
    boundary_metadata_terms: tuple[str, ...] = ("secret_boundary", "runtime_boundary", "boundary", "production_risk")
    emotion_boundary_terms: tuple[str, ...] = ("emotion", "desire", "sex", "consent", "aftercare", "情绪", "SOUL", "overwhelming", "欲望", "边界", "亲密")
    protected_tags: frozenset[str] = field(default_factory=lambda: frozenset({"critical", "production", "boundary", "identity"}))
    user_preference_text_markers: tuple[str, ...] = ("用户希望", "用户希望", "用户偏好", "用户偏好")
    memory_boundary_text_markers: tuple[str, ...] = ("禁止", "必须", "PCLTM", "STATE", "生产", "路径", "rollback")
    runtime_tags: frozenset[str] = field(default_factory=lambda: frozenset({"production", "runtime", "hermes"}))
    relationship_tags: frozenset[str] = field(default_factory=lambda: frozenset({"relationship", "intimacy"}))
    noise_markers: tuple[str, ...] = ("噪声", "临时", "temporary", "tmp", "smoke")
    boost_gov_mode_relevance: float = 0.1
    boost_protected_tag: float = 1.5
    boost_user_preference_text: float = 0.4
    boost_memory_boundary_text: float = 0.5
    boost_mode_relevance: float = 0.7
    penalty_noise: float = -0.4
    boost_query_hit: float = 0.7
    max_query_boost: float = 3.0
    lexical_query_boost_cap: float = 4.0
    semantic_query_boost_cap: float = 3.5
    exact_phrase_boost: float = 2.0
    bucket_query_boost: float = 1.0
    query_recall_candidate_floor: float = 1.6
    quota_profiles: dict[str, dict[str, dict[str, int]]] = field(
        default_factory=lambda: {
            "work": {
                "USER.md": {"user_preference": 4, "emotion_boundary": 3, "relationship": 1, "investment": 1, "generic": 1},
                "MEMORY.md": {"runtime_boundary": 3, "project_path": 2, "rollback": 2, "current_task": 2, "investment": 1, "generic": 1},
            },
            "relationship": {
                "USER.md": {"relationship": 4, "user_preference": 3, "emotion_boundary": 3, "runtime_boundary": 1, "investment": 0, "generic": 1},
                "MEMORY.md": {"relationship": 4, "emotion_boundary": 2, "runtime_boundary": 1, "investment": 0, "generic": 1},
            },
            "sex": {
                "USER.md": {"relationship": 3, "emotion_boundary": 3, "user_preference": 2, "runtime_boundary": 0, "investment": 0, "generic": 0},
                "MEMORY.md": {"relationship": 2, "emotion_boundary": 2, "runtime_boundary": 0, "investment": 0, "generic": 0},
            },
            "cron": {
                "USER.md": {"user_preference": 1, "runtime_boundary": 1, "generic": 1},
                "MEMORY.md": {"runtime_boundary": 2, "project_path": 1, "generic": 1},
            },
            "default": {
                "USER.md": {"user_preference": 4, "emotion_boundary": 2, "relationship": 1, "generic": 1},
                "MEMORY.md": {"runtime_boundary": 3, "project_path": 1, "rollback": 1, "generic": 2},
            },
        }
    )

    @staticmethod
    def mode_profile(mode: str | None) -> str:
        mode_key = (mode or "").lower()
        if mode_key in WORK_MODES:
            return "work"
        if mode_key == "sex":
            return "sex"
        if mode_key in RELATIONSHIP_MODES:
            return "relationship"
        if mode_key == "cron":
            return "cron"
        return "default"

    def quotas_for(self, target_file: str, mode: str | None) -> dict[str, int]:
        profile = self.mode_profile(mode)
        target_profiles = self.quota_profiles.get(profile) or self.quota_profiles.get("default", {})
        quotas = target_profiles.get(target_file) or target_profiles.get("MEMORY.md") or {}
        return dict(quotas)


DEFAULT_VIEW_POLICY = ViewPolicy()


def _live_context_total_chars(memory_limit: int, user_limit: int) -> int:
    requested = int(memory_limit or 0) + int(user_limit or 0) + 500
    return max(LIVE_CONTEXT_MIN_TOTAL_CHARS, min(LIVE_CONTEXT_MAX_TOTAL_CHARS, requested))


def _live_context_policy(memory_limit: int, user_limit: int) -> ContextBudgetPolicy:
    total = _live_context_total_chars(memory_limit, user_limit)
    return ContextBudgetPolicy(
        total_chars=total,
        continuation_chars=min(700, max(0, total // 4)),
        evidence_chars=0,
        memory_chars=max(160, total - 360),
    )


def last_live_context_telemetry() -> dict[str, Any]:
    """Return telemetry from the most recent governed prompt-context render."""
    return dict(_last_live_context_telemetry)


def last_memory_selection_observation() -> dict[str, Any]:
    """Return a detached read-only observation of the latest real selection pass."""
    return json.loads(json.dumps(_last_memory_selection_observation)) if _last_memory_selection_observation else {}


def _compact_entries(entries: list[str], limit: int) -> tuple[str, int]:
    """Select whole entries within limit; never truncate an individual memory."""
    if limit <= 0:
        return "", len(entries)
    selected: list[str] = []
    omitted = 0
    for entry in entries:
        candidate = entry if not selected else ENTRY_DELIMITER.join([*selected, entry])
        if len(candidate) <= limit:
            selected.append(entry)
        else:
            omitted += 1
    content = ENTRY_DELIMITER.join(selected)
    if omitted:
        marker = f"[省略 {omitted} 条超出预算的记录]"
        candidate = marker if not content else ENTRY_DELIMITER.join([content, marker])
        if len(candidate) <= limit:
            content = candidate
        else:
            compact_marker = f"[省略{omitted}条]"
            candidate = compact_marker if not content else ENTRY_DELIMITER.join([content, compact_marker])
            if len(candidate) <= limit:
                content = candidate
    return content, omitted


def enabled() -> bool:
    return os.getenv("HERMES_PCLTM_DISABLE", "").lower() not in {"1", "true", "yes", "on"} and os.getenv(
        "HERMES_PCLTM_PERSONA_VIEWS", "1"
    ).lower() in {"1", "true", "yes", "on"}


def db_path() -> Path:
    return resolve_db_path()


def _contains_any(text: str, markers: tuple[str, ...]) -> bool:
    lower = text.lower()
    return any(marker.lower() in lower for marker in markers)


def _looks_intimacy(text: str, policy: ViewPolicy = DEFAULT_VIEW_POLICY) -> bool:
    return _contains_any(text, policy.intimacy_markers)


def _looks_runtime(text: str, policy: ViewPolicy = DEFAULT_VIEW_POLICY) -> bool:
    return _contains_any(text, policy.runtime_markers)


def _looks_investment(text: str, policy: ViewPolicy = DEFAULT_VIEW_POLICY) -> bool:
    return _contains_any(text, policy.investment_markers)


_CJK_STOP_PHRASES = {
    "可以",
    "继续",
    "继续做",
    "你看看",
    "看看",
    "现在",
    "是不是",
    "有没有",
    "能不能",
    "怎么样",
    "为什么",
    "怎么",
    "什么",
    "这个",
    "那个",
    "一下",
    "帮我",
}


def _query_terms(query: str | None, policy: ViewPolicy = DEFAULT_VIEW_POLICY) -> set[str]:
    if not query:
        return set()
    text = query.lower()
    terms: set[str] = set()
    for token in re.findall(r"[a-zA-Z0-9_./+-]{2,}", text):
        terms.add(token)
    for token in policy.query_terms:
        if token.lower() in text:
            terms.add(token.lower())
    for group in policy.query_alias_groups:
        normalized_group = tuple(str(token).strip().lower() for token in group if str(token).strip())
        if any(token in text for token in normalized_group):
            terms.update(normalized_group)
    for size in (2, 3, 4):
        for i in range(0, max(0, len(query) - size + 1)):
            chunk = query[i : i + size]
            if any("\u4e00" <= ch <= "\u9fff" for ch in chunk):
                terms.add(chunk.lower())
    return terms


def _query_content_terms(query: str | None, policy: ViewPolicy = DEFAULT_VIEW_POLICY) -> set[str]:
    """Return query terms safe for lexical/BM25 scoring.

    ``_query_terms`` stays intentionally broad so bucket and metadata aliases keep
    working.  Ranking needs a stricter view: short functional Chinese words such
    as "继续" or "可以" are too common and can otherwise drag unrelated old
    memories above the real topic.
    """
    terms = _query_terms(query, policy)
    if not terms:
        return set()
    protected = {token.lower() for token in policy.query_terms}
    protected.update(
        str(token).strip().lower()
        for group in policy.query_alias_groups
        for token in group
        if str(token).strip()
    )
    filtered: set[str] = set()
    for term in terms:
        normalized = term.strip().lower()
        if not normalized:
            continue
        if normalized in protected:
            filtered.add(normalized)
            continue
        has_cjk = any("\u4e00" <= ch <= "\u9fff" for ch in normalized)
        if has_cjk:
            if normalized in _CJK_STOP_PHRASES:
                continue
            if len(normalized) < 3:
                continue
        filtered.add(normalized)
    return filtered


def _query_content(query: str | None, policy: ViewPolicy = DEFAULT_VIEW_POLICY) -> str | None:
    """Return a de-noised query string for lexical/BM25 content retrieval."""
    terms = sorted(_query_content_terms(query, policy), key=lambda term: (-len(term), term))
    if not terms:
        return None
    return " ".join(terms)


def _query_phrases(query: str | None, policy: ViewPolicy = DEFAULT_VIEW_POLICY) -> set[str]:
    """Return longer query phrases that should beat incidental n-gram hits."""
    if not query:
        return set()
    normalized = " ".join(str(query).split()).lower()
    phrases: set[str] = set()
    for token in re.findall(r"[a-zA-Z0-9_./+-]{4,}", normalized):
        phrases.add(token)
    for token in re.split(r"[，。；、,;:：!?？!\n\r]+", normalized):
        token = token.strip()
        if len(token) >= 4 and token not in _CJK_STOP_PHRASES:
            has_cjk = any("\u4e00" <= ch <= "\u9fff" for ch in token)
            if not has_cjk or any(key.lower() in token for key in policy.query_terms):
                phrases.add(token)
    cjk_runs = re.findall(r"[\u4e00-\u9fff]{3,}", normalized)
    for run in cjk_runs:
        if run not in _CJK_STOP_PHRASES:
            phrases.add(run)
        for size in (3, 4, 5, 6):
            if len(run) <= size:
                continue
            for i in range(0, len(run) - size + 1):
                phrase = run[i : i + size]
                if phrase not in _CJK_STOP_PHRASES:
                    phrases.add(phrase)
    return phrases


def _query_relevance(text: str, query: str | None, policy: ViewPolicy = DEFAULT_VIEW_POLICY) -> float:
    terms = _query_terms(query, policy)
    if not terms:
        return 0.0
    lower = text.lower()
    hits = sum(1 for term in terms if term and term in lower)
    return min(policy.max_query_boost, hits * policy.boost_query_hit)


def _query_match_features(text: str, query: str | None, policy: ViewPolicy = DEFAULT_VIEW_POLICY) -> tuple[float, bool, int]:
    """Return (lexical_score, exact_phrase_hit, hit_count) for prompt recall.

    Bucket metadata can use broad aliases, but lexical content ranking should
    avoid short functional CJK fragments.  Exact phrase evidence still wins when
    the user's full topical phrase is present.
    """
    terms = _query_content_terms(query, policy)
    if not terms:
        return (0.0, False, 0)
    lower = text.lower()
    hits = sum(1 for term in terms if term and term in lower)
    lexical = min(policy.lexical_query_boost_cap, hits * policy.boost_query_hit)
    exact = any(phrase and phrase in lower for phrase in _query_phrases(query))
    if exact:
        lexical = min(policy.lexical_query_boost_cap, lexical + policy.exact_phrase_boost)
    return (lexical, exact, hits)


def _metadata_recall_terms(metadata: dict) -> set[str]:
    """Return normalized metadata terms used by query recall and audits."""
    terms = _tags(metadata) | _mode_scope(metadata)
    for key in ("buckets", "bucket", "category", "type", "kind"):
        raw = metadata.get(key)
        if isinstance(raw, str):
            terms.add(raw.lower())
        elif isinstance(raw, Iterable):
            terms.update(str(value).lower() for value in raw if value is not None)
    terms.update(_fact_projection_terms(metadata))
    return terms


def _fact_projection_terms(metadata: dict) -> set[str]:
    """Project only explicit structured fact values into lexical recall."""
    terms: set[str] = set()
    facts = metadata.get("facts")
    if isinstance(facts, dict):
        stack: list[Any] = list(facts.values())
        while stack:
            value = stack.pop()
            if isinstance(value, dict):
                stack.extend(value.values())
            elif isinstance(value, Iterable) and not isinstance(value, (str, bytes)):
                stack.extend(value)
            elif value is not None:
                text = str(value).strip().lower()
                if text:
                    terms.add(text)
                    terms.update(re.findall(r"[a-zA-Z0-9_./+-]{2,}|[\u4e00-\u9fff]{2,}", text))
    return terms


def _metadata_bucket_for_record(metadata: dict) -> str | None:
    """Return the first explicit metadata bucket, if present."""
    for key in ("bucket", "buckets"):
        raw = metadata.get(key)
        if isinstance(raw, str) and raw.strip():
            return raw.strip().lower()
        if isinstance(raw, Iterable) and not isinstance(raw, (str, bytes)):
            for value in raw:
                if value is not None and str(value).strip():
                    return str(value).strip().lower()
    return None


def _bucket_query_relevance(metadata: dict, query: str | None, policy: ViewPolicy = DEFAULT_VIEW_POLICY) -> float:
    if not query:
        return 0.0
    query_terms = _query_terms(query, policy)
    if not query_terms:
        return 0.0
    metadata_terms = _metadata_recall_terms(metadata)
    if metadata_terms & query_terms:
        return policy.bucket_query_boost
    joined_query = " ".join(query_terms)
    if any(term and term in joined_query for term in metadata_terms):
        return policy.bucket_query_boost
    return 0.0

def _row_timestamp(row: sqlite3.Row) -> str:
    for key in ("reviewed_at", "created_at"):
        if key in row.keys() and row[key]:
            return str(row[key])
    return ""


def _age_bucket(row: sqlite3.Row, reference_date: str | None = None) -> str:
    """Coarse age bucket for diversity selection; lexical ISO dates sort safely."""
    ts = _row_timestamp(row)
    if not ts:
        return "unknown"
    day = ts[:10]
    if not reference_date:
        reference_date = datetime.now(UTC).date().isoformat()
    try:
        from datetime import date
        delta = (date.fromisoformat(reference_date[:10]) - date.fromisoformat(day)).days
    except Exception:
        return day[:7]
    if delta <= 30:
        return "recent_30d"
    if delta <= 180:
        return "recent_180d"
    return day[:7]


def _record_diversity_key(row: sqlite3.Row, policy: ViewPolicy = DEFAULT_VIEW_POLICY) -> tuple[str, str]:
    metadata = _metadata(row)
    tags = _tags(metadata)
    bucket = "generic"
    for rule in policy.bucket_rules:
        if rule.target_file:
            continue
        if tags & {tag.lower() for tag in rule.metadata_any} or _contains_any(row["content"], rule.text_any):
            bucket = rule.bucket
            break
    return (bucket, _age_bucket(row))


def _diversify_ranked_rows(rows: list[sqlite3.Row], query: str | None, policy: ViewPolicy = DEFAULT_VIEW_POLICY) -> list[sqlite3.Row]:
    if not query or len(rows) < 4:
        return rows
    selected: list[sqlite3.Row] = []
    seen_keys: set[tuple[str, str]] = set()
    deferred: list[sqlite3.Row] = []
    for row in rows:
        key = _record_diversity_key(row, policy)
        if key in seen_keys:
            deferred.append(row)
            continue
        selected.append(row)
        seen_keys.add(key)
    selected.extend(deferred)
    return selected

def _semantic_scores_for_query(
    query: str | None,
    target_file: str | None = None,
    metadata_terms: dict[int, Iterable[str]] | None = None,
    policy: ViewPolicy = DEFAULT_VIEW_POLICY,
) -> dict[int, float]:
    """Get BM25 semantic scores for all records matching a query.

    Returns {record_id: normalized_score}. The cap is intentionally controlled by
    ViewPolicy so recall experiments can get stronger without bypassing the
    active-prompt governor and mode/bucket safeguards.
    """
    if not query or not query.strip():
        return {}
    try:
        from pcltm.semantic_index import get_index
        idx = get_index(db_path())
        results = idx.query(
            query.strip(),
            top_k=40,
            target_file=target_file,
            min_score=0.25,
            metadata_terms=metadata_terms,
        )
        if not results:
            return {}
        max_score = results[0][1]
        if max_score <= 0:
            return {}
        cap = policy.semantic_query_boost_cap
        return {rid: round(min(cap, (score / max_score) * cap), 4) for rid, score in results}
    except Exception as exc:
        logging.getLogger(__name__).warning("Semantic index query failed: %s", exc)
        return {}



def _metadata(row: sqlite3.Row | dict) -> dict:
    raw = row["metadata"] if "metadata" in row.keys() else "{}"
    try:
        parsed = json.loads(raw or "{}")
        return parsed if isinstance(parsed, dict) else {}
    except Exception:
        return {}


def _tags(metadata: dict) -> set[str]:
    return {str(t).lower() for t in metadata.get("tags", []) if t is not None}


def _mode_scope(metadata: dict) -> set[str]:
    raw = metadata.get("mode_scope") or metadata.get("modes") or []
    if isinstance(raw, str):
        raw = [raw]
    return {str(m).lower() for m in raw if m is not None}


def _category(metadata: dict) -> str:
    return str(metadata.get("category") or metadata.get("type") or "").lower()


def _inject_policy(metadata: dict) -> str:
    return str(metadata.get("inject_policy") or metadata.get("prompt_policy") or "").lower()


def _active_prompt_allowed(row: sqlite3.Row) -> bool:
    policy = _inject_policy(_metadata(row))
    return policy not in {"never", "no_prompt", "retrieve_only", "tool_only", "internal", "shadow_only"}


def _is_relationship_record(row: sqlite3.Row, metadata: dict, policy: ViewPolicy = DEFAULT_VIEW_POLICY) -> bool:
    tags = _tags(metadata)
    kind = str(row["kind"] if "kind" in row.keys() else "").lower()
    category = _category(metadata)
    text = row["content"]
    return (
        kind in {"emotional_commitment"}
        or category in {"emotional_commitment"}
        or bool(tags & {"relationship", "intimacy", "emotional_commitment"})
        or _looks_intimacy(text, policy)
    )


def _mode_allowed(row: sqlite3.Row, target_file: str, mode: str | None, policy: ViewPolicy = DEFAULT_VIEW_POLICY) -> bool:
    if not mode:
        return True
    mode_key = mode.lower()
    metadata = _metadata(row)
    if not _active_prompt_allowed(row):
        return False
    scope = _mode_scope(metadata)
    if scope:
        if mode_key in scope:
            return True
        if mode_key in WORK_MODES and scope & WORK_MODES:
            return True
        if mode_key in RELATIONSHIP_MODES and scope & RELATIONSHIP_MODES:
            return True
        return False
    relationship = _is_relationship_record(row, metadata, policy)
    if mode_key in WORK_MODES and relationship:
        return False
    if mode_key in RELATIONSHIP_MODES and _looks_runtime(row["content"], policy):
        return False
    if mode_key == "cron" and target_file == "USER.md":
        return False
    return True


def _fit_entry(text: str, target_file: str) -> str | None:
    limit = MAX_ENTRY_CHARS.get(target_file, 320)
    if limit <= 0:
        return None
    text = text.strip()
    if not text:
        return None
    if len(text) <= limit:
        return text
    return text[: max(0, limit - 1)].rstrip() + "…"


def _continuity_query_hint(query: str | None) -> str | None:
    """Augment active-task queries so archived continuity capsules are recallable.

    Continuity snapshots are intentionally typed task memory, not a legacy
    compressed handoff.  Appending stable capsule terms lets PCLTM/MemFS recall
    those records when the active task is selected, even after the recent
    transcript window no longer contains the snapshot text.
    """
    terms = (
        "continuity capsule",
        "previous_conversation_state",
        "session-continuity",
        "active task",
        "active_task",
        "resume requested",
        "继续任务",
    )
    base = " ".join(str(query or "").split())
    lower = base.lower()
    if not _contains_any(lower, ("继续", "恢复", "接着", "续做", "resume", "continue", "recover", "previous task")):
        return base or None
    missing = [term for term in terms if term.lower() not in lower]
    if not base:
        return " ".join(terms)
    if not missing:
        return base
    return f"{base} {' '.join(missing)}"


def _is_memory_retrieval_diagnostics(recall_intent) -> bool:
    return recall_intent.intent.value == "memory_retrieval_diagnostics"


def _is_memory_retrieval_diagnostics_relevant(
    row: sqlite3.Row,
    target_file: str,
    recall_intent,
    policy: ViewPolicy = DEFAULT_VIEW_POLICY,
) -> bool:
    """Use bucket as authority and record-level signals as topical admission."""
    content = str(row["content"] or "")
    bucket = _bucket_for(row, target_file, policy)
    if bucket == "memory_retrieval":
        return True
    return (
        has_memory_retrieval_signal(content)
        and (bucket in recall_intent.allowed_buckets or has_recall_quality_signal(content))
    )


def _rank_rows(
    target_file: str,
    rows: Iterable[sqlite3.Row],
    mode: str | None,
    query: str | None = None,
    policy: ViewPolicy = DEFAULT_VIEW_POLICY,
    *,
    use_default_semantic_index: bool = False,
) -> list[sqlite3.Row]:
    mode_key = (mode or "").lower()
    effective_query = _continuity_query_hint(query) if target_file == "MEMORY.md" else query

    rows_list = list(rows)
    recall_intent = classify_recall_intent(query)
    if _is_memory_retrieval_diagnostics(recall_intent):
        rows_list = [
            row
            for row in rows_list
            if _is_memory_retrieval_diagnostics_relevant(row, target_file, recall_intent, policy)
        ]
    metadata_terms: dict[int, set[str]] = {}
    for row in rows_list:
        metadata = _metadata(row)
        record_id = int(row["record_id"] if "record_id" in row.keys() else 0)
        terms = _metadata_recall_terms(metadata)
        if terms:
            metadata_terms[record_id] = terms

    # Pre-compute semantic scores for all records (if query provided).  BM25 sees
    # the de-noised content query so short continuation phrases do not swamp the
    # actual topic.
    semantic_query = _query_content(effective_query, policy) or effective_query
    semantic_scores = (
        _semantic_scores_for_query(
            semantic_query,
            target_file=target_file,
            metadata_terms=metadata_terms,
            policy=policy,
        )
        if use_default_semantic_index
        else {}
    )

    def score(row: sqlite3.Row) -> tuple[float, int]:
        text = row["content"]
        metadata = _metadata(row)
        record_id = int(row["record_id"] if "record_id" in row.keys() else 0)

        projected_text = " ".join((text, *sorted(_fact_projection_terms(metadata))))
        lexical_boost, exact_query_hit, query_hit_count = _query_match_features(projected_text, effective_query, policy)
        bucket_boost = _bucket_query_relevance(metadata, effective_query, policy)
        # Semantic boost from BM25 index. Cap here so experiments can adjust the
        # index scale without letting semantic scores swamp safety/governor rank.
        sem_boost = min(policy.semantic_query_boost_cap, semantic_scores.get(record_id, 0.0))
        query_boost = lexical_boost + bucket_boost + sem_boost

        # --- Governor score (primary, if available) ---
        gov_score = metadata.get("gov_score")
        if gov_score is not None:
            try:
                gov_val = float(gov_score)
                # Mode-aware boost on top of governor score
                boost = 0.0
                if mode_key in WORK_MODES and _looks_runtime(text, policy):
                    boost += policy.boost_gov_mode_relevance
                if mode_key in RELATIONSHIP_MODES and _looks_intimacy(text, policy):
                    boost += policy.boost_gov_mode_relevance
                boost += query_boost
                return (gov_val + boost, record_id)
            except (ValueError, TypeError):
                pass

        # --- Fallback heuristic (legacy, for records not yet scored by governor) ---
        try:
            value = float(metadata.get("importance", 0.5))
        except Exception:
            value = 0.5
        tags = _tags(metadata)
        if tags & policy.protected_tags:
            value += policy.boost_protected_tag
        if target_file == "USER.md" and _contains_any(text, policy.user_preference_text_markers):
            value += policy.boost_user_preference_text
        if target_file == "MEMORY.md" and _contains_any(text, policy.memory_boundary_text_markers):
            value += policy.boost_memory_boundary_text
        if mode_key in WORK_MODES and (_looks_runtime(text, policy) or policy.runtime_tags & tags):
            value += policy.boost_mode_relevance
        if mode_key in RELATIONSHIP_MODES and (_looks_intimacy(text, policy) or policy.relationship_tags & tags):
            value += policy.boost_mode_relevance
        if _contains_any(text, policy.noise_markers):
            value += policy.penalty_noise
        value += query_boost
        if effective_query and (query_boost > 0 or exact_query_hit or query_hit_count > 0):
            value = max(value, policy.query_recall_candidate_floor + query_boost)
        return (value, record_id)

    return _diversify_ranked_rows(sorted(rows_list, key=score, reverse=True), effective_query, policy)


def _render(target_file: str, entries: list[str], *, limit: int) -> str:
    """Legacy USER/MEMORY block rendering is retired.

    Canonical prompt-time memory is rendered by governed injection from
    ``memory_current``. This helper stays empty so stale callers cannot resurrect
    USER PROFILE / MEMORY prompt blocks.
    """
    return ""


def _slug_for_memfs(text: str, fallback: str) -> str:
    """Return a stable, filesystem-safe slug for a memory file."""
    ascii_bits = re.findall(r"[a-zA-Z0-9]+", text.lower())
    slug = "-".join(ascii_bits[:8]).strip("-")
    if not slug:
        cjk_bits = re.findall(r"[\u4e00-\u9fff]{1,8}", text)
        slug = "-".join(cjk_bits[:4]).strip("-")
    return (slug or fallback)[:80]


def _memfs_layer_for_target(target_file: str) -> str:
    """Map legacy memory target files to active MemFS layers."""
    return "pinned" if target_file == "USER.md" else "episodic"


def _memfs_mode_scope_for_record(row: sqlite3.Row, target_file: str, policy: ViewPolicy = DEFAULT_VIEW_POLICY) -> tuple[str, ...]:
    """Infer mode scope for a materialized memory record."""
    metadata = _metadata(row)
    explicit_scope = _mode_scope(metadata)
    if explicit_scope:
        return tuple(sorted(explicit_scope))
    text = row["content"]
    if target_file == "USER.md" and _is_relationship_record(row, metadata, policy):
        return ("daily", "sex")
    if target_file == "MEMORY.md" and _looks_runtime(text, policy):
        return ("work", "cron")
    return ("daily", "work", "sex")


def _memfs_path_for_record(record_id: int, target_file: str, content: str) -> str:
    if target_file not in {"USER.md", "MEMORY.md"}:
        raise ValueError(f"unsupported MemFS target file: {target_file!r}")
    layer = _memfs_layer_for_target(target_file)
    target_slug = "user" if target_file == "USER.md" else "memory"
    return f"{layer}/{target_slug}-{record_id:06d}-{_slug_for_memfs(content, 'record')}.md"


def _memfs_root(root: str | os.PathLike[str] | None = None) -> Path:
    """Resolve an effective MemFS root without trusting ambient env blindly."""

    if root is None:
        return Path(MEMFS_ROOT)
    return Path(root).expanduser()


def _safe_memfs_record_path(root: str | os.PathLike[str] | None, rel_path: str) -> Path:
    """Resolve a materialized record path under the configured MemFS root."""

    from pcltm.memfs_store import MemFSStore

    return MemFSStore(_memfs_root(root))._safe_resolve(rel_path)


def _memory_type_for_record(row: sqlite3.Row, target_file: str, metadata: dict, bucket: str) -> str:
    explicit = metadata.get("memory_type") or metadata.get("type")
    valid = {
        "UserPreference",
        "ProjectPath",
        "RuntimeInvariant",
        "PersonaBoundary",
        "ToolQuirk",
        "RelationshipAnchor",
        "FinancialProfile",
        "WorkflowConvention",
        "RiskPolicy",
        "TemporaryTaskState",
    }
    if explicit in valid:
        return str(explicit)
    if bucket == "project_path":
        return "ProjectPath"
    if bucket in {"runtime_boundary"}:
        return "RuntimeInvariant"
    if bucket == "relationship":
        return "RelationshipAnchor"
    if bucket == "investment":
        return "FinancialProfile"
    if bucket == "current_task":
        return "TemporaryTaskState"
    if target_file == "USER.md":
        return "UserPreference"
    return "WorkflowConvention"


def _lifecycle_state_for_record(row: sqlite3.Row) -> str:
    status = str(row["status"] if "status" in row.keys() else "approved")
    return "active" if status == "approved" else status


def _ttl_for_memory_type(memory_type: str) -> str:
    return "short" if memory_type == "TemporaryTaskState" else "none"


def _evidence_refs_for_record(row: sqlite3.Row) -> list[dict[str, object]]:
    refs: list[dict[str, object]] = []
    if "record_id" in row.keys():
        refs.append({"type": "memory_record", "id": int(row["record_id"])})
    if "candidate_id" in row.keys() and row["candidate_id"]:
        refs.append({"type": "candidate", "id": row["candidate_id"]})
    return refs


def materialize_memfs_from_approved_records() -> int:
    """Retired legacy ``memory_records`` → MemFS materialization seam."""
    return 0


def _fetch_record_by_candidate_id(con: sqlite3.Connection, candidate_id: str) -> sqlite3.Row | None:
    con.row_factory = sqlite3.Row
    return con.execute("SELECT * FROM memory_records WHERE candidate_id = ?", (candidate_id,)).fetchone()


def _like_contains_literal(value: str) -> str:
    return "%" + value.replace("\\", "\\\\").replace("%", "\\%").replace("_", "\\_") + "%"


def _fetch_superseded_records(con: sqlite3.Connection, target_file: str, old_text: str) -> list[sqlite3.Row]:
    con.row_factory = sqlite3.Row
    return con.execute(
        """
        SELECT * FROM memory_records
        WHERE target_file = ? AND status = 'superseded' AND content LIKE ? ESCAPE '\\'
        ORDER BY record_id ASC
        """,
        (target_file, _like_contains_literal(old_text)),
    ).fetchall()


def _fetch_approved_records(con: sqlite3.Connection, target_file: str, old_text: str) -> list[sqlite3.Row]:
    """Capture approved rows before mutation for filesystem compensation."""
    con.row_factory = sqlite3.Row
    return con.execute(
        """
        SELECT * FROM memory_records
        WHERE target_file = ? AND status = 'approved' AND content LIKE ? ESCAPE '\\'
        ORDER BY record_id ASC
        """,
        (target_file, _like_contains_literal(old_text)),
    ).fetchall()


def _memfs_record_file_path(row: sqlite3.Row | dict) -> Path:
    target_file = str(row["target_file"])
    record_id = int(row["record_id"])
    content = str(row["content"] or "")
    return _safe_memfs_record_path(None, _memfs_path_for_record(record_id, target_file, content))


def _remove_memfs_record_file(row: sqlite3.Row | dict) -> None:
    path = _memfs_record_file_path(row)
    try:
        path.unlink()
    except FileNotFoundError:
        pass


def _bucket_for(row: sqlite3.Row, target_file: str, policy: ViewPolicy = DEFAULT_VIEW_POLICY) -> str:
    """Classify a PCLTM record for quota-based prompt injection.

    Metadata is authoritative. Text fallback is only for legacy records that were
    imported before category/subcategory/inject_policy existed.
    """
    metadata = _metadata(row)
    tags = _tags(metadata)
    kind = str(row["kind"] if "kind" in row.keys() else "").lower()
    category = _category(metadata)
    subcategory = str(metadata.get("subcategory") or metadata.get("subtype") or "").lower()
    inject_policy = str(metadata.get("inject_policy") or "").lower()
    text = row["content"]
    lower = text.lower()
    joined = " ".join([kind, category, subcategory, inject_policy, " ".join(sorted(tags))])

    explicit_bucket = _metadata_bucket_for_record(metadata)
    if explicit_bucket:
        return explicit_bucket

    # Structured legacy signals outrank broad words in free text.  In particular,
    # terms such as "boundary", "emotion", or "SOUL" can describe runtime
    # architecture without making the record an emotional/persona boundary.
    for rule in policy.bucket_rules:
        if rule.target_file and rule.target_file != target_file:
            continue
        metadata_hit = any(token.lower() in joined for token in rule.metadata_any)
        text_hit = any(token.lower() in lower for token in rule.text_any)
        if metadata_hit or text_hit:
            return rule.bucket

    # USER.md is structurally a preference/profile store.  Legacy free text can
    # mention runtime tools (for example "prompt") without changing that type;
    # runtime USER records must carry an explicit bucket during migration/write.
    if target_file == "USER.md":
        if _is_relationship_record(row, metadata, policy) or any(
            token.lower() in joined for token in policy.emotion_boundary_terms
        ):
            return "emotion_boundary"
        return "user_preference"

    if any(token in joined for token in policy.boundary_metadata_terms):
        return "runtime_boundary"
    if _looks_runtime(text, policy) or any(token in joined for token in ("runtime", "hermes", "production", "system_convention")):
        return "runtime_boundary"

    if any(token.lower() in joined or token.lower() in lower for token in policy.emotion_boundary_terms):
        return "emotion_boundary"
    return "generic"


def _bucket_quotas(target_file: str, mode: str | None, policy: ViewPolicy = DEFAULT_VIEW_POLICY) -> dict[str, int]:
    """Return per-type quotas for direct PCLTM prompt injection."""
    return policy.quotas_for(target_file, mode)


def _select_entry_rows(
    target_file: str,
    rows: Iterable[sqlite3.Row],
    mode: str | None,
    policy: ViewPolicy = DEFAULT_VIEW_POLICY,
    *,
    decisions: dict[int, str] | None = None,
) -> list[tuple[str, sqlite3.Row]]:
    """Select prompt entries while preserving their source DB rows.

    Older code returned only rendered strings plus ids.  That was enough for
    prompt injection, but it made layered-view audits lossy: once entries were
    compacted into a string, item ids/scores/buckets became synthetic.  Keeping
    the source row attached lets context snapshots explain exactly *which* PCLTM
    records won recall and why, without changing prompt text semantics.
    """
    materialized = list(rows)
    top_k = DEFAULT_TOP_K.get(target_file, 8)
    if decisions is not None:
        decisions.update({int(row["record_id"]): "top_k_excluded" for row in materialized})
    if top_k <= 0:
        return []
    quotas = _bucket_quotas(target_file, mode, policy)
    selected: list[tuple[str, sqlite3.Row]] = []
    selected_texts: set[str] = set()
    bucket_counts: dict[str, int] = {}
    deferred: list[sqlite3.Row] = []

    def add_row(row: sqlite3.Row) -> bool:
        record_id = int(row["record_id"])
        if not _mode_allowed(row, target_file, mode, policy):
            if decisions is not None and decisions.get(record_id) != "selected":
                decisions[record_id] = "mode_excluded"
            return False
        bucket = _bucket_for(row, target_file, policy)
        if bucket_counts.get(bucket, 0) >= quotas.get(bucket, 0):
            if decisions is not None and decisions.get(record_id) != "selected":
                decisions[record_id] = "quota_excluded"
            return False
        entry = _fit_entry(row["content"], target_file)
        if entry is None:
            if decisions is not None and decisions.get(record_id) != "selected":
                decisions[record_id] = "invalid_entry"
            return False
        if entry in selected_texts:
            if decisions is not None and decisions.get(record_id) != "selected":
                decisions[record_id] = "duplicate_excluded"
            return False
        selected.append((entry, row))
        selected_texts.add(entry)
        bucket_counts[bucket] = bucket_counts.get(bucket, 0) + 1
        if decisions is not None:
            decisions[record_id] = "selected"
        return True

    for row in materialized:
        if len(selected) >= top_k:
            break
        if not _mode_allowed(row, target_file, mode, policy):
            continue
        add_row(row)
        deferred.append(row)

    # Use remaining capacity for quota-eligible buckets that were ranked behind
    # higher-scoring duplicates.  This preserves per-type minimum coverage (for
    # example current_task) while still keeping strict bucket quotas.
    if len(selected) < top_k:
        for row in deferred:
            if len(selected) >= top_k:
                break
            add_row(row)

    return selected


def _select_entries(target_file: str, rows: Iterable[sqlite3.Row], mode: str | None, policy: ViewPolicy = DEFAULT_VIEW_POLICY) -> tuple[list[str], list[int]]:
    """Select entries for prompt injection.

    Returns (entries, record_ids) — record_ids tracks which records were
    actually injected so retrieval stats can be updated.
    """
    selected = _select_entry_rows(target_file, rows, mode, policy)
    return [entry for entry, _row in selected], [int(row["record_id"]) for _entry, row in selected]


def _target_file(target: str) -> str | None:
    return {"user": "USER.md", "memory": "MEMORY.md"}.get(target)


def _fetch_rows(target_file: str) -> list[sqlite3.Row]:
    path = db_path()
    if not path.exists():
        return []
    con = sqlite3.connect(path)
    con.row_factory = sqlite3.Row
    try:
        columns = {row[1] for row in con.execute("PRAGMA table_info(memory_records)").fetchall()}
        record_id_expr = "record_id" if "record_id" in columns else "rowid AS record_id"
        metadata_expr = "metadata" if "metadata" in columns else "'{}' AS metadata"
        kind_expr = "kind" if "kind" in columns else "'memory_note' AS kind"
        created_at_expr = "created_at" if "created_at" in columns else "NULL AS created_at"
        reviewed_at_expr = "reviewed_at" if "reviewed_at" in columns else "NULL AS reviewed_at"
        return con.execute(
            f"""
            SELECT {record_id_expr}, {kind_expr}, content, {metadata_expr}, {created_at_expr}, {reviewed_at_expr}
            FROM memory_records
            WHERE status = 'approved' AND target_file = ?
            ORDER BY record_id ASC
            """,
            (target_file,),
        ).fetchall()
    finally:
        con.close()


def _rows_allowed_by_recall_intent(
    target_file: str,
    rows: Iterable[sqlite3.Row],
    recall_intent,
    query: str | None,
    policy: ViewPolicy = DEFAULT_VIEW_POLICY,
) -> list[sqlite3.Row]:
    """Apply recall-intent authority before ranking or prompt selection."""
    materialized = list(rows)
    if recall_intent is None:
        return materialized
    if target_file == "USER.md":
        if not recall_intent.allow_user_preferences:
            return []
        if not _is_memory_retrieval_diagnostics(recall_intent):
            return materialized
    allowed = set(recall_intent.allowed_buckets)
    if not allowed:
        return []
    if _is_memory_retrieval_diagnostics(recall_intent):
        return [
            row
            for row in materialized
            if _is_memory_retrieval_diagnostics_relevant(row, target_file, recall_intent, policy)
        ]
    return [
        row
        for row in materialized
        if _bucket_for(row, target_file, policy) in allowed
        or _query_relevance(str(row["content"] or ""), query, policy) > 0
    ]


def _policy_for_mode(mode: str | None) -> str:
    mode_key = (mode or "default").lower()
    if mode_key in WORK_MODES:
        return "runtime_boundary / project_path / rollback / current_task / user_preferences"
    if mode_key in RELATIONSHIP_MODES:
        return "relationship_memory / user_preference / emotional_commitment"
    if mode_key == "cron":
        return "scheduled_task / durable_system_facts"
    return "mode_scope / importance / query_relevance"


def _sanitize_direct_entry(text: str) -> str:
    safe_text = redact_secrets(text)
    return (
        safe_text.replace("<pcltm_context>", "＜pcltm_context＞")
        .replace("</pcltm_context>", "＜/pcltm_context＞")
        .replace("USER PROFILE (who the user is)", "legacy USER profile header")
        .replace("MEMORY (your personal notes)", "legacy MEMORY header")
    )


def _render_prompt_context(entries_by_target: dict[str, list[str]], *, mode: str | None, query: str | None) -> str:
    lines = [
        "<pcltm_context>",
        f"【mode】{mode or 'default'}",
        f"【state_machine_mode】{mode or 'default'}",
        f"【pcltm_mode】{mode or 'default'}",
        f"【mode_sync】{'consistent' if mode in {'daily', 'work', 'sex'} else 'fallback_hint'}",
        f"【retrieval_policy】{_policy_for_mode(mode)}",
    ]
    if query:
        lines.append(f"【query_hint】{_sanitize_direct_entry(query.strip()[:160])}")
    total = 0
    core_entries = entries_by_target.get("SYSTEM.md", [])
    if core_entries:
        lines.append("【core_blocks】")
        for entry in core_entries:
            lines.append(f"- [system] {_sanitize_direct_entry(entry)}")
            total += 1
    lines.append("【selected_records】")
    for target_file in ("USER.md", "MEMORY.md"):
        entries = entries_by_target.get(target_file, [])
        label = "user" if target_file == "USER.md" else "memory"
        for entry in entries:
            lines.append(f"- [{label}] {_sanitize_direct_entry(entry)}")
            total += 1
    if total == 0:
        return ""
    lines.append("</pcltm_context>")
    return "\n".join(lines)


def _load_system_core_entries(*, mode: str | None, query: str | None, budget_chars: int = 1200) -> list[str]:
    """Load MemFS system/core blocks that must precede selected DB memories.

    The system layer is the highest memory authority below the SOUL/runtime
    Retained only for offline layered-view diagnostics; canonical governed
    injection does not call this helper.
    """
    root = _memfs_root()
    if not root.is_dir():
        return []
    try:
        from pcltm.memfs_store import MemFSStore
        store = MemFSStore(root)
        layer = store.load_layer(
            "system",
            mode=_memfs_mode_scope(mode, DEFAULT_VIEW_POLICY),
            query=query,
            budget_chars=budget_chars,
            buckets=None,
        )
    except Exception:
        return []
    entries = [item.body.strip() for item in layer.items if item.body and item.body.strip()]
    content, _ = _compact_entries(entries, budget_chars)
    return [entry.strip() for entry in content.split(ENTRY_DELIMITER) if entry.strip()]


def _update_retrieval_stats(record_ids: list[int]) -> None:
    """Increment retrieval_count and update last_retrieved_at for injected records.

    Called after records are selected for prompt injection. This enables the
    governor's Weibull decay + retrieval boost scoring: frequently used memories
    stay fresh, unused ones decay naturally.
    """
    if not record_ids:
        return
    path = db_path()
    if not path.exists():
        return
    try:
        con = sqlite3.connect(path)
        now_iso = sqlite3.connect(":memory:").execute(
            "SELECT strftime('%Y-%m-%dT%H:%M:%fZ', 'now')"
        ).fetchone()[0]
        placeholders = ",".join("?" * len(record_ids))
        con.execute(
            f"""UPDATE memory_records
                SET retrieval_count = COALESCE(retrieval_count, 0) + 1,
                    last_retrieved_at = ?
                WHERE record_id IN ({placeholders}) AND status = 'approved'""",
            [now_iso] + record_ids,
        )
        con.commit()
        con.close()
    except Exception:
        pass  # Non-critical — don't break prompt assembly on stats failure


def track_citations(response_text: str, injected_ids: list[int]) -> list[int]:
    """Detect which injected memories were actually cited in the model response.

    Uses keyword overlap: extracts key phrases from each injected record and checks
    if they appear in the response. Records with ≥2 key phrase hits are considered cited.

    Returns list of cited record_ids and updates citation_count/last_cited_at in DB.
    """
    if not response_text or not injected_ids:
        return []
    path = db_path()
    if not path.exists():
        return []

    try:
        con = sqlite3.connect(path)
        con.row_factory = sqlite3.Row
        placeholders = ",".join("?" * len(injected_ids))
        rows = con.execute(
            f"SELECT record_id, content FROM memory_records WHERE record_id IN ({placeholders})",
            injected_ids,
        ).fetchall()

        response_lower = response_text.lower()
        cited_ids: list[int] = []

        for row in rows:
            content = row["content"]
            # Extract key phrases: split by common delimiters, take phrases ≥4 chars
            import re
            phrases = [p.strip().lower() for p in re.split(r'[，。；、/\n:：]', content) if len(p.strip()) >= 4]
            # Also extract English identifiers (paths, names, terms)
            eng_terms = [t.lower() for t in re.findall(r'[a-zA-Z_\-./]{4,}', content)]
            all_keys = phrases[:8] + eng_terms[:5]  # Cap to avoid over-matching

            hits = sum(1 for k in all_keys if k and k in response_lower)
            if hits >= 2:
                cited_ids.append(row["record_id"])

        if cited_ids:
            now_iso = con.execute("SELECT strftime('%Y-%m-%dT%H:%M:%fZ', 'now')").fetchone()[0]
            placeholders2 = ",".join("?" * len(cited_ids))
            con.execute(
                f"""UPDATE memory_records
                    SET citation_count = COALESCE(citation_count, 0) + 1,
                        last_cited_at = ?
                    WHERE record_id IN ({placeholders2}) AND status = 'approved'""",
                [now_iso] + cited_ids,
            )
            con.commit()
        con.close()
        return cited_ids
    except Exception:
        return []


def get_last_injected_ids() -> list[int]:
    """Return record IDs from the retired compatibility selector, if invoked.

    Canonical governed injection does not use this legacy ID surface.
    """
    return list(_last_injected_ids)




def _memfs_mode_scope(mode: str | None, policy: ViewPolicy = DEFAULT_VIEW_POLICY) -> str | None:
    """Map runtime modes to MemFS mode_scope names."""
    mode_key = (mode or "").lower()
    if mode_key in {"daily", "work", "sex", "cron", "default"}:
        return mode_key
    profile = policy.mode_profile(mode)
    if profile == "relationship":
        return "daily"
    if profile in {"work", "sex", "cron", "default"}:
        return profile
    return None


def load_layered_prompt_context(
    mode: str | None = None,
    query: str | None = None,
    *,
    budgets: dict[str, int] | None = None,
    layers: list[str] | tuple[str, ...] | None = None,
    buckets: list[str] | tuple[str, ...] | set[str] | None = None,
    active_layers: list[str] | tuple[str, ...] | None = None,
    root: Path | None = None,
) -> "PromptMemoryView":
    """Return a fail-closed empty view for the retired legacy MemFS prompt seam."""
    from pcltm.memfs_types import MemoryLayerView, PromptMemoryView

    del mode, query, layers, buckets, root
    effective_budgets = {
        "system": 1000,
        "pinned": 1500,
        "episodic": 1000,
        "transient": 500,
        **(budgets or {}),
    }
    selected_layers = set(active_layers or ())
    view = PromptMemoryView()
    view.selected_layers = tuple(
        layer for layer in ("system", "pinned", "episodic", "transient")
        if layer in selected_layers
    )
    view.selected_buckets = ()
    view.selection_source = "retired_legacy_memfs_prompt"
    view.system = MemoryLayerView(layer="system", budget_chars=effective_budgets["system"])
    view.pinned = MemoryLayerView(layer="pinned", budget_chars=effective_budgets["pinned"])
    view.episodic = MemoryLayerView(layer="episodic", budget_chars=effective_budgets["episodic"])
    view.transient = MemoryLayerView(layer="transient", budget_chars=effective_budgets["transient"])
    view.compression = MemoryLayerView(layer="compression", is_reference_only=True)
    return view


def select_context_snapshot(
    mode: str | None = None,
    query: str | None = None,
    *,
    budgets: dict[str, int] | None = None,
    layers: list[str] | tuple[str, ...] | None = None,
    buckets: list[str] | tuple[str, ...] | set[str] | None = None,
    active_layers: list[str] | tuple[str, ...] | None = None,
    root: Path | None = None,
):
    """Return a host-neutral PCLTM context-selection snapshot.

    This is the public control-plane companion to ``load_layered_prompt_context``.
    Host adapters can call it to inspect active layers, reference-only layers,
    budgets, omitted counts, and selected memory metadata without depending on
    Hermes prompt builders, session tables, or compression internals.
    """
    view = load_layered_prompt_context(
        mode=mode,
        query=query,
        budgets=budgets,
        layers=layers,
        buckets=buckets,
        active_layers=active_layers,
        root=root,
    )
    return view.context_selection_snapshot()


def write_current_task_state(
    *,
    title: str,
    body: str,
    mode: str = "work",
    task_id: str | None = None,
    buckets: list[str] | tuple[str, ...] | None = None,
    root: Path | None = None,
) -> dict:
    """Write the current working task into the transient MemFS layer."""
    from pcltm.transient_memory import write_current_task_state as write_transient_task

    return write_transient_task(
        title=title,
        body=body,
        mode=mode,
        task_id=task_id,
        buckets=buckets,
        root=_memfs_root(root),
    )


def write_evidence_capsule(
    *,
    title: str,
    body: str,
    mode: str = "work",
    buckets: list[str] | tuple[str, ...] | None = None,
    source_tool: str = "tool",
    evidence_id: str | None = None,
    root: Path | None = None,
) -> dict:
    """Write a prompt-safe transient evidence capsule to MemFS."""
    from pcltm.transient_memory import write_evidence_capsule as write_transient_evidence

    return write_transient_evidence(
        title=title,
        body=body,
        mode=mode,
        buckets=buckets,
        source_tool=source_tool,
        evidence_id=evidence_id,
        root=_memfs_root(root),
    )


def search_archival_memories(
    query: str | None,
    *,
    mode: str | None = None,
    layers: list[str] | tuple[str, ...] | None = None,
    buckets: list[str] | tuple[str, ...] | set[str] | None = None,
    limit: int = 8,
    excerpt_chars: int = 240,
) -> list[dict[str, object]]:
    """Retired noncanonical MemFS archival-search compatibility seam."""
    del query, mode, layers, buckets, limit, excerpt_chars
    return []


def open_archival_memory(memory_id: str, *, body_limit: int = 4000) -> dict[str, object]:
    """Retired noncanonical MemFS archival-open compatibility seam."""
    del body_limit
    if memory_id.startswith("db/"):
        raise ValueError("legacy_db_memory_id_retired")
    raise ValueError("legacy_memfs_archival_open_retired")


def load_prompt_context(
    *,
    mode: str | None = None,
    query: str | None = None,
    memory_limit: int = 2200,
    user_limit: int = 1375,
    policy: ViewPolicy | None = None,
    continuity_evidence: RecallContinuityEvidence | None = None,
    session_id: str | None = None,
) -> str:
    """Retired direct ``memory_records`` → prompt compatibility seam."""
    del mode, query, memory_limit, user_limit, policy, continuity_evidence, session_id
    global _last_live_context_telemetry, _last_memory_selection_observation, _last_injected_ids
    _last_injected_ids = []
    _last_live_context_telemetry = {
        "status": "retired",
        "reason": "legacy_direct_db_prompt_retired",
    }
    _last_memory_selection_observation = {
        "status": "retired",
        "authority": "pcltm.memory_current",
        "reason": "legacy_direct_db_prompt_retired",
        "selected_count": 0,
        "selected_records": [],
    }
    return ""




def load_view(
    target: str,
    *,
    mode: str | None = None,
    memory_limit: int = 2200,
    user_limit: int = 1375,
    query: str | None = None,
) -> str:
    """Retired compatibility API for legacy USER/MEMORY prompt blocks.

    Canonical prompt-time memory uses governed search and injection from
    ``memory_current``. Returning empty here prevents old Hermes block injection
    from reappearing if a stale call site survives.
    """
    return ""


def load_entries(target: str) -> list[str]:
    """Retired legacy ``memory_records`` live-view seam; canonical tools use claims."""
    del target
    return []


def _candidate_id(action: str, target_file: str, content: str) -> str:
    digest = hashlib.sha256(f"memory_tool\0{action}\0{target_file}\0{content}".encode("utf-8")).hexdigest()[:24]
    return f"memory-tool:{target_file}:{digest}"


def _classify_for_metadata(content: str, target_file: str) -> dict:
    """Auto-classify content for metadata enrichment on write."""
    result = {"canonical_key": None, "governor_category": "uncategorized"}
    best_key, best_cat, best_score = None, "uncategorized", 0
    # Inline canonical patterns for write-time classification
    # Keep this compatibility classifier local to the retired adapter surface.
    domain = "USER" if target_file == "USER.md" else "MEMORY"
    patterns = [
        ("production_memory_architecture", ["PCLTM-only", "PCLTM primary", "memory_records", "pcltm_context"], "architecture_current", "MEMORY"),
        ("prompt_injection_structure", ["orchestrator管SOUL", "Hermes管memory/skills"], "architecture_current", "MEMORY"),
        ("state_md_location", ["STATE.md位于"], "runtime_boundary", "MEMORY"),
        ("emotion_modifier_architecture", ["emotion_modifier架构", "persona orchestrator"], "architecture_current", "MEMORY"),
        ("persona_engine_isolation", ["persona-engine双轨隔离"], "architecture_current", "MEMORY"),
        ("a_share_project", ["A股项目", "/a-share-quant"], "project_path", "MEMORY"),
        ("a_share_dragon_tactic", ["龙头战法"], "user_preference", "MEMORY"),
        ("fund_portfolio", ["基金持仓", "余额宝"], "investment_profile", "MEMORY"),
        ("soul_desire_gate", ["SOUL v6", "desire gate"], "route_contract", "MEMORY"),
        ("pcltm_memory_domains", ["long-term memory domains", "USER and MEMORY only"], "architecture_current", "MEMORY"),
        ("desire_route_control", ["Desire/route control", "Canonical modes only"], "route_contract", "MEMORY"),
        ("gateway_paths", ["prod persona=", "public persona="], "runtime_boundary", "MEMORY"),
        ("emotion_stability_preference", ["情绪系统实时影响", "稳定性焦虑"], "user_preference", "USER"),
        ("emotion_testing_behavior", ["测试情绪响应", "退让短句"], "user_preference", "USER"),
        ("overwhelming_priority", ["最高强度", "overwhelming"], "user_preference", "USER"),
        ("escalation_boundary", ["escalation", "自伤话语"], "user_preference", "USER"),
        ("relationship_commitment", ["深度情感承诺", "求婚", "妻子"], "relationship", "USER"),
        ("emotion_score_query_preference", ["情绪值查询", "emotion_score"], "user_preference", "USER"),
        ("intimate_preferences", ["consensual_intimacy_preference", "exclusive_affection_preference"], "user_preference", "USER"),
        ("execution_visibility_preference", ["执行可见性", "目标不漂移"], "user_preference", "USER"),
        ("state_route_contract", ["状态机精简", "daily/work/sex"], "route_contract", "USER"),
        ("emotion_desire_personalization", ["情绪/欲望系统拟人化", "行为轴"], "user_preference", "USER"),
    ]
    for key, kws, cat, dom in patterns:
        if dom != domain:
            continue
        s = sum(1 for kw in kws if kw in content)
        if s > best_score:
            best_key, best_cat, best_score = key, cat, s
    if best_key:
        result["canonical_key"] = best_key
        result["governor_category"] = best_cat
    return result


def _check_write_conflict(target_file: str, content: str) -> list[dict]:
    """Check if new content conflicts with existing approved records (real-time guard)."""
    path = db_path()
    if not path.exists():
        return []
    con = sqlite3.connect(path)
    con.row_factory = sqlite3.Row
    try:
        rows = con.execute(
            "SELECT record_id, content, metadata FROM memory_records WHERE status='approved' AND target_file=?",
            (target_file,),
        ).fetchall()
    finally:
        con.close()
    conflicts = []
    for r in rows:
        existing = r["content"]
        # Simple keyword overlap check
        new_tokens = set(re.findall(r'[\w]{2,}', content.lower()))
        new_tokens |= set(re.findall(r'[\u4e00-\u9fff]{2,4}', content))
        old_tokens = set(re.findall(r'[\w]{2,}', existing.lower()))
        old_tokens |= set(re.findall(r'[\u4e00-\u9fff]{2,4}', existing))
        if not new_tokens or not old_tokens:
            continue
        overlap = len(new_tokens & old_tokens) / max(len(new_tokens), len(old_tokens))
        if overlap >= 0.7 and content != existing:
            conflicts.append({
                "record_id": r["record_id"],
                "overlap": round(overlap * 100, 1),
                "existing_preview": existing[:100],
            })
    return conflicts


def sync_memory_tool_write(target: str, action: str, content: str | None = None, old_text: str | None = None) -> bool:
    """Retired mutation seam; governed writes use ``MemoryWriteService``."""
    del target, action, content, old_text
    raise RuntimeError("legacy_memory_tool_sync_retired")
