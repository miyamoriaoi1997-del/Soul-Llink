#!/usr/bin/env python3
"""PCLTM Memory Governor — full governance pipeline.

Features:
- Rule-based canonical classification
- Rule-based + LLM semantic conflict detection
- Auto-merge overlapping records
- Memory scoring (importance, freshness, stability)
- STATE.md cross-validation
- Pending reflection cleanup
- Prompt injection guard

Modes:
- Default (daily cron): rule-based audit + scoring + auto-cleanup
- --semantic: enable LLM semantic conflict detection
- --merge: auto-apply merge suggestions
- --apply: write mutations to DB
- --dry-run: preview without writing
"""

import argparse
import json
import math
import os
import re
import sqlite3
import sys
from collections import defaultdict
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

from .runtime_paths import DEFAULT_DB, resolve_db_path

STATE_PATH = Path(os.getenv("HERMES_STATE_PATH", str(Path.home() / "soul-link" / "state" / "STATE.md")))

# ---------------------------------------------------------------------------
# Canonical key classification
# ---------------------------------------------------------------------------

CANONICAL_PATTERNS = [
    ("production_memory_architecture", ["PCLTM-only", "PCLTM primary", "memory_records", "legacy USER/MEMORY blocks", "pcltm_context"], "architecture_current", "MEMORY"),
    ("prompt_injection_structure", ["orchestrator管SOUL", "Hermes管memory/skills", "active模式"], "architecture_current", "MEMORY"),
    ("state_md_location", ["STATE.md位于", "STATE.md（非子目录）"], "runtime_boundary", "MEMORY"),
    ("emotion_modifier_architecture", ["emotion_modifier架构", "persona orchestrator", "emotion_state_manager"], "architecture_current", "MEMORY"),
    ("persona_engine_isolation", ["persona-engine双轨隔离", "公开用模板", "真实聊天样本留私有"], "architecture_current", "MEMORY"),
    ("a_share_project", ["A股项目", "/a-share-quant"], "project_path", "MEMORY"),
    ("a_share_dragon_tactic", ["龙头战法"], "user_preference", "MEMORY"),
    ("a_share_data_source", ["金融/市场分析应优先使用"], "project_path", "MEMORY"),
    ("fund_portfolio", ["基金持仓", "余额宝", "广发电力", "华夏全球"], "investment_profile", "MEMORY"),
    ("soul_desire_gate", ["SOUL v6", "desire gate", "emotion-first"], "route_contract", "MEMORY"),
    ("persona_memory_maintenance_cron", ["persona-memory-maintenance", "周一03:00"], "runtime_boundary", "MEMORY"),
    ("pcltm_memory_domains", ["long-term memory domains", "USER and MEMORY only"], "architecture_current", "MEMORY"),
    ("desire_route_control", ["Desire/route control", "Canonical modes only", "daily/work/sex"], "route_contract", "MEMORY"),
    ("gateway_paths", ["prod persona=", "public persona=", "Hermes="], "runtime_boundary", "MEMORY"),
    ("qq_message_injection", ["QQ消息经过中转站", "中转站"], "runtime_boundary", "MEMORY"),
    ("emotion_stability_preference", ["情绪系统实时影响", "稳定性焦虑", "不轻易重构"], "user_preference", "USER"),
    ("emotional_response_expectation", ["期待明确的情感回应", "辛苦了", "🥹"], "user_preference", "USER"),
    ("emotion_testing_behavior", ["测试情绪响应", "退让短句", "关系否定语"], "user_preference", "USER"),
    ("overwhelming_priority", ["最高强度", "overwhelming", "情绪注入优先级"], "user_preference", "USER"),
    ("escalation_boundary", ["escalation", "极端", "自伤话语", "拒绝越界"], "user_preference", "USER"),
    ("relationship_commitment", ["深度情感承诺", "求婚", "妻子"], "relationship", "USER"),
    ("emotion_score_query_preference", ["情绪值查询", "emotion_score", "STATE.md"], "user_preference", "USER"),
    ("intimate_preferences", ["consensual_intimacy_preference", "exclusive_affection_preference"], "user_preference", "USER"),
    ("execution_visibility_preference", ["执行可见性", "目标不漂移", "只读扫描"], "user_preference", "USER"),
    ("pcltm_direct_injection_preference", ["PCLTM direct injection", "active assembly", "selected_records"], "user_preference", "USER"),
    ("pcltm_dac_role", ["PCLTM-DAC", "短期连续性", "压缩边界审计"], "user_preference", "USER"),
    ("state_route_contract", ["状态机精简", "daily/work/sex", "明确退出"], "route_contract", "USER"),
    ("emotion_desire_personalization", ["情绪/欲望系统拟人化", "行为轴", "少硬编码"], "user_preference", "USER"),
    ("state_machine_test_env_isolation", ["env -u HERMES_STATE_PATH", "生产 STATE 污染", "fixture"], "user_preference", "USER"),
    ("emotion_tuning_preference", ["简洁、可泛化", "patience 扣减", "自评系统"], "user_preference", "USER"),
    ("ordinary_chat_emotion_gain", ["普通聊天", "少量提升情绪值", "0.5"], "user_preference", "USER"),
    ("hermes_substrate_preference", ["replaceable execution substrate", "PCLTM-context/persona runtime", "broad Hermes changes"], "user_preference", "USER"),
    ("cleanup_preference", ["留尾巴", "彻底删除活跃路径", "无用归档状态文件"], "user_preference", "USER"),
    ("migration_staged_preference", ["分阶段推进", "readiness/doctor", "回滚线"], "user_preference", "USER"),
    ("migration_minimal_rollback_preference", ["避免完整可回滚切换层", "最小临时保护", "临时脚手架"], "user_preference", "USER"),
    ("upstream_skill_preference", ["Hermes 自带/上游技能", "用户自有技能", "覆盖层"], "user_preference", "USER"),
    ("autonomous_execution_preference", ["自主推进低风险决策", "用户暂时不在", "安全边界"], "user_preference", "USER"),
    ("config_driven_rules_preference", ["避免零散硬编码", "结构化、可配置", "评测集覆盖"], "user_preference", "USER"),
    ("skill_maintenance_preference", ["class-level umbrella", "references/", "一会话一技能"], "user_preference", "USER"),
    ("public_readme_preference", ["README", "说明文档", "覆盖每个系统和模块"], "user_preference", "USER"),
    ("public_edition_isolation", ["isolated publishable project", "private data removed", "agent integrations"], "user_preference", "USER"),
    ("pcltm_architecture", ["PCLTM=memory+state governance", "DAC=schema summaries", "semantic compression=reflection"], "architecture_current", "MEMORY"),
    ("pcltm_migration_cleanup", ["active hermes_persona_lcm imports", "PCLTM DAC support modules now live under", "Soul-Link cleanup 2026-05-13"], "architecture_current", "MEMORY"),
    ("public_persona_retirement", ["persona-engine-public", "旧同步/状态脚本", "脱敏发布流水线"], "runtime_boundary", "MEMORY"),
    ("soul_link_scope", ["Hermes itself is not a migration target", "dynamic model router", "thin Hermes adapters"], "architecture_current", "MEMORY"),
    ("pcltm_context_sanitizer", ["orphan/late/duplicate tool-result", "compaction-handoff-as-reference", "context-engine adapter"], "architecture_current", "MEMORY"),
    ("model_router_runtime_path", ["model-router runtime port", "model-router-proxy.service", "model-router.example.yaml"], "runtime_boundary", "MEMORY"),
    ("public_repo_sync", ["published/mirrored to public repositories", "public-reference", "persona-engine"], "runtime_boundary", "MEMORY"),
    ("sex_fixture_policy", ["sex fixture 漂移", "现有状态机输出为准", "不主动改路由规则"], "route_contract", "MEMORY"),
]

CONFLICT_RULES = [
    ("production_memory_architecture", "PCLTM-only", ["两套独立系统", "dual memory", "双系统"]),
    ("production_memory_architecture", "PCLTM-only", ["MEMORY.md", "USER.md", "file-backed"]),
    ("state_route_contract", "daily/work/sex", ["intimacy", "repair", "conflict", "sex_candidate"]),
    ("soul_desire_gate", "emotion-first", []),
    ("pcltm_memory_domains", "USER and MEMORY only", ["relationship-memory", "MOMENTS"]),
]

# Category → base importance weight
CATEGORY_IMPORTANCE = {
    "architecture_current": 0.95,
    "route_contract": 0.90,
    "runtime_boundary": 0.75,
    "user_preference": 0.70,
    "relationship": 0.80,
    "project_path": 0.55,
    "investment_profile": 0.60,
    "uncategorized": 0.40,
}

# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------

@dataclass
class Record:
    record_id: int
    kind: str
    target_file: str
    content: str
    status: str
    metadata: dict
    created_at: str
    reviewer: str
    decision_reason: str

@dataclass
class ClassifiedRecord:
    record: Record
    canonical_key: Optional[str]
    category: str
    domain: str
    is_current: bool
    is_stale: bool

@dataclass
class Conflict:
    canonical_key: str
    current_records: list = field(default_factory=list)
    stale_records: list = field(default_factory=list)
    description: str = ""
    source: str = "rule"  # "rule" or "semantic"

@dataclass
class ScoredRecord:
    classified: ClassifiedRecord
    importance_score: float = 0.0
    freshness_score: float = 0.0
    stability_score: float = 0.0
    overall_score: float = 0.0

@dataclass
class MergeSuggestion:
    canonical_key: str
    record_ids: list
    merged_content: str
    overlap_pct: float

# ---------------------------------------------------------------------------
# Memory type classification (Episodic / Semantic / Procedural)
# ---------------------------------------------------------------------------

# Episodic markers: specific events, timestamps, "那次/当时/上次" references
_EPISODIC_MARKERS = (
    "那次", "当时", "上次", "之前", "昨天", "今天", "刚才", "第一次",
    "求婚", "表白", "吵架", "和好", "做爱", "高潮",
    "2024", "2025", "2026",
)

# Procedural markers: how-to, workflow, steps
_PROCEDURAL_MARKERS = (
    "步骤", "流程", "方法", "脚本", "命令", "配置", "部署",
    "cron", "script", "pipeline", "workflow",
)


def classify_memory_type(record: Record) -> str:
    """Classify a record as semantic, episodic, or procedural.

    - Episodic: specific events with temporal/contextual anchors. Cannot be UPDATE'd, only appended.
    - Semantic: decontextualized facts/preferences. Can be UPDATE'd (superseded).
    - Procedural: reusable workflows/steps. Can be UPDATE'd.
    """
    text = record.content.lower()
    meta = record.metadata

    # Explicit override from metadata
    if meta.get("memory_type"):
        return meta["memory_type"]

    # Check for episodic markers
    episodic_hits = sum(1 for m in _EPISODIC_MARKERS if m.lower() in text)
    if episodic_hits >= 2:
        return "episodic"

    # Check for procedural markers
    procedural_hits = sum(1 for m in _PROCEDURAL_MARKERS if m.lower() in text)
    if procedural_hits >= 2:
        return "procedural"

    # USER.md records are almost always semantic (preferences, facts)
    if record.target_file == "USER.md":
        return "semantic"

    # MEMORY.md with relationship/emotional content and temporal markers → episodic
    if record.target_file == "MEMORY.md" and episodic_hits >= 1:
        category = meta.get("gov_category", meta.get("category", ""))
        if category in ("relationship", "user_preference"):
            return "episodic"

    return "semantic"



def get_db() -> str:
    return str(resolve_db_path())

def load_records(db: str, statuses=None) -> list[Record]:
    if statuses is None:
        statuses = ("approved",)
    con = sqlite3.connect(db)
    con.row_factory = sqlite3.Row
    ph = ",".join("?" * len(statuses))
    rows = con.execute(
        f"SELECT * FROM memory_records WHERE status IN ({ph}) ORDER BY target_file, record_id",
        statuses,
    ).fetchall()
    con.close()
    return [Record(
        record_id=r["record_id"], kind=r["kind"], target_file=r["target_file"],
        content=r["content"], status=r["status"],
        metadata=json.loads(r["metadata"] or "{}"),
        created_at=r["created_at"], reviewer=r["reviewer"],
        decision_reason=r["decision_reason"],
    ) for r in rows]

def classify_record(record: Record) -> ClassifiedRecord:
    text = record.content
    best_key, best_cat, best_dom, best_score = None, "uncategorized", record.target_file, 0
    for key, kws, cat, dom in CANONICAL_PATTERNS:
        s = sum(1 for kw in kws if kw in text)
        if s > best_score:
            best_key, best_cat, best_dom, best_score = key, cat, dom, s
    stale = _check_staleness(record, best_key)
    return ClassifiedRecord(record=record, canonical_key=best_key, category=best_cat,
                            domain=best_dom, is_current=not stale, is_stale=stale)

def _negation_present(text: str) -> bool:
    return any(n in text for n in ("Do not reintroduce", "retired", "已retire", "不要重新引入", "禁止"))

def _check_staleness(record: Record, canonical_key: Optional[str]) -> bool:
    if not canonical_key:
        return False
    meta = record.metadata
    if meta.get("inject_policy") in ("no_prompt", "never", "retrieve_only"):
        if record.status == "approved" and record.kind == "reflection":
            return True
    for key, cur_kw, stale_pats in CONFLICT_RULES:
        if canonical_key == key:
            for pat in stale_pats:
                if _negation_present(record.content):
                    continue
                if pat in record.content and cur_kw not in record.content:
                    return True
    return False

# ---------------------------------------------------------------------------
# Rule-based conflict detection
# ---------------------------------------------------------------------------

def detect_conflicts(classified: list[ClassifiedRecord]) -> list[Conflict]:
    by_key = defaultdict(list)
    for cr in classified:
        if cr.canonical_key:
            by_key[cr.canonical_key].append(cr)
    conflicts = []
    for key, items in by_key.items():
        currents = [cr for cr in items if cr.is_current]
        stales = [cr for cr in items if cr.is_stale]
        if stales and currents:
            conflicts.append(Conflict(canonical_key=key, current_records=currents,
                                      stale_records=stales,
                                      description=f"{key}: {len(currents)} current, {len(stales)} stale",
                                      source="rule"))
        elif len(items) > 3:
            conflicts.append(Conflict(canonical_key=key, current_records=items, stale_records=[],
                                      description=f"{key}: {len(items)} records (possible dedup)",
                                      source="rule"))
    # Cross-key
    all_content = " ".join(cr.record.content for cr in classified if cr.is_current)
    for key, cur_kw, stale_pats in CONFLICT_RULES:
        for pat in stale_pats:
            if pat in all_content and cur_kw in all_content:
                matching = [cr for cr in classified if pat in cr.record.content and cr.is_current
                            and not _negation_present(cr.record.content)]
                if matching:
                    conflicts.append(Conflict(canonical_key=key, current_records=[],
                                              stale_records=matching,
                                              description=f"{key}: '{pat}' alongside '{cur_kw}'",
                                              source="rule"))
    return conflicts

# ---------------------------------------------------------------------------
# LLM semantic conflict detection
# ---------------------------------------------------------------------------

def _llm_call(prompt: str, timeout: float = 30, llm_client=None) -> Optional[str]:
    """Call an optional semantic-review LLM client.

    PCLTM core must stay host-neutral.  Callers that want semantic conflict
    detection should pass a client object/function from their host adapter.  The
    accepted forms are intentionally small:

    - callable(prompt=..., timeout=...) -> str | response
    - object with call(prompt=..., timeout=...) -> str | response
    - object with complete(prompt=..., timeout=...) -> str | response

    Response extraction accepts plain strings and simple OpenAI-like response
    objects/dicts so adapters can remain thin.
    """
    if llm_client is None:
        print("  [LLM] Semantic review skipped: no llm_client provided", file=sys.stderr)
        return None
    try:
        if callable(llm_client):
            response = llm_client(prompt=prompt, timeout=timeout)
        elif hasattr(llm_client, "call"):
            response = llm_client.call(prompt=prompt, timeout=timeout)
        elif hasattr(llm_client, "complete"):
            response = llm_client.complete(prompt=prompt, timeout=timeout)
        else:
            raise TypeError("llm_client must be callable or expose call()/complete()")
        return _extract_llm_text(response)
    except Exception as e:
        print(f"  [LLM] Error: {e}", file=sys.stderr)
        return None


def _extract_llm_text(response) -> Optional[str]:
    """Extract text from common host-adapter LLM response shapes."""
    if response is None:
        return None
    if isinstance(response, str):
        return response
    if isinstance(response, dict):
        if isinstance(response.get("content"), str):
            return response["content"]
        choices = response.get("choices") or []
        if choices:
            first = choices[0]
            if isinstance(first, dict):
                message = first.get("message") or {}
                if isinstance(message, dict) and isinstance(message.get("content"), str):
                    return message["content"]
                if isinstance(first.get("text"), str):
                    return first["text"]
    content = getattr(response, "content", None)
    if isinstance(content, str):
        return content
    choices = getattr(response, "choices", None)
    if choices:
        first = choices[0]
        message = getattr(first, "message", None)
        msg_content = getattr(message, "content", None)
        if isinstance(msg_content, str):
            return msg_content
        text = getattr(first, "text", None)
        if isinstance(text, str):
            return text
    return str(response)

def detect_semantic_conflicts(classified: list[ClassifiedRecord], llm_client=None) -> list[Conflict]:
    """Use an optional LLM client to find semantic conflicts by canonical_key."""
    by_key = defaultdict(list)
    for cr in classified:
        if cr.canonical_key and cr.is_current:
            by_key[cr.canonical_key].append(cr)
    conflicts = []
    for key, items in by_key.items():
        if len(items) < 2:
            continue
        # Pre-filter: only check pairs with >20% keyword overlap (skip obviously unrelated)
        for i in range(len(items)):
            for j in range(i + 1, len(items)):
                a, b = items[i], items[j]
                overlap = _keyword_overlap(a.record.content, b.record.content)
                if overlap < 0.2:
                    continue
                prompt = (
                    f"Are these two memory records contradictory?\n"
                    f"A: {a.record.content[:200]}\n"
                    f"B: {b.record.content[:200]}\n"
                    'Reply JSON only: {"c":bool,"s":bool,"n":"A"|"B"|"=","r":"short reason"}\n'
                    "c=contradictory, s=same fact, n=which is newer (or = if same)"
                )
                resp = _llm_call(prompt, llm_client=llm_client)
                if not resp:
                    continue
                try:
                    json_match = re.search(r'\{[^}]+\}', resp, re.DOTALL)
                    if not json_match:
                        continue
                    result = json.loads(json_match.group())
                    if result.get("c"):
                        newer = result.get("n", "=")
                        current = a if newer == "A" else (b if newer == "B" else a)
                        stale = b if newer == "A" else (a if newer == "B" else b)
                        conflicts.append(Conflict(
                            canonical_key=key,
                            current_records=[current],
                            stale_records=[stale],
                            description=f"Semantic: {key} — {result.get('r', 'no reason')}",
                            source="semantic",
                        ))
                except (json.JSONDecodeError, KeyError):
                    continue
    return conflicts

# ---------------------------------------------------------------------------
# Auto-merge overlapping records
# ---------------------------------------------------------------------------

def _keyword_overlap(a: str, b: str) -> float:
    """Compute keyword overlap ratio between two texts."""
    def tokens(t: str) -> set[str]:
        toks = set()
        for w in re.findall(r'[\w]{2,}', t.lower()):
            toks.add(w)
        for w in re.findall(r'[\u4e00-\u9fff]{2,4}', t):
            toks.add(w)
        return toks
    ta, tb = tokens(a), tokens(b)
    if not ta or not tb:
        return 0.0
    return len(ta & tb) / max(len(ta), len(tb))

def suggest_merges(classified: list[ClassifiedRecord]) -> list[MergeSuggestion]:
    by_key = defaultdict(list)
    for cr in classified:
        if cr.canonical_key and cr.is_current:
            by_key[cr.canonical_key].append(cr)
    suggestions = []
    for key, items in by_key.items():
        if len(items) < 2:
            continue
        # Check first pair only (groups are typically small)
        a, b = items[0], items[1]
        overlap = _keyword_overlap(a.record.content, b.record.content)
        if overlap >= 0.4:
            # Build merged content: take the longer one and append unique info from shorter
            long_rec, short_rec = (a, b) if len(a.record.content) >= len(b.record.content) else (b, a)
            long_text, short_text = long_rec.record.content, short_rec.record.content
            # Find sentences in short that aren't in long
            short_parts = re.split(r'[；;。]', short_text)
            unique_parts = [p.strip() for p in short_parts if p.strip() and p.strip() not in long_text]
            merged = long_text
            if unique_parts:
                merged = long_text.rstrip("。；") + "；" + "；".join(unique_parts)
            suggestions.append(MergeSuggestion(
                canonical_key=key,
                record_ids=[a.record.record_id, b.record.record_id],
                merged_content=merged[:400],  # cap at reasonable length
                overlap_pct=round(overlap * 100, 1),
            ))
    return suggestions

# ---------------------------------------------------------------------------
# Memory scoring
# ---------------------------------------------------------------------------

def _weibull_decay(age_hours: float, eta: float = 1440, kappa: float = 0.7) -> float:
    """Weibull decay function.

    κ < 1 gives rapid early forgetting that slows over time (matches human memory).
    η (scale) controls the time constant — default 1440h ≈ 60 days.
    """
    if age_hours <= 0:
        return 1.0
    return math.exp(-((age_hours / eta) ** kappa))


def _retrieval_boost(retrieval_count: int, last_retrieved_at: str | None, now: datetime) -> float:
    """Boost score based on retrieval frequency and recency.

    Each retrieval resets the effective age. Frequently retrieved memories
    stay fresh longer — this implements the "testing effect" from cognitive science.
    """
    if retrieval_count <= 0:
        return 0.0
    # Recency of last retrieval
    recency = 0.0
    if last_retrieved_at:
        try:
            last_ret = datetime.fromisoformat(last_retrieved_at.replace("Z", "+00:00"))
            hours_since = max(0, (now - last_ret).total_seconds() / 3600)
            recency = _weibull_decay(hours_since, eta=720, kappa=0.8)
        except Exception:
            pass
    # Frequency component: diminishing returns via log
    freq = min(0.3, math.log1p(retrieval_count) * 0.08)
    return round(freq + recency * 0.15, 4)


def score_records(classified: list[ClassifiedRecord], db: str) -> list[ScoredRecord]:
    scored = []
    now = datetime.now(timezone.utc)

    # Pre-fetch retrieval + citation stats for all records in one query
    retrieval_stats: dict[int, tuple[int, str | None]] = {}
    citation_stats: dict[int, tuple[int, str | None]] = {}
    try:
        con = sqlite3.connect(db)
        con.row_factory = sqlite3.Row
        rows = con.execute(
            "SELECT record_id, retrieval_count, last_retrieved_at, citation_count, last_cited_at FROM memory_records WHERE status='approved'"
        ).fetchall()
        for r in rows:
            retrieval_stats[r["record_id"]] = (r["retrieval_count"] or 0, r["last_retrieved_at"])
            citation_stats[r["record_id"]] = (r["citation_count"] or 0, r["last_cited_at"])
        con.close()
    except Exception:
        pass

    for cr in classified:
        # Importance from category
        imp = CATEGORY_IMPORTANCE.get(cr.category, 0.5)
        # Boost for critical tags
        meta = cr.record.metadata
        tags = set(str(t).lower() for t in meta.get("tags", []) if t)
        if tags & {"critical", "production", "boundary", "identity"}:
            imp = min(1.0, imp + 0.15)

        # Freshness: Weibull decay from created_at (or last_retrieved_at if newer)
        try:
            created = datetime.fromisoformat(cr.record.created_at.replace("Z", "+00:00"))
            ret_count, last_ret_at = retrieval_stats.get(cr.record.record_id, (0, None))
            # Effective age = time since last retrieval OR creation, whichever is newer
            effective_time = created
            if last_ret_at:
                try:
                    ret_time = datetime.fromisoformat(last_ret_at.replace("Z", "+00:00"))
                    if ret_time > effective_time:
                        effective_time = ret_time
                except Exception:
                    pass
            age_hours = max(0, (now - effective_time).total_seconds() / 3600)
            fresh = _weibull_decay(age_hours)
        except Exception:
            fresh = 0.5
            ret_count, last_ret_at = 0, None

        # Retrieval boost
        ret_boost = _retrieval_boost(ret_count, last_ret_at, now)

        # Citation boost: records actually used by the model get higher weight
        cite_count, last_cited = citation_stats.get(cr.record.record_id, (0, None))
        cite_boost = 0.0
        if cite_count > 0:
            # Frequency: diminishing returns via log, capped at 0.4
            cite_freq = min(0.4, math.log1p(cite_count) * 0.12)
            # Recency of last citation
            cite_recency = 0.0
            if last_cited:
                try:
                    cited_time = datetime.fromisoformat(last_cited.replace("Z", "+00:00"))
                    hours_since_cite = max(0, (now - cited_time).total_seconds() / 3600)
                    cite_recency = _weibull_decay(hours_since_cite, eta=480, kappa=0.9)
                except Exception:
                    pass
            cite_boost = round(cite_freq + cite_recency * 0.2, 4)

        # Stability: check how many superseded siblings exist for this canonical_key
        stab = 0.8  # default
        if cr.canonical_key:
            try:
                con = sqlite3.connect(db)
                sup_count = con.execute(
                    "SELECT count(*) FROM memory_records WHERE status='superseded' AND target_file=? AND content LIKE ?",
                    (cr.record.target_file, f"%{cr.canonical_key.split('_')[0]}%"),
                ).fetchone()[0]
                con.close()
                # More superseded siblings = topic is volatile = lower stability
                stab = max(0.3, 0.9 - sup_count * 0.03)
            except Exception:
                pass

        # Final score: importance 40% + freshness 20% + stability 23% + retrieval 8% + citation 9%
        overall = round(imp * 0.40 + fresh * 0.20 + stab * 0.23 + ret_boost * 0.08 + cite_boost * 0.09, 3)
        scored.append(ScoredRecord(
            classified=cr,
            importance_score=round(imp, 3),
            freshness_score=round(fresh, 3),
            stability_score=round(stab, 3),
            overall_score=overall,
        ))
    return sorted(scored, key=lambda s: s.overall_score, reverse=True)


def persist_scores(db: str, scored: list[ScoredRecord]) -> int:
    """Write governor scores and related_ids into PCLTM record metadata."""
    # Compute memory links using semantic index
    related_map: dict[int, list[int]] = {}
    try:
        sys.path.insert(0, str(Path(__file__).resolve().parent))
        from semantic_index import get_index
        idx = get_index(db)
        for s in scored:
            rid = s.classified.record.record_id
            related = idx.related(rid, top_k=3)
            if related:
                related_map[rid] = [r_id for r_id, _ in related]
    except Exception:
        pass

    con = sqlite3.connect(db)
    updated = 0
    for s in scored:
        rid = s.classified.record.record_id
        # Classify memory type
        mem_type = classify_memory_type(s.classified.record)
        new_meta = {
            **s.classified.record.metadata,
            "gov_score": s.overall_score,
            "gov_importance": s.importance_score,
            "gov_freshness": s.freshness_score,
            "gov_stability": s.stability_score,
            "gov_canonical_key": s.classified.canonical_key,
            "gov_category": s.classified.category,
            "gov_scored_at": datetime.now(timezone.utc).isoformat(),
            "memory_type": mem_type,
        }
        # Memory linking: store related record IDs
        if rid in related_map:
            new_meta["related_ids"] = related_map[rid]
        con.execute(
            "UPDATE memory_records SET metadata=?, memory_type=? WHERE record_id=? AND status='approved'",
            (json.dumps(new_meta, ensure_ascii=False), mem_type, rid),
        )
        updated += 1
    con.commit()
    con.close()
    return updated

# ---------------------------------------------------------------------------
# STATE.md cross-validation
# ---------------------------------------------------------------------------

def cross_validate_state(classified: list[ClassifiedRecord]) -> list[dict]:
    if not STATE_PATH.exists():
        return []
    state_text = STATE_PATH.read_text(encoding="utf-8", errors="replace")
    issues = []
    # Check if any MEMORY record references specific emotion/desire values that contradict STATE
    for cr in classified:
        if cr.record.target_file != "MEMORY.md":
            continue
        content = cr.record.content
        # Look for explicit score/desire references
        score_matches = re.findall(r'(?:score|emotion_score|好感度|信任度|占有欲)[\s:=]+[\d.]+', content, re.I)
        for match in score_matches:
            # Try to extract the value
            val_match = re.search(r'([\d.]+)', match)
            if val_match:
                val = float(val_match.group(1))
                if val > 200 or val < 0:
                    issues.append({
                        "record_id": cr.record.record_id,
                        "issue": f"Unusual score reference: {match}",
                        "canonical_key": cr.canonical_key,
                        "severity": "low",
                    })
    return issues

# ---------------------------------------------------------------------------
# Pending / policy checks (unchanged)
# ---------------------------------------------------------------------------

def check_pending_pollution(db: str) -> list[dict]:
    con = sqlite3.connect(db)
    con.row_factory = sqlite3.Row
    rows = con.execute(
        "SELECT record_id, kind, target_file, substr(content,1,200) content, metadata "
        "FROM memory_records WHERE status='pending' ORDER BY record_id"
    ).fetchall()
    con.close()
    issues = []
    for r in rows:
        meta = json.loads(r["metadata"] or "{}")
        if meta.get("inject_policy") != "retrieve_only" and meta.get("category") == "reflection":
            issues.append({"record_id": r["record_id"], "issue": "pending reflection without retrieve_only",
                           "kind": r["kind"], "content_preview": r["content"][:100]})
    return issues

def check_inject_policy_consistency(classified: list[ClassifiedRecord]) -> list[dict]:
    issues = []
    for cr in classified:
        if cr.record.status != "approved":
            continue
        policy = cr.record.metadata.get("inject_policy", "")
        if cr.record.kind == "reflection" and policy not in ("retrieve_only", "no_prompt", "never", ""):
            issues.append({"record_id": cr.record.record_id,
                           "issue": f"reflection with inject_policy={policy}",
                           "canonical_key": cr.canonical_key})
    return issues

# ---------------------------------------------------------------------------
# Report generation
# ---------------------------------------------------------------------------

def generate_report(
    classified: list[ClassifiedRecord],
    conflicts: list[Conflict],
    pending_issues: list[dict],
    policy_issues: list[dict],
    scored: list[ScoredRecord] = None,
    merges: list[MergeSuggestion] = None,
    state_issues: list[dict] = None,
) -> str:
    L = []
    L.append("=" * 60)
    L.append("PCLTM MEMORY GOVERNOR AUDIT REPORT")
    L.append(f"Generated: {datetime.now().isoformat()}")
    L.append("=" * 60)

    approved = [cr for cr in classified if cr.record.status == "approved"]
    categorized = [cr for cr in approved if cr.canonical_key]
    uncategorized = [cr for cr in approved if not cr.canonical_key]
    stale = [cr for cr in approved if cr.is_stale]
    semantic_conflicts = [c for c in conflicts if c.source == "semantic"]

    L.append(f"\n## SUMMARY")
    L.append(f"Approved: {len(approved)} | Categorized: {len(categorized)} | Uncategorized: {len(uncategorized)}")
    L.append(f"Stale: {len(stale)} | Rule conflicts: {len(conflicts) - len(semantic_conflicts)} | Semantic conflicts: {len(semantic_conflicts)}")
    L.append(f"Pending risks: {len(pending_issues)} | Policy issues: {len(policy_issues)}")
    if merges:
        L.append(f"Merge candidates: {len(merges)}")
    if state_issues:
        L.append(f"STATE inconsistencies: {len(state_issues)}")

    # Classification
    L.append(f"\n## CANONICAL CLASSIFICATION")
    by_key = defaultdict(list)
    for cr in categorized:
        by_key[cr.canonical_key].append(cr)
    for key in sorted(by_key):
        items = by_key[key]
        L.append(f"\n### {key} ({len(items)})")
        for cr in items:
            m = " [STALE]" if cr.is_stale else ""
            sc = ""
            if scored:
                for s in scored:
                    if s.classified.record.record_id == cr.record.record_id:
                        sc = f" score={s.overall_score}"
                        break
            L.append(f"  [{cr.record.record_id}] {cr.category}/{cr.record.kind}{m}{sc}")
            L.append(f"    {cr.record.content[:100]}")

    if uncategorized:
        L.append(f"\n### UNCATEGORIZED ({len(uncategorized)})")
        for cr in uncategorized:
            L.append(f"  [{cr.record.record_id}] {cr.record.kind}: {cr.record.content[:100]}")

    # Conflicts
    if conflicts:
        L.append(f"\n## CONFLICTS ({len(conflicts)})")
        for c in conflicts:
            tag = " [SEMANTIC]" if c.source == "semantic" else ""
            L.append(f"\n### {c.canonical_key}{tag}")
            L.append(f"  {c.description}")
            for cr in c.stale_records:
                L.append(f"    -> [{cr.record.record_id}] {cr.record.content[:100]}")

    # Merges
    if merges:
        L.append(f"\n## MERGE SUGGESTIONS ({len(merges)})")
        for m in merges:
            L.append(f"\n### {m.canonical_key} (overlap={m.overlap_pct}%)")
            L.append(f"  Records: {m.record_ids}")
            L.append(f"  Merged: {m.merged_content[:150]}")

    # Scores
    if scored:
        L.append(f"\n## SCORES (top 5 / bottom 5)")
        for s in scored[:5]:
            L.append(f"  [{s.classified.record.record_id}] {s.overall_score} (imp={s.importance_score} fresh={s.freshness_score} stab={s.stability_score}) {s.classified.canonical_key}")
        if len(scored) > 5:
            L.append(f"  ...")
            for s in scored[-5:]:
                L.append(f"  [{s.classified.record.record_id}] {s.overall_score} (imp={s.importance_score} fresh={s.freshness_score} stab={s.stability_score}) {s.classified.canonical_key}")

    # STATE
    if state_issues:
        L.append(f"\n## STATE CROSS-VALIDATION ({len(state_issues)})")
        for si in state_issues:
            L.append(f"  [{si['record_id']}] {si['issue']} (key={si['canonical_key']})")

    # Pending / Policy
    if pending_issues:
        L.append(f"\n## PENDING RISKS ({len(pending_issues)})")
        for i in pending_issues:
            L.append(f"  [{i['record_id']}] {i['issue']}")
    if policy_issues:
        L.append(f"\n## POLICY ISSUES ({len(policy_issues)})")
        for i in policy_issues:
            L.append(f"  [{i['record_id']}] {i['issue']}")

    # Recommendations
    L.append(f"\n## RECOMMENDATIONS")
    recs = []
    if conflicts:
        recs.append(f"Review {len(conflicts)} conflict(s)")
    if merges:
        recs.append(f"Apply {len(merges)} merge(s)")
    if uncategorized:
        recs.append(f"Review {len(uncategorized)} uncategorized record(s)")
    if state_issues:
        recs.append(f"Check {len(state_issues)} STATE inconsistency(ies)")
    if pending_issues:
        recs.append(f"Clean {len(pending_issues)} pending reflection(s)")
    if not recs:
        recs.append("No issues found. Memory is clean.")
    for r in recs:
        L.append(f"  - {r}")

    return "\n".join(L)

# ---------------------------------------------------------------------------
# Apply fixes
# ---------------------------------------------------------------------------

def apply_fixes(
    db: str,
    conflicts: list[Conflict],
    pending_issues: list[dict],
    merges: list[MergeSuggestion] = None,
    dry_run: bool = False,
) -> str:
    con = sqlite3.connect(db)
    actions = []

    # 1. Supersede pending reflections
    for issue in pending_issues:
        rid = issue["record_id"]
        if not dry_run:
            con.execute(
                "UPDATE memory_records SET status='superseded', decision_reason='governor: pending reflection', reviewed_at=strftime('%Y-%m-%dT%H:%M:%fZ','now') WHERE record_id=? AND status='pending'",
                (rid,),
            )
        actions.append(f"supersede pending [{rid}]")

    # 2. Supersede stale/conflicting records
    for conflict in conflicts:
        for cr in conflict.stale_records:
            rid = cr.record.record_id
            if cr.record.status == "approved" and cr.is_stale:
                if not dry_run:
                    con.execute(
                        "UPDATE memory_records SET status='superseded', decision_reason='governor: conflict', reviewed_at=strftime('%Y-%m-%dT%H:%M:%fZ','now') WHERE record_id=?",
                        (rid,),
                    )
                actions.append(f"supersede stale [{rid}] ({conflict.canonical_key})")

    # 3. Apply merges
    if merges:
        for m in merges:
            keep_id = m.record_ids[0]
            supersede_ids = m.record_ids[1:]
            if not dry_run:
                # Update the kept record with merged content
                con.execute(
                    "UPDATE memory_records SET content=?, metadata=json_set(COALESCE(metadata,'{}'),'$.canonical_key',?), reviewed_at=strftime('%Y-%m-%dT%H:%M:%fZ','now') WHERE record_id=? AND status='approved'",
                    (m.merged_content, m.canonical_key, keep_id),
                )
                # Supersede the others
                for sid in supersede_ids:
                    con.execute(
                        "UPDATE memory_records SET status='superseded', decision_reason='governor: merged into '||?, reviewed_at=strftime('%Y-%m-%dT%H:%M:%fZ','now') WHERE record_id=?",
                        (str(keep_id), sid),
                    )
            actions.append(f"merge {m.record_ids} -> [{keep_id}] ({m.canonical_key})")

    if not dry_run:
        con.commit()
    con.close()
    mode = "DRY RUN" if dry_run else "APPLIED"
    result = f"[{mode}] {len(actions)} action(s): " + "; ".join(actions[:15])
    if len(actions) > 15:
        result += f"; ... +{len(actions) - 15} more"
    return result

# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="PCLTM Memory Governor")
    parser.add_argument("--apply", action="store_true", help="Apply fixes")
    parser.add_argument("--dry-run", action="store_true", help="Preview changes")
    parser.add_argument("--semantic", action="store_true", help="Enable LLM semantic conflict detection")
    parser.add_argument("--merge", action="store_true", help="Apply merge suggestions")
    parser.add_argument("--score", action="store_true", help="Output scores (default: always score)")
    parser.add_argument("--db", default=None, help="PCLTM DB path")
    args = parser.parse_args()

    db = args.db or get_db()
    if not Path(db).exists():
        print(f"ERROR: DB not found at {db}"); sys.exit(1)

    mode = "APPLY" if args.apply else ("DRY-RUN" if args.dry_run else "READ-ONLY")
    print(f"PCLTM Memory Governor — DB: {db} — Mode: {mode}")

    # Load + classify
    records = load_records(db)
    classified = [classify_record(r) for r in records]
    print(f"Loaded {len(records)} approved records")

    # Rule-based conflicts
    conflicts = detect_conflicts(classified)

    # Semantic conflicts (only with --semantic)
    if args.semantic:
        print("Running LLM semantic conflict detection...")
        sem_conflicts = detect_semantic_conflicts(classified)
        conflicts.extend(sem_conflicts)
        print(f"  Found {len(sem_conflicts)} semantic conflict(s)")

    # Scoring (always)
    scored = score_records(classified, db)

    # Merges (always compute, only apply with --merge flag)
    merges = suggest_merges(classified)
    if merges:
        print(f"Found {len(merges)} merge candidate(s)")

    # Pending + policy
    pending_issues = check_pending_pollution(db)
    policy_issues = check_inject_policy_consistency(classified)

    # STATE cross-validation
    state_issues = cross_validate_state(classified)

    # Report
    report = generate_report(classified, conflicts, pending_issues, policy_issues,
                             scored=scored, merges=merges, state_issues=state_issues)
    print(report)

    # Apply
    if args.apply or args.dry_run:
        result = apply_fixes(db, conflicts, pending_issues, merges=merges, dry_run=args.dry_run)
        print(f"\n{result}")
        # Always persist scores when applying (even if no conflicts/merges)
        if args.apply and scored:
            n = persist_scores(db, scored)
            print(f"[SCORES] Persisted governor scores for {n} record(s)")
    else:
        has_issues = any([conflicts, pending_issues, policy_issues, merges, state_issues])
        if not has_issues:
            print(f"\n[READ-ONLY] Memory is clean. No actions needed.")


if __name__ == "__main__":
    main()
