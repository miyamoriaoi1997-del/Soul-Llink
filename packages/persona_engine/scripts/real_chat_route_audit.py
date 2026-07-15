#!/usr/bin/env python3
"""Read-only audit for persona runtime shadow routing against Hermes session messages.

This script reads the configured persona_orchestrator.log_path, matches redacted
runtime-shadow rows to local Hermes user messages, and prints a compact summary
plus optional row/candidate records. It is intentionally read-only: it does not
write fixtures, logs, database rows, or config.

Matching strategy:
- Prefer session_id + user_message_hash.
- If repeated short messages produce multiple rows for the same session/hash,
  resolve to the nearest message timestamp and mark the row as
  session_hash_ambiguous_time_resolved instead of pretending it is exact.
- If the session does not match, fall back to hash-only matching and mark the
  result as hash_only or hash_only_ambiguous_time_resolved.

The built-in labeler is deliberately conservative. It only labels obvious
samples, so its accuracy numbers are a triage signal, not a final real-world
accuracy claim.
"""
from __future__ import annotations

import argparse
import collections
import datetime as dt
import hashlib
import json
import pathlib
import sqlite3
from typing import Any

from persona_orchestrator.lexicon import (
    CONTEXT_ACTION_TERMS,
    CONTEXT_CONTINUATION_TERMS,
    CONTEXT_QUESTION_TERMS,
    SYSTEM_DOMAIN_TERMS,
    WORK_DOMAIN_TERMS,
    combined,
)

try:
    import yaml
except ModuleNotFoundError:  # Keep this read-only audit script dependency-light.
    yaml = None  # type: ignore[assignment]

DEFAULT_CONFIG = (pathlib.Path.home() / ".hermes" / "config.yaml")
DEFAULT_STATE_DB = (pathlib.Path.home() / ".hermes" / "state.db")
DEFAULT_LOG_FALLBACK = pathlib.Path("<persona-engine-prod-root>/logs/runtime_shadow.jsonl")


def sha16(text: str) -> str:
    return hashlib.sha256((text or "").encode("utf-8")).hexdigest()[:16]


def utc_iso(ts: float | None) -> str | None:
    if ts is None:
        return None
    try:
        return dt.datetime.fromtimestamp(float(ts), tz=dt.timezone.utc).isoformat()
    except Exception:
        return None


def preview(text: str | None, limit: int) -> str | None:
    if text is None:
        return None
    normalized = " ".join(text.replace("\r", " ").replace("\n", " ").split())
    if len(normalized) <= limit:
        return normalized
    return normalized[: limit - 1] + "…"


def load_config(config_path: pathlib.Path) -> dict[str, Any]:
    if yaml is not None:
        with config_path.open("r", encoding="utf-8") as fh:
            return yaml.safe_load(fh) or {}

    # Minimal fallback for this script's only required setting:
    # persona_orchestrator.log_path. This avoids adding PyYAML as a runtime
    # dependency to the persona venv just for a read-only audit command.
    text = config_path.read_text(encoding="utf-8")
    in_persona = False
    persona: dict[str, Any] = {}
    for raw in text.splitlines():
        line = raw.rstrip()
        stripped = line.strip()
        if not stripped or stripped.startswith("#"):
            continue
        if not raw.startswith((" ", "\t")) and stripped.endswith(":"):
            in_persona = stripped[:-1] == "persona_orchestrator"
            continue
        if in_persona and raw.startswith((" ", "\t")) and ":" in stripped:
            key, value = stripped.split(":", 1)
            value = value.strip().strip('"').strip("'")
            if key == "log_path":
                persona["log_path"] = value
    return {"persona_orchestrator": persona}


def resolve_log_path(config_path: pathlib.Path) -> pathlib.Path:
    cfg = load_config(config_path)
    po = cfg.get("persona_orchestrator") or {}
    return pathlib.Path(po.get("log_path") or DEFAULT_LOG_FALLBACK)


def load_shadow_records(
    log_path: pathlib.Path,
    platform: str | None,
    limit: int | None,
    only_new_fields: bool,
) -> list[dict[str, Any]]:
    if not log_path.exists():
        return []
    records: list[dict[str, Any]] = []
    with log_path.open("r", encoding="utf-8") as fh:
        for line in fh:
            try:
                row = json.loads(line)
            except Exception:
                continue
            if platform and row.get("platform") != platform:
                continue
            if only_new_fields and row.get("message_timestamp") is None:
                continue
            records.append(row)
    return records[-limit:] if limit else records


def load_user_messages(state_db: pathlib.Path, source: str | None, max_chars: int) -> list[dict[str, Any]]:
    con = sqlite3.connect(str(state_db))
    con.row_factory = sqlite3.Row
    where = "where m.role='user'"
    params: list[Any] = []
    if source:
        where += " and s.source=?"
        params.append(source)
    rows = con.execute(
        f"""
        select m.id, m.session_id, m.content, m.timestamp, s.source, s.title
        from messages m join sessions s on m.session_id=s.id
        {where}
        order by m.timestamp asc
        """,
        params,
    ).fetchall()
    messages: list[dict[str, Any]] = []
    for row in rows:
        item = dict(row)
        content = item.get("content") or ""
        if max_chars and len(content) > max_chars:
            # Very large messages are usually compacted context or injected prompts;
            # keep the audit focused on human-turn samples.
            continue
        item["hash"] = sha16(content)
        messages.append(item)
    return messages


def expected_from_text(
    text: str | None,
    previous_mode: str | None,
    safety_flags: list[Any] | None = None,
) -> tuple[str | None, str | None, str]:
    """Conservative labeler for obvious samples only.

    Ambiguous short continuations are intentionally left unlabeled unless the
    previous mode makes the expected hold bucket obvious. Sex wording is also
    gate-aware: if the runtime recorded a restrained desire gate, the expected
    active mode/bucket remains relationship-side instead of sex.
    """
    flags = set(str(flag) for flag in (safety_flags or []))
    if text is None:
        return None, None, "unmatched/no raw session row"
    normalized = " ".join(text.strip().split())
    lower = normalized.lower()

    if normalized in {"测试角色", "测试角色测试角色", "在吗", "老婆", "老婆老婆"}:
        return "daily", "relationship", "simple relationship ping"

    if normalized in {"继续", "好继续", "好，继续", "好继续吧", "继续吧", "好听你的继续"}:
        if previous_mode == "work":
            return "work", "task", "short continuation inherits active work mode"
        if previous_mode == "sex":
            return "sex", "sex", "short continuation inherits active sex mode"
        if previous_mode == "daily":
            return "daily", None, "short continuation inherits relationship-side previous mode"
        return None, None, "short continuation requires previous-mode context"

    if previous_mode == "work" and any(
        term in normalized
        for term in combined(CONTEXT_CONTINUATION_TERMS, CONTEXT_ACTION_TERMS, CONTEXT_QUESTION_TERMS)
    ):
        return "work", "task", "contextual continuation inherits active work mode"

    system_terms = combined(SYSTEM_DOMAIN_TERMS, ["自评价", "fixture", "路由", "情绪值", "情绪数值", "emotion_score", "desire_tier"])
    work_terms = combined(WORK_DOMAIN_TERMS)
    meta_edit_terms = ["这句话", "这段", "不要", "去掉", "修改", "放开", "限制", "不直接写", "成人幻想"]
    has_system_or_work = any(term in lower for term in system_terms) or any(term in normalized for term in system_terms)
    has_work_wording = any(term in lower for term in work_terms) or any(term in normalized for term in work_terms)
    has_meta_edit = any(term in normalized for term in meta_edit_terms)
    if has_system_or_work or has_work_wording or has_meta_edit:
        return "work", "task", "system/persona maintenance wording"

    sex_terms = ["想要你", "做爱", "插入", "射", "高潮", "别停", "不要停"]
    if any(term in normalized for term in sex_terms):
        if "sex_desire_gate_restrained" in flags:
            return "daily", "relationship", "explicit sex wording blocked by restrained desire gate"
        return "sex", "sex", "explicit sex wording with no restrained gate flag"

    relationship_terms = ["我爱你", "喜欢你", "亲吻", "亲亲", "抱抱", "对不起", "是不是不在乎", "不爱我", "算了"]
    if any(term in normalized for term in relationship_terms):
        return "daily", "relationship", "obvious relationship wording"

    return None, None, "unlabeled"


def choose_nearest(candidates: list[dict[str, Any]], message_ts: float | None) -> tuple[dict[str, Any] | None, float | None]:
    if not candidates:
        return None, None
    if message_ts is None:
        return candidates[-1], None
    matched = min(candidates, key=lambda msg: abs(float(msg.get("timestamp") or 0) - float(message_ts)))
    return matched, abs(float(matched.get("timestamp") or 0) - float(message_ts))


def build_indexes(messages: list[dict[str, Any]]) -> tuple[dict[tuple[str, str], list[dict[str, Any]]], dict[str, list[dict[str, Any]]]]:
    by_session_hash: dict[tuple[str, str], list[dict[str, Any]]] = collections.defaultdict(list)
    by_hash: dict[str, list[dict[str, Any]]] = collections.defaultdict(list)
    for msg in messages:
        by_session_hash[(msg["session_id"], msg["hash"])].append(msg)
        by_hash[msg["hash"]].append(msg)
    return by_session_hash, by_hash


def classify_match(
    record: dict[str, Any],
    by_session_hash: dict[tuple[str, str], list[dict[str, Any]]],
    by_hash: dict[str, list[dict[str, Any]]],
) -> tuple[str, list[dict[str, Any]], dict[str, Any] | None, float | None]:
    h = record.get("user_message_hash")
    session_id = record.get("session_id")
    message_ts = record.get("message_timestamp")

    candidates = by_session_hash.get((session_id, h), [])
    if candidates:
        matched, delta = choose_nearest(candidates, message_ts)
        match_type = "exact_session_hash" if len(candidates) == 1 else "session_hash_ambiguous_time_resolved"
        return match_type, candidates, matched, delta

    candidates = by_hash.get(h, [])
    if candidates:
        matched, delta = choose_nearest(candidates, message_ts)
        match_type = "hash_only" if len(candidates) == 1 else "hash_only_ambiguous_time_resolved"
        return match_type, candidates, matched, delta

    return "unmatched", [], None, None


def audit(
    config_path: pathlib.Path,
    state_db: pathlib.Path,
    platform: str | None,
    limit: int | None,
    only_new_fields: bool,
    preview_chars: int,
    max_message_chars: int,
) -> dict[str, Any]:
    log_path = resolve_log_path(config_path)
    records = load_shadow_records(log_path, platform=platform, limit=limit, only_new_fields=only_new_fields)
    messages = load_user_messages(state_db, source=platform, max_chars=max_message_chars)
    by_session_hash, by_hash = build_indexes(messages)

    rows: list[dict[str, Any]] = []
    for record in records:
        match_type, candidates, matched, delta = classify_match(record, by_session_hash, by_hash)
        text = matched.get("content") if matched else None
        expected_mode, expected_bucket, label_reason = expected_from_text(
            text,
            record.get("previous_mode"),
            record.get("safety_flags"),
        )
        strict_ok = None if expected_mode is None else record.get("mode") == expected_mode
        bucket_ok = None if expected_bucket is None else record.get("route_bucket") == expected_bucket
        rows.append(
            {
                "shadow_ts": record.get("ts"),
                "message_ts": record.get("message_timestamp"),
                "message_time_utc": utc_iso(record.get("message_timestamp")),
                "session_id": record.get("session_id"),
                "platform": record.get("platform"),
                "hash": record.get("user_message_hash"),
                "match_type": match_type,
                "candidate_count": len(candidates),
                "match_delta_seconds": None if delta is None else round(delta, 3),
                "message_id": matched.get("id") if matched else None,
                "session_title": matched.get("title") if matched else None,
                "text_preview": preview(text, preview_chars),
                "previous_mode": record.get("previous_mode"),
                "mode": record.get("mode"),
                "transition": record.get("transition"),
                "confidence": record.get("confidence"),
                "selected_layers": record.get("selected_layers"),
                "safety_flags": record.get("safety_flags"),
                "route_bucket": record.get("route_bucket"),
                "model_hint": record.get("model_hint"),
                "switch_allowed": record.get("switch_allowed"),
                "switch_reason": record.get("switch_reason"),
                "expected_mode": expected_mode,
                "expected_bucket": expected_bucket,
                "label_reason": label_reason,
                "strict_ok": strict_ok,
                "bucket_ok": bucket_ok,
            }
        )

    nonambiguous = [r for r in rows if r["match_type"] in {"exact_session_hash", "hash_only"}]
    time_resolved = [r for r in rows if r["match_type"].endswith("time_resolved")]
    labeled = [r for r in rows if r["strict_ok"] is not None and r["match_type"] != "unmatched"]
    labeled_bucket = [r for r in rows if r["bucket_ok"] is not None and r["match_type"] != "unmatched"]
    possible_issues = [
        r
        for r in rows
        if (r["strict_ok"] is False or r["bucket_ok"] is False)
        or (isinstance(r.get("confidence"), (int, float)) and r["confidence"] < 0.7)
        or r["match_type"].endswith("time_resolved")
    ]

    summary = {
        "config_path": str(config_path),
        "state_db": str(state_db),
        "log_path": str(log_path),
        "platform": platform,
        "records": len(records),
        "messages_indexed": len(messages),
        "matched_nonambiguous": len(nonambiguous),
        "time_resolved_repeated_hash": len(time_resolved),
        "unmatched": sum(1 for r in rows if r["match_type"] == "unmatched"),
        "labeled_mode_rows": len(labeled),
        "labeled_bucket_rows": len(labeled_bucket),
        "strict_mode_accuracy_obvious_labels": None
        if not labeled
        else sum(bool(r["strict_ok"]) for r in labeled) / len(labeled),
        "route_bucket_accuracy_obvious_labels": None
        if not labeled_bucket
        else sum(bool(r["bucket_ok"]) for r in labeled_bucket) / len(labeled_bucket),
        "mode_distribution": dict(collections.Counter(r["mode"] for r in rows)),
        "bucket_distribution": dict(collections.Counter(str(r["route_bucket"]) for r in rows)),
        "transition_top10": collections.Counter(r["transition"] for r in rows).most_common(10),
        "low_confidence_lt_0_7": sum(
            1 for r in rows if isinstance(r.get("confidence"), (int, float)) and r["confidence"] < 0.7
        ),
        "switch_allowed_values": dict(collections.Counter(str(r["switch_allowed"]) for r in rows)),
        "possible_issue_rows": len(possible_issues),
    }
    return {"summary": summary, "rows": rows, "possible_issues": possible_issues}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=pathlib.Path, default=DEFAULT_CONFIG)
    parser.add_argument("--state-db", type=pathlib.Path, default=DEFAULT_STATE_DB)
    parser.add_argument("--platform", default="telegram", help="Platform/source filter. Use empty string for all.")
    parser.add_argument("--limit", type=int, default=120, help="Last N shadow records after platform filtering.")
    parser.add_argument("--include-old-fields", action="store_true", help="Include rows without message_timestamp.")
    parser.add_argument("--preview-chars", type=int, default=120)
    parser.add_argument("--max-message-chars", type=int, default=8000)
    parser.add_argument("--json", action="store_true", help="Print full JSON object.")
    parser.add_argument("--jsonl", action="store_true", help="Print summary then all rows as JSONL.")
    parser.add_argument("--max-issues", type=int, default=30, help="Maximum issue rows printed without --json/--jsonl.")
    args = parser.parse_args()

    platform = args.platform or None
    result = audit(
        config_path=args.config,
        state_db=args.state_db,
        platform=platform,
        limit=args.limit,
        only_new_fields=not args.include_old_fields,
        preview_chars=args.preview_chars,
        max_message_chars=args.max_message_chars,
    )

    if args.json:
        print(json.dumps(result, ensure_ascii=False, indent=2))
        return

    print("SUMMARY", json.dumps(result["summary"], ensure_ascii=False))
    if args.jsonl:
        for row in result["rows"]:
            print(json.dumps(row, ensure_ascii=False))
    elif args.max_issues > 0:
        for row in result["possible_issues"][: args.max_issues]:
            print("ISSUE", json.dumps(row, ensure_ascii=False))


if __name__ == "__main__":
    main()
