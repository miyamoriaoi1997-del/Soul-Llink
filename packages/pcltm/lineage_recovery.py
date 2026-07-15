"""Read-only evidence-chain recovery for legacy governed memories."""
from __future__ import annotations

import hashlib
import json
import re
import sqlite3
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable


def parse_legacy_time(value: Any) -> datetime | None:
    text = str(value or "").strip()
    if not text:
        return None
    try:
        epoch = float(text)
        if not 0.0 <= epoch <= 253402300799.0:
            return None
        return datetime.fromtimestamp(epoch, tz=timezone.utc)
    except (ValueError, OverflowError, OSError):
        pass
    try:
        parsed = datetime.fromisoformat(text.replace("Z", "+00:00"))
        return parsed.astimezone(timezone.utc) if parsed.tzinfo else parsed.replace(tzinfo=timezone.utc)
    except ValueError:
        return None


def _normalize(value: Any) -> str:
    return re.sub(r"\s+", "", str(value or "")).strip().casefold()


def _grams(value: Any, width: int = 2) -> set[str]:
    text = _normalize(value)
    if not text:
        return set()
    if len(text) <= width:
        return {text}
    return {text[index:index + width] for index in range(len(text) - width + 1)}


def _coverage(candidate: Any, evidence: Any) -> float:
    candidate_grams = _grams(candidate)
    evidence_grams = _grams(evidence)
    if not candidate_grams or not evidence_grams:
        return 0.0
    return len(candidate_grams.intersection(evidence_grams)) / len(candidate_grams)


def _sha256(value: Any) -> str:
    return hashlib.sha256(str(value or "").encode("utf-8")).hexdigest()


def _json_ids(value: Any) -> list[int]:
    try:
        parsed = json.loads(value or "[]")
    except (TypeError, json.JSONDecodeError):
        return []
    if not isinstance(parsed, list):
        return []
    return [item for item in parsed if type(item) is int]


class LineageRecovery:
    """Recover strong user→short-term→summary evidence without writing source DB."""

    def __init__(self, database: str | Path, *, minimum_coverage: float = 0.42) -> None:
        self.database = Path(database)
        self.minimum_coverage = minimum_coverage

    def corroborate(
        self,
        *,
        limit: int = 30,
        minimum_score: float = 0.60,
        minimum_margin: float = 0.10,
        record_ids: Iterable[int] | None = None,
    ) -> dict[str, Any]:
        """Find strong, unambiguous user-message corroboration without claiming lineage."""
        if type(limit) is not int or not 1 <= limit <= 30:
            raise ValueError("limit must be an integer from 1 to 30")
        if not 0.0 <= minimum_score <= 1.0 or not 0.0 <= minimum_margin <= 1.0:
            raise ValueError("score and margin must be between 0 and 1")
        allowed_ids = None if record_ids is None else {item for item in record_ids if type(item) is int}
        connection = sqlite3.connect(self.database.resolve().as_uri() + "?mode=ro", uri=True)
        connection.row_factory = sqlite3.Row
        try:
            memories = connection.execute(
                """
                SELECT record_id, content, created_at
                FROM memory_records
                WHERE status='approved' AND sensitivity='normal'
                ORDER BY record_id
                """
            ).fetchall()
            raw_rows = connection.execute(
                """
                SELECT raw_id, session_id, role, content, created_at
                FROM dac_raw_messages
                WHERE lower(role)='user'
                ORDER BY raw_id
                """
            ).fetchall()
            corroborated: list[dict[str, Any]] = []
            rejected = Counter()
            for memory in memories:
                record_id = int(memory["record_id"])
                if allowed_ids is not None and record_id not in allowed_ids:
                    continue
                scored = sorted(
                    (
                        (_coverage(memory["content"], row["content"]), int(row["raw_id"]), row)
                        for row in raw_rows
                    ),
                    key=lambda item: (-item[0], item[1]),
                )
                best_score, _best_id, best_row = scored[0] if scored else (0.0, -1, None)
                second_score = scored[1][0] if len(scored) > 1 else 0.0
                margin = best_score - second_score
                if best_row is None or best_score < minimum_score:
                    rejected["score_below_threshold"] += 1
                    continue
                if margin < minimum_margin:
                    rejected["ambiguous_user_evidence"] += 1
                    continue
                corroborated.append({
                    "record_id": record_id,
                    "memory_sha256": _sha256(memory["content"]),
                    "raw_id": int(best_row["raw_id"]),
                    "evidence_sha256": _sha256(best_row["content"]),
                    "session_sha256": _sha256(best_row["session_id"]),
                    "score": round(best_score, 6),
                    "margin": round(margin, 6),
                    "memory_created_at": memory["created_at"],
                    "evidence_created_at": best_row["created_at"],
                    "evidence_level": "user_corroborated",
                    "status": "pending_human_review",
                })
            corroborated.sort(
                key=lambda row: (-float(row["score"]), -float(row["margin"]), int(row["record_id"]))
            )
            selected = corroborated[:limit]
            return {
                "corroborated_count": len(selected),
                "corroborated_total_before_limit": len(corroborated),
                "corroborated": selected,
                "rejected_reason_counts": dict(rejected),
                "scope": "read_only_user_corroboration_report",
            }
        finally:
            connection.close()

    def recover(
        self,
        *,
        limit: int = 30,
        record_ids: Iterable[int] | None = None,
    ) -> dict[str, Any]:
        if type(limit) is not int or not 1 <= limit <= 30:
            raise ValueError("limit must be an integer from 1 to 30")
        allowed_ids = None if record_ids is None else {item for item in record_ids if type(item) is int}
        connection = sqlite3.connect(self.database.resolve().as_uri() + "?mode=ro", uri=True)
        connection.row_factory = sqlite3.Row
        try:
            memories = connection.execute(
                """
                SELECT record_id, content, created_at
                FROM memory_records
                WHERE status='approved' AND sensitivity='normal'
                ORDER BY record_id
                """
            ).fetchall()
            raw_rows = connection.execute(
                "SELECT raw_id, session_id, role, content, created_at FROM dac_raw_messages ORDER BY raw_id"
            ).fetchall()
            short_rows = connection.execute(
                """
                SELECT short_event_id, session_id, role, source, content, created_at
                FROM short_term_events ORDER BY short_event_id
                """
            ).fetchall()
            summary_rows = connection.execute(
                """
                SELECT node_id, session_id, summary, source_type, source_ids, created_at
                FROM dac_summary_nodes ORDER BY node_id
                """
            ).fetchall()
            short_by_session: dict[str, list[sqlite3.Row]] = defaultdict(list)
            summaries_by_session: dict[str, list[sqlite3.Row]] = defaultdict(list)
            for row in short_rows:
                short_by_session[str(row["session_id"])].append(row)
            for row in summary_rows:
                summaries_by_session[str(row["session_id"])].append(row)

            eligible: list[dict[str, Any]] = []
            rejected = Counter()
            for memory in memories:
                record_id = int(memory["record_id"])
                if allowed_ids is not None and record_id not in allowed_ids:
                    continue
                candidate = memory["content"]
                user_matches = [
                    (row, _coverage(candidate, row["content"]))
                    for row in raw_rows
                    if str(row["role"]).casefold() == "user"
                    and _coverage(candidate, row["content"]) >= self.minimum_coverage
                ]
                if not user_matches:
                    rejected["no_user_evidence"] += 1
                    continue
                user_matches.sort(key=lambda item: (-item[1], int(item[0]["raw_id"])))
                raw, raw_score = user_matches[0]
                raw_session_id = str(raw["session_id"])
                matching_short = [
                    row for row in short_rows
                    if str(row["role"]).casefold() == "user"
                    and _coverage(raw["content"], row["content"]) >= 0.90
                    and _coverage(candidate, row["content"]) >= self.minimum_coverage
                ]
                if not matching_short:
                    rejected["no_short_term_corroboration"] += 1
                    continue
                short_ids = {int(row["short_event_id"]) for row in matching_short}
                short_session_ids = {str(row["session_id"]) for row in matching_short}
                matching_summaries = []
                for short_session_id in short_session_ids:
                    for row in summaries_by_session.get(short_session_id, []):
                        if str(row["source_type"]) != "short_term_events":
                            continue
                        source_ids = set(_json_ids(row["source_ids"]))
                        if not source_ids.intersection(short_ids):
                            continue
                        if _coverage(candidate, row["summary"]) < self.minimum_coverage:
                            continue
                        matching_summaries.append(row)
                if not matching_summaries:
                    rejected["no_summary_lineage"] += 1
                    continue
                matching_summaries.sort(key=lambda row: int(row["node_id"]))
                selected_short = sorted(matching_short, key=lambda row: int(row["short_event_id"]))
                evidence_hash_input = "\n".join(
                    [str(raw["content"])]
                    + [str(row["content"]) for row in selected_short]
                    + [str(row["summary"]) for row in matching_summaries]
                )
                eligible.append({
                    "record_id": record_id,
                    "memory_sha256": _sha256(candidate),
                    "evidence_sha256": _sha256(evidence_hash_input),
                    "raw_ids": [int(raw["raw_id"])],
                    "short_event_ids": [int(row["short_event_id"]) for row in selected_short],
                    "summary_node_ids": [int(row["node_id"]) for row in matching_summaries],
                    "session_sha256": _sha256(raw_session_id + "\n" + "\n".join(sorted(short_session_ids))),
                    "roles": {"user": 1},
                    "coverage": round(raw_score, 6),
                    "memory_created_at": memory["created_at"],
                    "evidence_created_at": raw["created_at"],
                    "status": "pending_human_review",
                })
            eligible.sort(key=lambda row: (-float(row["coverage"]), int(row["record_id"])))
            selected = eligible[:limit]
            return {
                "eligible_count": len(selected),
                "eligible_total_before_limit": len(eligible),
                "eligible": selected,
                "rejected_reason_counts": dict(rejected),
                "scope": "read_only_evidence_report",
            }
        finally:
            connection.close()


__all__ = ["LineageRecovery", "parse_legacy_time"]
