"""Read-only integrity checks for Persona LCM."""

from __future__ import annotations

from typing import Any

from .store import CURRENT_SCHEMA_VERSION
from .secret_policy import detect_secret_categories


_SENSITIVITY_RANK = {"normal": 0, "private": 1, "restricted": 2, "secret": 3}


class PersonaLCMDoctor:
    """Read-only doctor for LCM store integrity and safety policy violations."""

    def __init__(self, store: Any):
        self.store = store

    def run_checks(self) -> dict[str, Any]:
        issues: list[dict[str, Any]] = []
        schema_version = self.store.schema_version()
        if schema_version != CURRENT_SCHEMA_VERSION:
            issues.append(
                {
                    "code": "schema_version_mismatch",
                    "expected": CURRENT_SCHEMA_VERSION,
                    "actual": schema_version,
                }
            )
        issues.extend(self._check_orphan_summary_nodes())
        issues.extend(self._check_sensitivity_downgrades())
        issues.extend(self._check_unredacted_summary_secrets())
        issues.extend(self._check_unredacted_memory_record_secrets())
        issues.extend(self._check_pending_memory_record_secrets())
        issues.extend(self._check_fts_consistency())
        issues.extend(self._check_policy_violations())
        issues.extend(self._check_ingest_links())
        return {"ok": not issues, "issues": issues, "schema_version": schema_version}

    def _summary_rows(self) -> list[dict[str, Any]]:
        rows = self.store._conn.execute("SELECT * FROM summary_nodes ORDER BY node_id ASC").fetchall()
        return [dict(row) for row in rows]

    def _check_orphan_summary_nodes(self) -> list[dict[str, Any]]:
        issues = []
        for node in self._summary_rows():
            event_count = self.store._conn.execute(
                "SELECT count(*) AS n FROM summary_event_edges WHERE node_id = ?",
                (node["node_id"],),
            ).fetchone()["n"]
            child_count = self.store._conn.execute(
                "SELECT count(*) AS n FROM summary_node_edges WHERE parent_node_id = ?",
                (node["node_id"],),
            ).fetchone()["n"]
            if event_count == 0 and child_count == 0:
                issues.append({"code": "orphan_summary_node", "node_id": node["node_id"]})
        return issues

    def _check_sensitivity_downgrades(self) -> list[dict[str, Any]]:
        issues = []
        for node in self._summary_rows():
            event_ids = self.store._source_event_ids(node["node_id"])
            child_ids = self.store._source_node_ids(node["node_id"])
            if not event_ids and not child_ids:
                continue
            expected = self.store._max_source_sensitivity(event_ids, child_ids)
            if _SENSITIVITY_RANK.get(node["sensitivity"], 0) < _SENSITIVITY_RANK.get(expected, 0):
                issues.append(
                    {
                        "code": "sensitivity_downgrade",
                        "node_id": node["node_id"],
                        "expected_min_sensitivity": expected,
                        "actual_sensitivity": node["sensitivity"],
                    }
                )
        return issues

    def _check_unredacted_summary_secrets(self) -> list[dict[str, Any]]:
        issues = []
        for node in self._summary_rows():
            categories = detect_secret_categories(node["summary"] or "")
            if categories:
                issues.append({"code": "unredacted_secret_in_summary", "node_id": node["node_id"], "categories": categories})
        return issues

    def _check_unredacted_memory_record_secrets(self) -> list[dict[str, Any]]:
        issues: list[dict[str, Any]] = []
        rows = self.store._conn.execute(
            """
            SELECT record_id, target_file, sensitivity, status, content
            FROM memory_records
            WHERE status = 'approved'
            ORDER BY record_id ASC
            """
        ).fetchall()
        for row in rows:
            categories = detect_secret_categories(row["content"] or "")
            if categories:
                issues.append(
                    {
                        "code": "approved_memory_contains_secret",
                        "record_id": row["record_id"],
                        "target_file": row["target_file"],
                        "sensitivity": row["sensitivity"],
                        "status": row["status"],
                        "categories": categories,
                    }
                )
        return issues

    def _check_pending_memory_record_secrets(self) -> list[dict[str, Any]]:
        issues: list[dict[str, Any]] = []
        rows = self.store._conn.execute(
            """
            SELECT record_id, target_file, sensitivity, status, content
            FROM memory_records
            WHERE status = 'pending'
            ORDER BY record_id ASC
            """
        ).fetchall()
        for row in rows:
            categories = detect_secret_categories(row["content"] or "")
            if categories:
                issues.append(
                    {
                        "code": "pending_memory_contains_secret",
                        "record_id": row["record_id"],
                        "target_file": row["target_file"],
                        "sensitivity": row["sensitivity"],
                        "status": row["status"],
                        "categories": categories,
                    }
                )
        return issues

    def _check_fts_consistency(self) -> list[dict[str, Any]]:
        counts = self.store.fts_counts()
        issues: list[dict[str, Any]] = []
        if counts["events"] != counts["event_fts"]:
            issues.append(
                {
                    "code": "fts_event_count_mismatch",
                    "expected": counts["events"],
                    "actual": counts["event_fts"],
                }
            )
        if counts["summaries"] != counts["summary_fts"]:
            issues.append(
                {
                    "code": "fts_summary_count_mismatch",
                    "expected": counts["summaries"],
                    "actual": counts["summary_fts"],
                }
            )
        return issues

    def _check_policy_violations(self) -> list[dict[str, Any]]:
        issues: list[dict[str, Any]] = []
        rows = self.store._conn.execute(
            """
            SELECT event_id, subcategory, sensitivity, inject_policy
            FROM events
            WHERE sensitivity = 'secret'
              AND inject_policy = 'mode_relevant'
            ORDER BY event_id ASC
            """
        ).fetchall()
        for row in rows:
            issues.append(
                {
                    "code": "policy_secret_mode_relevant",
                    "event_id": row["event_id"],
                    "subcategory": row["subcategory"],
                    "sensitivity": row["sensitivity"],
                    "inject_policy": row["inject_policy"],
                }
            )
        return issues

    def _check_ingest_links(self) -> list[dict[str, Any]]:
        issues: list[dict[str, Any]] = []
        rows = self.store._conn.execute(
            """
            SELECT ingest_events.external_id, ingest_events.event_id
            FROM ingest_events
            LEFT JOIN events ON events.event_id = ingest_events.event_id
            WHERE events.event_id IS NULL
            ORDER BY ingest_events.ingest_id ASC
            """
        ).fetchall()
        for row in rows:
            issues.append(
                {
                    "code": "orphan_ingest_event",
                    "external_id": row["external_id"],
                    "event_id": row["event_id"],
                }
            )
        return issues