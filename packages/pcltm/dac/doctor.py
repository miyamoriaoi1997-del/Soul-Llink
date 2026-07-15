from __future__ import annotations

import json
from typing import Any

from .store import DACStore


class DACDoctor:
    """Structural checks for PCLTM DAC tables."""

    def __init__(self, store: DACStore):
        self.store = store
        self.conn = store.conn

    def run_checks(self) -> dict[str, Any]:
        issues: list[str] = []
        issues.extend(self._check_schema())
        issues.extend(self._check_nodes())
        issues.extend(self._check_snapshots())
        issues.extend(self._check_cycles())
        return {"ok": not issues, "issues": issues}

    def _check_schema(self) -> list[str]:
        issues: list[str] = []
        version = self.store.event_store.schema_version()
        if version < 8:
            issues.append(f"schema version below DAC minimum: {version}")
        tables = {
            row[0]
            for row in self.conn.execute(
                "SELECT name FROM sqlite_master WHERE type IN ('table','virtual table')"
            ).fetchall()
        }
        for name in ("dac_summary_nodes", "dac_context_snapshots", "dac_summary_nodes_fts"):
            if name not in tables:
                issues.append(f"missing DAC table: {name}")
        return issues

    def _check_nodes(self) -> list[str]:
        issues: list[str] = []
        nodes = {node.node_id: node for node in self.store.all_nodes()}
        for node in nodes.values():
            if not node.summary.strip():
                issues.append(f"node {node.node_id} has empty summary")
            if node.source_type == "nodes":
                for child_id in node.source_ids:
                    if child_id not in nodes:
                        issues.append(f"node {node.node_id} missing child node {child_id}")
            try:
                json.dumps(node.metadata)
            except (TypeError, ValueError):
                issues.append(f"node {node.node_id} metadata is not JSON serializable")
        return issues

    def _check_snapshots(self) -> list[str]:
        issues: list[str] = []
        rows = self.conn.execute("SELECT * FROM dac_context_snapshots").fetchall()
        for row in rows:
            for column in ("selected_node_ids", "selected_raw_ids"):
                try:
                    value = json.loads(row[column])
                except json.JSONDecodeError:
                    issues.append(f"snapshot {row['snapshot_id']} {column} is invalid JSON")
                    continue
                if not isinstance(value, list):
                    issues.append(f"snapshot {row['snapshot_id']} {column} is not a list")
            try:
                json.loads(row["metadata"])
            except json.JSONDecodeError:
                issues.append(f"snapshot {row['snapshot_id']} metadata is invalid JSON")
        return issues

    def _check_cycles(self) -> list[str]:
        nodes = {node.node_id: node for node in self.store.all_nodes()}
        graph = {
            node_id: [child_id for child_id in node.source_ids if child_id in nodes]
            if node.source_type == "nodes"
            else []
            for node_id, node in nodes.items()
        }
        visiting: set[int] = set()
        visited: set[int] = set()

        def visit(node_id: int, path: list[int]) -> list[str]:
            if node_id in visiting:
                return [f"cycle detected: {' -> '.join(map(str, path + [node_id]))}"]
            if node_id in visited:
                return []
            visiting.add(node_id)
            issues: list[str] = []
            for child_id in graph.get(node_id, []):
                issues.extend(visit(child_id, path + [node_id]))
            visiting.remove(node_id)
            visited.add(node_id)
            return issues

        issues: list[str] = []
        for node_id in graph:
            issues.extend(visit(node_id, []))
        return issues
