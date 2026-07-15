from __future__ import annotations

from typing import Any

from .doctor import DACDoctor
from .store import DACStore, DACSummaryNode


def _node_to_dict(node: DACSummaryNode) -> dict[str, Any]:
    return {
        "node_id": node.node_id,
        "session_id": node.session_id,
        "node_type": node.node_type,
        "depth": node.depth,
        "summary": node.summary,
        "source_type": node.source_type,
        "source_ids": node.source_ids,
        "persona_mode": node.persona_mode,
        "inject_policy": node.inject_policy,
        "sensitivity": node.sensitivity,
        "metadata": node.metadata,
        "created_at": node.created_at,
        "expand_hint": node.expand_hint,
        "status": node.status,
    }


class DACRecall:
    """Read-only recall facade for PCLTM DAC nodes."""

    def __init__(self, store: DACStore):
        self.store = store
        self.conn = store.conn

    def status(self) -> dict[str, Any]:
        doctor = DACDoctor(self.store).run_checks()
        nodes = self.conn.execute("SELECT count(*) FROM dac_summary_nodes").fetchone()[0]
        snapshots = self.conn.execute("SELECT count(*) FROM dac_context_snapshots").fetchone()[0]
        return {
            "doctor_ok": doctor["ok"],
            "doctor_issues": doctor["issues"],
            "nodes": int(nodes),
            "snapshots": int(snapshots),
        }

    def grep(self, query: str, *, session_id: str | None = None, limit: int = 20) -> list[dict[str, Any]]:
        return [_node_to_dict(node) for node in self.store.search_nodes(query, session_id=session_id, limit=limit)]

    def describe(self, node_id: int) -> dict[str, Any]:
        node = self.store.get_node(node_id)
        if node is None:
            raise KeyError(f"DAC node not found: {node_id}")
        return _node_to_dict(node)

    def expand(self, node_id: int, *, recursive: bool = True) -> dict[str, Any]:
        expanded = self.store.expand_node(node_id, recursive=recursive)
        node = expanded["node"]
        result: dict[str, Any] = {
            "node": _node_to_dict(node),
            "children": [_node_to_dict(child) for child in expanded.get("children", [])],
            "short_term_events": [],
        }
        if node.source_type == "short_term_events" and node.source_ids:
            placeholders = ",".join("?" for _ in node.source_ids)
            rows = self.conn.execute(
                f"SELECT * FROM short_term_events WHERE short_event_id IN ({placeholders}) ORDER BY short_event_id",
                tuple(node.source_ids),
            ).fetchall()
            result["short_term_events"] = [dict(row) for row in rows]
        if recursive and expanded.get("expanded_children"):
            result["expanded_children"] = [
                self.expand(child["node"].node_id, recursive=True)
                for child in expanded["expanded_children"]
            ]
        return result

    def expand_query(
        self,
        query: str,
        *,
        session_id: str | None = None,
        limit: int = 5,
        recursive: bool = True,
    ) -> dict[str, Any]:
        """Search DAC summary nodes and expand the matching lineage.

        This is the PCLTM-DAC equivalent of LCM's search-then-expand recall:
        keep the foreground prompt small, then recover exact source lineage only
        when a query needs it.
        """
        hits = self.store.search_nodes(query, session_id=session_id, limit=limit)
        expanded = [self.expand(node.node_id, recursive=recursive) for node in hits]
        return {
            "query": query,
            "session_id": session_id,
            "count": len(expanded),
            "results": expanded,
        }
