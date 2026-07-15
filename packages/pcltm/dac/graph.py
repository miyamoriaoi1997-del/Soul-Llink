from __future__ import annotations

from typing import Any

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


class DACGraph:
    """Read-only DAG helper for PCLTM-DAC summary node lineage."""

    def __init__(self, store: DACStore):
        self.store = store

    def describe_subtree(self, node_id: int) -> dict[str, Any]:
        node = self.store.get_node(node_id)
        if node is None:
            raise KeyError(f"DAC node not found: {node_id}")
        seen: set[int] = set()
        return self._describe_node(node, seen=seen, root_id=node.node_id)

    def _describe_node(
        self,
        node: DACSummaryNode,
        *,
        seen: set[int],
        root_id: int,
    ) -> dict[str, Any]:
        if node.node_id in seen:
            return {
                "node": _node_to_dict(node),
                "children": [],
                "descendant_node_ids": [],
                "cycle": True,
            }
        seen.add(node.node_id)

        child_descriptions: list[dict[str, Any]] = []
        descendant_ids: list[int] = []
        if node.source_type == "nodes":
            for child_id in node.source_ids:
                child = self.store.get_node(child_id)
                if child is None:
                    continue
                child_descriptions.append(self._describe_node(child, seen=seen, root_id=root_id))
                if child.node_id != root_id:
                    descendant_ids.append(child.node_id)
                    descendant_ids.extend(child_descriptions[-1].get("descendant_node_ids", []))

        return {
            "node": _node_to_dict(node),
            "children": child_descriptions,
            "descendant_node_ids": descendant_ids,
        }
