from __future__ import annotations

from typing import Any

from .store import DACRawMessage, DACStore, DACSummaryNode


def _estimate_tokens(text: str) -> int:
    return max(1, len(text) // 4)


def _node_item(node: DACSummaryNode) -> dict[str, Any]:
    return {
        "type": "summary_node",
        "node_id": node.node_id,
        "node_type": node.node_type,
        "depth": node.depth,
        "summary": node.summary,
        "source_type": node.source_type,
        "source_ids": node.source_ids,
        "persona_mode": node.persona_mode,
        "estimated_tokens": _estimate_tokens(node.summary),
    }


def _raw_item(message: DACRawMessage) -> dict[str, Any]:
    return {
        "type": "raw_message",
        "raw_id": message.raw_id,
        "role": message.role,
        "content": message.content,
        "persona_mode": message.persona_mode,
        "sequence": message.sequence,
        "estimated_tokens": message.token_count or _estimate_tokens(message.content),
    }


class DACAssembler:
    """Builds DAC context snapshots for active assembly and audit modes."""

    def __init__(self, store: DACStore):
        self.store = store

    def build_audit_snapshot(
        self,
        *,
        session_id: str,
        budget_tokens: int,
        fresh_tail_limit: int = 16,
    ) -> dict[str, Any]:
        return self.build_snapshot(
            session_id=session_id,
            budget_tokens=budget_tokens,
            fresh_tail_limit=fresh_tail_limit,
            mode="audit",
        )

    def build_snapshot(
        self,
        *,
        session_id: str,
        budget_tokens: int,
        fresh_tail_limit: int = 16,
        mode: str = "active",
    ) -> dict[str, Any]:
        budget_tokens = max(1, int(budget_tokens))
        items: list[dict[str, Any]] = []
        used_nodes: list[int] = []
        estimated = 0

        fresh_tail_messages = self.store.fresh_tail(session_id=session_id, limit=fresh_tail_limit)
        raw_items = [_raw_item(message) for message in fresh_tail_messages]
        raw_budget = sum(item["estimated_tokens"] for item in raw_items)

        remaining_for_nodes = max(0, budget_tokens - raw_budget)
        candidate_nodes = [
            node for node in self.store.all_nodes()
            if node.session_id == session_id and node.status == "active"
        ]
        candidate_nodes.sort(key=lambda node: (node.depth, node.node_id), reverse=True)
        for node in candidate_nodes:
            item = _node_item(node)
            if estimated + item["estimated_tokens"] <= remaining_for_nodes:
                items.append(item)
                used_nodes.append(node.node_id)
                estimated += item["estimated_tokens"]

        for item in raw_items:
            if estimated + item["estimated_tokens"] <= budget_tokens:
                items.append(item)
                estimated += item["estimated_tokens"]

        snapshot_id = self.store.add_context_snapshot(
            session_id=session_id,
            mode=mode,
            budget_tokens=budget_tokens,
            selected_node_ids=used_nodes,
            selected_raw_ids=[item["raw_id"] for item in raw_items if "raw_id" in item],
            fresh_tail_count=len(raw_items),
            metadata={
                "fresh_tail_limit": fresh_tail_limit,
                "used_nodes": used_nodes,
                "estimated_tokens": estimated,
                "context_items": items,
            },
        )
        return {
            "snapshot_id": snapshot_id,
            "mode": mode,
            "session_id": session_id,
            "budget_tokens": budget_tokens,
            "estimated_tokens": estimated,
            "used_nodes": used_nodes,
            "fresh_tail": raw_items,
            "context_items": items,
        }
