from __future__ import annotations

import hashlib
from typing import Any

from .recall import DACRecall
from .store import DACStore


class DACMemoryCandidateLinker:
    """Create pending long-term memory candidates from DAC recall evidence.

    DAC remains a short-term/recall layer. This linker only writes pending
    memory_records with DAC evidence metadata; it never approves records and
    never mutates USER/MEMORY prompt views directly.
    """

    def __init__(self, event_store: Any):
        self.event_store = event_store
        self.dac = DACStore(event_store)

    def propose_from_query(
        self,
        query: str,
        *,
        session_id: str | None,
        kind: str,
        target_file: str,
        content: str,
        tags: list[str] | None = None,
        importance: float | None = None,
        limit: int = 5,
    ) -> dict[str, Any]:
        recall = DACRecall(self.dac).expand_query(query, session_id=session_id, limit=limit)
        evidence = self._evidence_from_recall(recall)
        if not evidence:
            return {"ok": False, "error": "no_dac_evidence", "query": query, "session_id": session_id}

        source_node_ids = []
        for item in evidence:
            node_id = item["node_id"]
            if node_id not in source_node_ids:
                source_node_ids.append(node_id)

        metadata: dict[str, Any] = {
            "source": "dac_recall",
            "query": query,
            "session_id": session_id,
            "evidence": evidence,
            "requires_human_confirmation": True,
        }
        if tags is not None:
            metadata["tags"] = tags
        if importance is not None:
            metadata["importance"] = float(importance)

        candidate_id = self._candidate_id(kind, target_file, content, source_node_ids)
        record_id, created = self.event_store.add_memory_record(
            candidate_id=candidate_id,
            kind=kind,
            target_file=target_file,
            content=content,
            confidence=0.7,
            sensitivity=self._max_sensitivity(evidence),
            source_event_ids=[],
            source_node_ids=source_node_ids,
            status="pending",
            reviewer=None,
            decision_reason="Proposed from DAC recall evidence; pending human review.",
            metadata=metadata,
        )
        record = self._get_memory_record(record_id)
        return {"ok": True, "created": created, "record": record, "evidence": evidence}

    @staticmethod
    def _candidate_id(kind: str, target_file: str, content: str, source_node_ids: list[int]) -> str:
        material = f"dac|{kind}|{target_file}|{content}|{','.join(map(str, source_node_ids))}"
        return "dac:" + hashlib.sha256(material.encode("utf-8")).hexdigest()[:20]

    @staticmethod
    def _evidence_from_recall(recall: dict[str, Any]) -> list[dict[str, Any]]:
        evidence: list[dict[str, Any]] = []
        for result in recall.get("results", []):
            node = result.get("node") or {}
            if not node.get("node_id"):
                continue
            evidence.append(
                {
                    "node_id": node["node_id"],
                    "session_id": node.get("session_id"),
                    "node_type": node.get("node_type"),
                    "summary": node.get("summary"),
                    "sensitivity": node.get("sensitivity", "normal"),
                    "expand_hint": node.get("expand_hint", ""),
                }
            )
        return evidence

    @staticmethod
    def _max_sensitivity(evidence: list[dict[str, Any]]) -> str:
        rank = {"normal": 0, "private": 1, "restricted": 2, "secret": 3}
        return max((item.get("sensitivity") or "normal" for item in evidence), key=lambda value: rank.get(value, 0))

    def _get_memory_record(self, record_id: int) -> dict[str, Any]:
        for record in self.event_store.list_memory_records():
            if record["record_id"] == record_id:
                return record
        raise KeyError(f"memory record not found after insert: {record_id}")
