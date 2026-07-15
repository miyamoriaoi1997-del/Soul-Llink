"""Evidence-backed persona memory candidate extraction."""

from __future__ import annotations

import hashlib
from typing import Any


_TARGET_BY_MODE = {
    "work": ("system_convention", "MEMORY.md"),
    "system_maintenance": ("system_convention", "MEMORY.md"),
    "daily": ("user_preference", "USER.md"),
    "intimacy": ("user_preference", "USER.md"),
    "conflict": ("user_preference", "USER.md"),
    "repair": ("user_preference", "USER.md"),
    "sex_candidate": ("user_preference", "USER.md"),
    "sex": ("user_preference", "USER.md"),
}


class PersonaCandidateExtractor:
    """Deterministic candidate extractor that never writes persona files."""

    def __init__(self, store: Any):
        self.store = store

    def extract(self, *, scope: dict[str, Any], limit: int = 100) -> list[dict[str, Any]]:
        events = self.store.list_events(
            session_id=scope.get("session_id"),
            conversation_id=scope.get("conversation_id"),
            platform=scope.get("platform"),
            persona_mode=scope.get("persona_mode"),
            source=scope.get("source"),
            limit=limit,
        )
        candidates = []
        for event in events:
            candidate = self._candidate_from_event(event)
            if candidate is not None:
                candidates.append(candidate)
        return candidates

    def _candidate_from_event(self, event: dict[str, Any]) -> dict[str, Any] | None:
        if event.get("role") != "user" or event.get("source") != "chat":
            return None
        if event.get("inject_policy") != "candidate_only":
            return None
        mode = event.get("category") or "unknown"
        if mode not in _TARGET_BY_MODE:
            return None
        kind, target_file = _TARGET_BY_MODE[mode]
        content = event.get("content") or ""
        if not content.strip():
            return None
        confidence = float(event.get("classification_confidence") or 0.0)
        return self._build_candidate(
            event,
            kind=kind,
            target_file=target_file,
            content=content,
            confidence=confidence,
            rationale=f"state-machine mode: {mode}",
        )

    def _build_candidate(
        self,
        event: dict[str, Any],
        *,
        kind: str,
        target_file: str,
        content: str,
        confidence: float,
        rationale: str,
    ) -> dict[str, Any]:
        source_event_ids = [event["event_id"]]
        candidate_id = self._candidate_id(kind, content, source_event_ids)
        return {
            "candidate_id": candidate_id,
            "kind": kind,
            "target_file": target_file,
            "content": content,
            "confidence": confidence,
            "sensitivity": event.get("sensitivity") or "normal",
            "source_event_ids": source_event_ids,
            "source_node_ids": [],
            "requires_human_confirmation": True,
            "rationale": rationale,
        }

    @staticmethod
    def _candidate_id(kind: str, content: str, source_event_ids: list[int]) -> str:
        material = f"{kind}|{content}|{','.join(map(str, source_event_ids))}"
        return hashlib.sha256(material.encode("utf-8")).hexdigest()[:16]
