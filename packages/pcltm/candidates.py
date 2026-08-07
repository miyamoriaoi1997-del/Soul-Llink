"""Evidence-backed persona memory candidate extraction."""

from __future__ import annotations

import hashlib
import math
import re
from typing import Any

from .classifier import parse_memory_command, parse_stable_memory_assertion
from .memory_contracts import AuthorityRef

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
            order="desc",
        )
        candidates = []
        for event in events:
            candidate = self._candidate_from_event(event)
            if candidate is not None:
                candidates.append(candidate)
        # Select the newest bounded window, then fold commands in authority
        # order so a later replace/forget deterministically wins.
        candidates.reverse()
        return candidates

    def _candidate_from_event(self, event: dict[str, Any]) -> dict[str, Any] | None:
        # Hermes canonical history is ingested with source="hermes_state_db";
        # both it and direct "chat" ingestion are real user conversation.
        if event.get("role") != "user" or event.get("source") not in ("chat", "hermes_state_db"):
            return None
        if event.get("inject_policy") != "candidate_only":
            return None
        raw_content = str(event.get("content") or "")
        command = parse_memory_command(raw_content)
        stable_assertion = parse_stable_memory_assertion(raw_content)
        if command is None and stable_assertion is None:
            return None
        mode = event.get("category") or "unknown"
        if mode not in _TARGET_BY_MODE:
            return None
        if stable_assertion is not None:
            kind = stable_assertion.kind
            target_file = stable_assertion.target_file
            action = "memory"
            semantic_key = stable_assertion.semantic_key
            content = stable_assertion.content
            rationale = f"stable natural-language assertion: {kind}"
        else:
            kind, target_file = _TARGET_BY_MODE[mode]
            action, semantic_key, content = command
            rationale = f"explicit durable-memory command in mode: {mode}"
        confidence = event.get("classification_confidence")
        if isinstance(confidence, bool) or type(confidence) not in {int, float}:
            return None
        confidence = float(confidence)
        if not math.isfinite(confidence) or not 0.0 <= confidence <= 1.0:
            return None
        return self._build_candidate(
            event,
            kind=kind,
            target_file=target_file,
            content=content,
            confidence=confidence,
            mode=mode,
            rationale=rationale,
            identity_action=action,
            semantic_key=semantic_key,
        )

    def _build_candidate(
        self,
        event: dict[str, Any],
        *,
        kind: str,
        target_file: str,
        content: str,
        confidence: float,
        mode: str,
        rationale: str,
        identity_action: str,
        semantic_key: str | None,
    ) -> dict[str, Any]:
        payload_sha256 = event.get("payload_sha256")
        if type(payload_sha256) is not str or not re.fullmatch(r"[0-9a-f]{64}", payload_sha256):
            raise ValueError("event payload commitment is missing or malformed")
        source_event_ids = [event["event_id"]]
        source_refs = (AuthorityRef(
            "event", str(event["event_id"]), int(event.get("source_revision") or 1),
            payload_sha256,
        ),)
        candidate_id = self._candidate_id(kind, content, source_event_ids)
        return {
            "candidate_id": candidate_id,
            "kind": kind,
            "target_file": target_file,
            "content": content,
            "confidence": confidence,
            "mode": mode,
            "sensitivity": event.get("sensitivity") or "normal",
            "source_event_ids": source_event_ids,
            "source_refs": source_refs,
            "source_node_ids": [],
            "requires_human_confirmation": False,
            "memory_worthiness": "high",
            "identity_action": identity_action,
            "semantic_key": semantic_key,
            "canonical_key": (
                f"persona:{target_file}:{kind}:semantic:{semantic_key}"
                if semantic_key else None
            ),
            "rationale": rationale,
        }

    @staticmethod
    def _candidate_id(kind: str, content: str, source_event_ids: list[int]) -> str:
        material = f"{kind}|{content}|{','.join(map(str, source_event_ids))}"
        return hashlib.sha256(material.encode("utf-8")).hexdigest()[:16]
