"""Evidence-backed persona memory candidate extraction."""

from __future__ import annotations

import hashlib
import math
import os
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

NATURAL_ASSERTION_BASE_CONFIDENCE = 0.70
INDEPENDENT_SESSION_EVIDENCE_BONUS = 0.16


def _bounded_confidence_env(name: str, default: float) -> float:
    try:
        value = float(os.getenv(name, default))
    except (TypeError, ValueError):
        return default
    return value if math.isfinite(value) and 0.0 <= value <= 1.0 else default


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
            evidence = self._supporting_events(stable_assertion)
            if not evidence:
                return None
            source_event_ids = [int(item["event_id"]) for item in evidence]
            if any(
                type(item.get("payload_sha256")) is not str
                or re.fullmatch(r"[0-9a-f]{64}", str(item.get("payload_sha256"))) is None
                for item in evidence
            ):
                raise ValueError("event payload commitment is missing or malformed")
            source_refs = tuple(
                AuthorityRef(
                    "event", str(item["event_id"]), int(item.get("source_revision") or 1),
                    str(item["payload_sha256"]),
                )
                for item in evidence
            )
            independent_session_count = len({str(item["session_id"]) for item in evidence})
            supporting_contents = {str(item.get("content") or "").strip() for item in evidence}
            admission_tier = stable_assertion.admission_tier
            if (
                admission_tier == "pending_review"
                and independent_session_count >= 2
                and len(supporting_contents) >= 2
            ):
                admission_tier = "auto_activate"
            requires_human_confirmation = admission_tier != "auto_activate"
            configured_base = _bounded_confidence_env(
                "SOULLINK_PCLTM_NATURAL_ASSERTION_BASE_CONFIDENCE",
                NATURAL_ASSERTION_BASE_CONFIDENCE,
            )
            confidence = max(configured_base, stable_assertion.admission_confidence)
            if admission_tier == "auto_activate":
                confidence = max(confidence, 0.85)
            elif independent_session_count >= 2:
                confidence += _bounded_confidence_env(
                    "SOULLINK_PCLTM_INDEPENDENT_EVIDENCE_BONUS",
                    INDEPENDENT_SESSION_EVIDENCE_BONUS,
                )
        else:
            kind, target_file = _TARGET_BY_MODE[mode]
            action, semantic_key, content = command
            requires_human_confirmation = False
            rationale = f"explicit durable-memory command in mode: {mode}"
            source_event_ids = [int(event["event_id"])]
            if (
                type(event.get("payload_sha256")) is not str
                or re.fullmatch(r"[0-9a-f]{64}", str(event.get("payload_sha256"))) is None
            ):
                raise ValueError("event payload commitment is missing or malformed")
            source_refs = (AuthorityRef(
                "event", str(event["event_id"]), int(event.get("source_revision") or 1),
                str(event["payload_sha256"]),
            ),)
            independent_session_count = 1
            confidence = event.get("classification_confidence")
            admission_tier = "auto_activate"
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
            source_event_ids=source_event_ids,
            source_refs=source_refs,
            evidence_count=len(source_event_ids),
            independent_session_count=independent_session_count,
            requires_human_confirmation=requires_human_confirmation,
            admission_tier=admission_tier,
            reason_codes=(
                stable_assertion.reason_codes
                if stable_assertion is not None else ("explicit_memory_command",)
            ),
            lexical_score=(stable_assertion.lexical_score if stable_assertion is not None else 1.0),
            semantic_score=(stable_assertion.semantic_score if stable_assertion is not None else 1.0),
            stability_score=(stable_assertion.stability_score if stable_assertion is not None else 1.0),
            future_value_score=(stable_assertion.future_value_score if stable_assertion is not None else 1.0),
        )

    def _supporting_events(self, assertion) -> list[dict[str, Any]]:
        """Collect exact semantic support while keeping sessions independent."""
        supporting: list[dict[str, Any]] = []
        # list_events is bounded; take the newest window, then restore authority
        # order so candidate IDs and the projection anchor remain deterministic.
        for item in reversed(self.store.list_events(limit=500, order="desc")):
            if item.get("role") != "user" or item.get("source") not in ("chat", "hermes_state_db"):
                continue
            parsed = parse_stable_memory_assertion(str(item.get("content") or ""))
            if parsed is None:
                continue
            if (
                parsed.kind == assertion.kind
                and parsed.target_file == assertion.target_file
                and parsed.semantic_key == assertion.semantic_key
            ):
                supporting.append(item)
        return supporting

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
        source_event_ids: list[int],
        source_refs: tuple[AuthorityRef, ...],
        evidence_count: int,
        independent_session_count: int,
        requires_human_confirmation: bool,
        admission_tier: str,
        reason_codes: tuple[str, ...],
        lexical_score: float,
        semantic_score: float,
        stability_score: float,
        future_value_score: float,
    ) -> dict[str, Any]:
        payload_sha256 = event.get("payload_sha256")
        if type(payload_sha256) is not str or not re.fullmatch(r"[0-9a-f]{64}", payload_sha256):
            raise ValueError("event payload commitment is missing or malformed")
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
            "requires_human_confirmation": requires_human_confirmation,
            "admission_tier": admission_tier,
            "admission_reason_codes": list(reason_codes),
            "memory_worthiness": "high" if confidence >= 0.85 else "medium",
            "evidence_count": evidence_count,
            "independent_session_count": independent_session_count,
            "lexical_score": lexical_score,
            "semantic_score": semantic_score,
            "stability_score": stability_score,
            "future_value_score": future_value_score,
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
