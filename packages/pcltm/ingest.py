"""Production-safe ingest adapter for PCLTM."""

from __future__ import annotations

import hashlib
import json
from typing import Any


_KIND_SOURCE = {
    "chat_message": "chat",
    "assistant_message": "chat",
    "tool_result": "tool",
    "tool_call": "tool",
    "cron_output": "cron",
    "system_observation": "system",
}


class PCLTMIngestAdapter:
    """Normalize external transcript-like payloads into PCLTM raw events."""

    def __init__(self, store: Any):
        self.store = store

    def ingest(self, payload: dict[str, Any]) -> dict[str, Any]:
        external_id = payload["external_id"]
        kind = payload.get("kind", "chat_message")
        source = payload.get("source") or _KIND_SOURCE.get(kind, "chat")
        role = payload.get("role") or self._role_for_kind(kind)
        attachments = self._sanitize_attachments(payload.get("attachments") or [])
        source_hash = self._source_hash(payload)
        event_id, status = self.store.upsert_external_event(
            external_id=external_id,
            source_hash=source_hash,
            kind=kind,
            attachments=attachments,
            payload_metadata={
                "created_at": payload.get("created_at"),
                "ingest_mode": "pcltm",
            },
            session_id=payload["session_id"],
            conversation_id=payload["conversation_id"],
            platform=payload.get("platform", "unknown"),
            role=role,
            source=source,
            content=payload.get("content", ""),
            persona_mode=payload.get("persona_mode"),
            route_bucket=payload.get("route_bucket"),
            model_hint=payload.get("model_hint"),
            sensitivity=payload.get("sensitivity", self._attachment_sensitivity(attachments)),
            category="raw_conversation",
            subcategory=role,
            inject_policy="retrieve_only",
            classifier_version="pcltm-ingest-v2",
        )
        ingest = self.store.find_ingest_event(external_id)
        if ingest is None:
            raise RuntimeError(f"atomic ingest did not publish external event: {external_id}")
        return {
            "created": status == "inserted",
            "updated": status == "updated",
            "event_id": event_id,
            "ingest_id": ingest["ingest_id"],
            "source_hash": source_hash,
        }

    @staticmethod
    def _role_for_kind(kind: str) -> str:
        if kind == "assistant_message":
            return "assistant"
        if kind in {"tool_result", "tool_call", "cron_output"}:
            return "tool"
        if kind == "system_observation":
            return "system"
        return "user"

    @staticmethod
    def _sanitize_attachments(attachments: list[dict[str, Any]]) -> list[dict[str, Any]]:
        allowed = {"media_type", "file_id", "path", "caption", "sensitivity", "size", "mime_type"}
        clean = []
        for attachment in attachments:
            clean.append({key: value for key, value in attachment.items() if key in allowed})
        return clean

    @staticmethod
    def _attachment_sensitivity(attachments: list[dict[str, Any]]) -> str:
        rank = {"normal": 0, "private": 1, "restricted": 2, "secret": 3}
        value = "normal"
        for attachment in attachments:
            sensitivity = attachment.get("sensitivity", "normal")
            if rank.get(sensitivity, 0) > rank[value]:
                value = sensitivity
        return value

    @staticmethod
    def _source_hash(payload: dict[str, Any]) -> str:
        normalized = json.dumps(payload, sort_keys=True, ensure_ascii=False, default=str)
        return hashlib.sha256(normalized.encode("utf-8")).hexdigest()
