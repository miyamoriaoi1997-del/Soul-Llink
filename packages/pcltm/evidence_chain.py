"""Canonical hashing helpers for immutable PCLTM evidence events."""

from __future__ import annotations

import hashlib
import json
from typing import Any


def sha256_text(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def normalize_source_created_at(value: Any) -> str | None:
    """Canonicalize optional source timestamps for TEXT storage and comparison."""
    if value is None or value == "":
        return None
    if type(value) not in {str, int, float}:
        return str(value)
    return str(value)


def canonical_event_envelope(values: dict[str, Any]) -> str:
    return json.dumps(values, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def chain_hash(
    *,
    previous_chain_hash: str | None,
    event_id: int,
    session_id: str,
    conversation_id: str,
    platform: str,
    role: str,
    source: str,
    payload_sha256: str,
    recorded_at: str,
    schema_version: int,
    external_event_id: str | None = None,
    source_revision: int = 1,
    source_created_at: str | None = None,
    turn_id: str | None = None,
    parent_event_id: int | None = None,
    sensitivity: str = "normal",
    category: str = "unknown",
    subcategory: str = "unknown",
    visibility: str = "retrieve_only",
    source_hash: str | None = None,
) -> str:
    envelope = canonical_event_envelope(
        {
            "category": category,
            "conversation_id": conversation_id,
            "event_id": event_id,
            "external_event_id": external_event_id,
            "parent_event_id": parent_event_id,
            "payload_sha256": payload_sha256,
            "platform": platform,
            "recorded_at": recorded_at,
            "role": role,
            "schema_version": schema_version,
            "sensitivity": sensitivity,
            "session_id": session_id,
            "source": source,
            "source_created_at": source_created_at,
            "source_hash": source_hash,
            "source_revision": source_revision,
            "subcategory": subcategory,
            "turn_id": turn_id,
            "visibility": visibility,
        }
    )
    return sha256_text(f"{previous_chain_hash or ''}\n{envelope}")
