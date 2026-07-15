"""Idempotent ingestion of Hermes' canonical session database into PCLTM."""
from __future__ import annotations
import hashlib
import json
import sqlite3
from pathlib import Path
from typing import Any
from .store import EventStore

_PROMPT_ROLES = {"system", "developer"}

def _sha256(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()

def _json_value(value: Any) -> Any:
    if not isinstance(value, str) or not value:
        return None
    try:
        return json.loads(value)
    except (TypeError, ValueError):
        return value

def _prompt_reference(role: str, content: str) -> str:
    return json.dumps({"storage": "hash_only", "role": role, "sha256": _sha256(content), "chars": len(content)}, ensure_ascii=False, sort_keys=True)

class HermesHistoryIngestor:
    """Mirror canonical Hermes messages and lifecycle into retrieve-only events."""
    def __init__(self, store: EventStore, hermes_db: str | Path):
        self.store = store
        self.hermes_db = Path(hermes_db)

    def ingest(self, *, session_id: str | None = None) -> dict[str, int]:
        if not self.hermes_db.is_file():
            raise FileNotFoundError(f"Hermes session database is missing: {self.hermes_db}")
        uri = self.hermes_db.resolve().as_uri() + "?mode=ro"
        with sqlite3.connect(uri, uri=True) as conn:
            conn.row_factory = sqlite3.Row
            session_columns = {row[1] for row in conn.execute("PRAGMA table_info(sessions)")}
            message_columns = {row[1] for row in conn.execute("PRAGMA table_info(messages)")}
            if not {"id", "source"}.issubset(session_columns):
                raise RuntimeError("unsupported Hermes sessions schema")
            if not {"id", "session_id", "role", "content", "timestamp"}.issubset(message_columns):
                raise RuntimeError("unsupported Hermes messages schema")
            sessions = self._sessions(conn, session_id=session_id)
            scanned = inserted = updated = existing = 0
            for session in sessions.values():
                status = self._ingest_session(session)
                inserted += status == "inserted"
                updated += status == "updated"
                existing += status == "existing"
            for row in self._messages(conn, session_id=session_id):
                scanned += 1
                session = sessions.get(str(row["session_id"]), {})
                status = self._ingest_message(dict(row), session)
                inserted += status == "inserted"
                updated += status == "updated"
                existing += status == "existing"
        return {"scanned": scanned, "inserted": inserted, "updated": updated, "existing": existing, "sessions": len(sessions)}

    @staticmethod
    def _sessions(conn: sqlite3.Connection, *, session_id: str | None) -> dict[str, dict[str, Any]]:
        sql, params = "SELECT * FROM sessions", ()
        if session_id:
            sql, params = sql + " WHERE id = ?", (session_id,)
        return {str(row["id"]): dict(row) for row in conn.execute(sql, params).fetchall()}

    @staticmethod
    def _messages(conn: sqlite3.Connection, *, session_id: str | None):
        sql, params = "SELECT * FROM messages", ()
        if session_id:
            sql, params = sql + " WHERE session_id = ?", (session_id,)
        return conn.execute(sql + " ORDER BY id ASC", params)

    def _event(self, *, external_id: str, kind: str, metadata: dict[str, Any], session_id: str, platform: str, role: str, content: str, sensitivity: str, subcategory: str) -> str:
        source_hash = _sha256(json.dumps({"external_id": external_id, "session_id": session_id, "role": role, "content": content, "metadata": metadata}, ensure_ascii=False, sort_keys=True))
        _, status = self.store.upsert_external_event(
            external_id=external_id, source_hash=source_hash, kind=kind, payload_metadata=metadata,
            session_id=session_id, conversation_id=session_id, platform=platform, role=role,
            source="hermes_state_db", content=content, persona_mode=None, route_bucket=None,
            model_hint=None, sensitivity=sensitivity, category="raw_conversation",
            subcategory=subcategory, inject_policy="retrieve_only", classification_confidence=None,
            classifier_version="hermes-history-v1",
        )
        return status

    def _ingest_session(self, session: dict[str, Any]) -> str:
        session_id = str(session["id"])
        metadata = {
            "hermes_session_id": session_id, "source": session.get("source"),
            "parent_session_id": session.get("parent_session_id"), "started_at": session.get("started_at"),
            "ended_at": session.get("ended_at"), "end_reason": session.get("end_reason"),
            "archived": bool(session.get("archived", 0)), "rewind_count": int(session.get("rewind_count") or 0),
        }
        prompt = str(session.get("system_prompt") or "")
        if prompt:
            metadata["system_prompt"] = {"storage": "hash_only", "sha256": _sha256(prompt), "chars": len(prompt)}
        content = json.dumps({"session_id": session_id, "state": "ended" if session.get("ended_at") else "active"}, ensure_ascii=False, sort_keys=True)
        return self._event(external_id=f"hermes-session:{session_id}", kind="hermes_session", metadata=metadata, session_id=session_id, platform=str(session.get("source") or "unknown"), role="lifecycle", content=content, sensitivity="restricted", subcategory="session_lifecycle")

    def _ingest_message(self, message: dict[str, Any], session: dict[str, Any]) -> str:
        role = str(message.get("role") or "unknown")
        raw_content = str(message.get("content") or "")
        content = _prompt_reference(role, raw_content) if role in _PROMPT_ROLES else raw_content
        metadata = self._metadata(message, session)
        return self._event(external_id=f"hermes-message:{int(message['id'])}", kind="hermes_message", metadata=metadata, session_id=str(message.get("session_id") or ""), platform=str(session.get("source") or "unknown"), role=role, content=content, sensitivity="restricted" if role in _PROMPT_ROLES else "normal", subcategory=role)

    @staticmethod
    def _metadata(message: dict[str, Any], session: dict[str, Any]) -> dict[str, Any]:
        metadata = {
            "hermes_message_id": int(message["id"]), "timestamp": message.get("timestamp"),
            "active": bool(message.get("active", 1)), "compacted": bool(message.get("compacted", 0)),
            "observed": bool(message.get("observed", 0)), "token_count": message.get("token_count"),
            "finish_reason": message.get("finish_reason"), "platform_message_id": message.get("platform_message_id"),
            "tool_call_id": message.get("tool_call_id"), "tool_name": message.get("tool_name"),
            "session": {"source": session.get("source"), "parent_session_id": session.get("parent_session_id"), "started_at": session.get("started_at")},
        }
        tool_calls = _json_value(message.get("tool_calls"))
        if tool_calls is not None:
            metadata["tool_calls"] = tool_calls
        return metadata

__all__ = ["HermesHistoryIngestor"]
