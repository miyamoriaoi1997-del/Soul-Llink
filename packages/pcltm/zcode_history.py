"""Idempotent ingestion of ZCode's canonical session database into PCLTM.

ZCode stores conversations in a user-scope SQLite database
(``~/.zcode/cli/db/db.sqlite``) with a ``message`` table (role/session
metadata in JSON ``data``) and a ``part`` table (the actual text pieces,
also JSON ``data``). This ingestor mirrors those turns into retrieve-only
PCLTM events, following the ``HermesHistoryIngestor`` pattern so both host
histories converge on the same governed event store.
"""
from __future__ import annotations

import hashlib
import json
import sqlite3
from pathlib import Path
from typing import Any

from .classifier import EventClassifier
from .projections.runtime import drain_transcript_projections
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


class ZCodeHistoryIngestor:
    """Mirror canonical ZCode messages and lifecycle into retrieve-only events."""

    def __init__(self, store: EventStore, zcode_db: str | Path):
        self.store = store
        self.zcode_db = Path(zcode_db)

    def ingest(self, *, session_id: str | None = None, persona_mode: str | None = None) -> dict[str, int]:
        if not self.zcode_db.is_file():
            raise FileNotFoundError(f"ZCode session database is missing: {self.zcode_db}")
        uri = self.zcode_db.resolve().as_uri() + "?mode=ro"
        conn = sqlite3.connect(uri, uri=True)
        try:
            conn.row_factory = sqlite3.Row
            message_columns = {row[1] for row in conn.execute("PRAGMA table_info(message)")}
            part_columns = {row[1] for row in conn.execute("PRAGMA table_info(part)")}
            if not {"id", "session_id", "data"}.issubset(message_columns):
                raise RuntimeError("unsupported ZCode message schema")
            if not {"message_id", "data"}.issubset(part_columns):
                raise RuntimeError("unsupported ZCode part schema")
            parts = [dict(row) for row in self._parts(conn, session_id=session_id)]
            scanned = inserted = updated = existing = 0
            for message in self._messages(conn, session_id=session_id):
                scanned += 1
                status = self._ingest_message(dict(message), parts, persona_mode=persona_mode)
                inserted += status == "inserted"
                updated += status == "updated"
                existing += status == "existing"
        finally:
            conn.close()
        drain_transcript_projections(self.store)
        return {"scanned": scanned, "inserted": inserted, "updated": updated, "existing": existing}

    @staticmethod
    def _messages(conn: sqlite3.Connection, *, session_id: str | None):
        sql, params = "SELECT * FROM message", ()
        if session_id:
            sql, params = sql + " WHERE session_id = ?", (session_id,)
        return conn.execute(sql + " ORDER BY id ASC", params)

    @staticmethod
    def _parts(conn: sqlite3.Connection, *, session_id: str | None):
        sql, params = "SELECT * FROM part", ()
        if session_id:
            sql, params = sql + " WHERE session_id = ?", (session_id,)
        for row in conn.execute(sql + " ORDER BY id ASC", params):
            item = dict(row)
            data = _json_value(item.get("data")) or {}
            item["type"] = str(data.get("type") or "")
            item["text"] = str(data.get("text") or "")
            yield item

    def _event(self, *, external_id: str, kind: str, metadata: dict[str, Any], session_id: str, platform: str, role: str, content: str, sensitivity: str, subcategory: str, persona_mode: str | None = None) -> str:
        source_hash = _sha256(json.dumps({"external_id": external_id, "session_id": session_id, "role": role, "content": content, "metadata": metadata}, ensure_ascii=False, sort_keys=True))
        classification = EventClassifier().classify(
            role=role,
            source="chat" if role == "user" else "zcode_db",
            content=content,
            persona_mode=persona_mode,
            sensitivity=sensitivity,
        )
        _, status = self.store.upsert_external_event(
            external_id=external_id, source_hash=source_hash, kind=kind, payload_metadata=metadata,
            session_id=session_id, conversation_id=session_id, platform=platform, role=role,
            source="zcode_db", content=content, persona_mode=persona_mode, route_bucket=None,
            model_hint=None, sensitivity=classification.sensitivity if sensitivity == "normal" else sensitivity,
            category=classification.category, subcategory=subcategory, inject_policy=classification.inject_policy,
            classification_confidence=classification.confidence,
            classifier_version=classification.classifier_version,
        )
        return status

    def _ingest_message(self, message: dict[str, Any], parts: list[dict[str, Any]], *, persona_mode: str | None = None) -> str:
        data = _json_value(message.get("data")) or {}
        message_id = str(message["id"])
        session_id = str(message.get("session_id") or "")
        role = str(data.get("role") or "unknown")
        content = "".join(
            str(part["text"])
            for part in parts
            if str(part["message_id"]) == message_id and part.get("type") == "text" and part.get("text")
        )
        raw_content = content
        content = _prompt_reference(role, raw_content) if role in _PROMPT_ROLES else raw_content
        metadata = {
            "zcode_message_id": message_id,
            "timestamp": data.get("time"),
            "parent_id": data.get("parentID"),
            "model_id": data.get("modelID") or data.get("model"),
            "provider_id": data.get("providerID"),
            "mode": data.get("mode"),
            "agent": data.get("agent"),
            "path": data.get("path"),
            "cost": data.get("cost"),
            "tokens": data.get("tokens"),
            "finish": data.get("finish"),
            "sequence": message.get("sequence"),
        }
        return self._event(
            external_id=f"zcode-message:{message_id}", kind="zcode_message", metadata=metadata,
            session_id=session_id, platform="zcode", role=role, content=content,
            sensitivity="restricted" if role in _PROMPT_ROLES else "normal",
            subcategory=role, persona_mode=persona_mode,
        )


def main(argv: list[str] | None = None) -> int:
    import argparse

    from .runtime_paths import resolve_db_path

    parser = argparse.ArgumentParser(prog="soullink-zcode-history")
    parser.add_argument("--zcode-db", required=True, help="path to ZCode db.sqlite")
    parser.add_argument("--db", default=None, help="PCLTM db path (default: resolved runtime path)")
    parser.add_argument("--session-id", default=None)
    parser.add_argument("--persona-mode", default=None)
    args = parser.parse_args(argv)
    store = EventStore(resolve_db_path(args.db))
    try:
        result = ZCodeHistoryIngestor(store, args.zcode_db).ingest(
            session_id=args.session_id, persona_mode=args.persona_mode
        )
    finally:
        store.close()
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0


__all__ = ["ZCodeHistoryIngestor"]
