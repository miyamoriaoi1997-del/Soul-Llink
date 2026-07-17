from __future__ import annotations

from pathlib import Path

from pcltm.ingest import PCLTMIngestAdapter
from pcltm.store import EventStore


def _payload(content: str) -> dict[str, object]:
    return {
        "external_id": "gateway-message:42",
        "session_id": "session-1",
        "conversation_id": "conversation-1",
        "platform": "desktop",
        "kind": "chat_message",
        "role": "user",
        "content": content,
        "created_at": "2026-07-17T02:00:00Z",
    }


def test_ingest_adapter_atomically_persists_external_revision(tmp_path: Path) -> None:
    store = EventStore(tmp_path / "pcltm.db")
    adapter = PCLTMIngestAdapter(store)
    try:
        first = adapter.ingest(_payload("第一版原文"))
        second = adapter.ingest(_payload("第二版原文"))
        current = store.find_ingest_event("gateway-message:42")
        events = store.list_events(limit=10)
        counts = store.fts_counts()
    finally:
        store.close()

    assert first["created"] is True
    assert second["created"] is False
    assert second["updated"] is True
    assert current is not None
    assert current["event_id"] == second["event_id"]
    assert first["event_id"] != second["event_id"]
    assert {event["content"] for event in events} == {"第一版原文", "第二版原文"}
    assert counts["events"] == counts["event_fts"] == 2


def test_ingest_adapter_keeps_noise_as_retrieve_only_permanent_evidence(tmp_path: Path) -> None:
    store = EventStore(tmp_path / "pcltm.db")
    adapter = PCLTMIngestAdapter(store)
    payload = _payload("ok")
    payload["external_id"] = "gateway-message:noise"
    try:
        result = adapter.ingest(payload)
        event = store.get_event(result["event_id"])
        short_term = store.list_short_term_events()
    finally:
        store.close()

    assert event["content"] == "ok"
    assert event["category"] == "raw_conversation"
    assert event["inject_policy"] == "retrieve_only"
    assert event["visibility"] == "retrieve_only"
    assert short_term == []
