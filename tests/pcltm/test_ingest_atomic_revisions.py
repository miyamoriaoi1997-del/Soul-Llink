from __future__ import annotations

from pathlib import Path

import pytest

import pcltm.projections.runtime as projection_runtime
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


def test_ingest_adapter_converges_transcript_projections_before_return(tmp_path: Path) -> None:
    store = EventStore(tmp_path / "pcltm.db")
    adapter = PCLTMIngestAdapter(store)
    try:
        result = adapter.ingest(_payload("public projection convergence"))
        chunks = store._conn.execute(
            "SELECT chunk_text FROM event_chunks WHERE event_id=?", (result["event_id"],)
        ).fetchall()
        statuses = store._conn.execute(
            "SELECT projection_kind, status FROM projection_outbox WHERE event_seq=? ORDER BY projection_kind",
            (result["event_id"],),
        ).fetchall()
    finally:
        store.close()

    assert [row["chunk_text"] for row in chunks] == ["public projection convergence"]
    assert [(row["projection_kind"], row["status"]) for row in statuses] == [
        ("transcript_chunks", "applied"),
        ("transcript_fts", "applied"),
    ]


def test_duplicate_ingest_fails_while_fts_projection_waits_for_retry(tmp_path: Path, monkeypatch) -> None:
    store = EventStore(tmp_path / "pcltm.db")
    adapter = PCLTMIngestAdapter(store)
    real_apply = projection_runtime._apply_fts_job

    def fail_fts(*args, **kwargs):
        raise RuntimeError("forced FTS failure")

    try:
        monkeypatch.setattr(projection_runtime, "_apply_fts_job", fail_fts)
        with pytest.raises(RuntimeError, match="forced FTS failure"):
            adapter.ingest(_payload("public FTS retry"))
        monkeypatch.setattr(projection_runtime, "_apply_fts_job", real_apply)

        with pytest.raises(RuntimeError, match="projections are not converged"):
            adapter.ingest(_payload("public FTS retry"))
    finally:
        store.close()
