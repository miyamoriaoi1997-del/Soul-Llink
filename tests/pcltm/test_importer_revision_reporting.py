from __future__ import annotations

import json
from pathlib import Path

from pcltm.importer import JSONLTranscriptImporter
from pcltm.store import EventStore


def test_jsonl_import_reports_created_updated_and_duplicate_separately(tmp_path: Path) -> None:
    transcript = tmp_path / "events.jsonl"
    first = {
        "external_id": "message:1", "session_id": "s", "conversation_id": "c",
        "kind": "chat_message", "role": "user", "content": "v1",
    }
    transcript.write_text(json.dumps(first, ensure_ascii=False) + "\n", encoding="utf-8")
    store = EventStore(tmp_path / "pcltm.db")
    try:
        created = JSONLTranscriptImporter(store).import_file(transcript)
        second = {**first, "content": "v2"}
        transcript.write_text(json.dumps(second, ensure_ascii=False) + "\n", encoding="utf-8")
        updated = JSONLTranscriptImporter(store).import_file(transcript)
        duplicate = JSONLTranscriptImporter(store).import_file(transcript)
    finally:
        store.close()

    assert (created["created"], created["updated"], created["skipped_duplicate"]) == (1, 0, 0)
    assert (updated["created"], updated["updated"], updated["skipped_duplicate"]) == (0, 1, 0)
    assert (duplicate["created"], duplicate["updated"], duplicate["skipped_duplicate"]) == (0, 0, 1)


def test_jsonl_import_converges_transcript_projections_before_return(tmp_path: Path) -> None:
    transcript = tmp_path / "events.jsonl"
    payload = {
        "external_id": "message:projection", "session_id": "s", "conversation_id": "c",
        "kind": "chat_message", "role": "user", "content": "public importer convergence",
    }
    transcript.write_text(json.dumps(payload) + "\n", encoding="utf-8")
    store = EventStore(tmp_path / "pcltm.db")
    try:
        report = JSONLTranscriptImporter(store).import_file(transcript)
        event_id = store.find_ingest_event("message:projection")["event_id"]
        chunks = store._conn.execute(
            "SELECT chunk_text FROM event_chunks WHERE event_id=?", (event_id,)
        ).fetchall()
    finally:
        store.close()

    assert report["ok"] is True
    assert [row["chunk_text"] for row in chunks] == ["public importer convergence"]
