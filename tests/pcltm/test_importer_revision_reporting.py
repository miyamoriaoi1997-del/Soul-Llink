from __future__ import annotations

import json
from pathlib import Path

from pcltm.projections.transcript_chunks import TranscriptChunkProjector
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
        "kind": "chat_message", "role": "user", "content": "JSONL 原文",
    }
    transcript.write_text(json.dumps(payload, ensure_ascii=False) + "\n", encoding="utf-8")
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
    assert [row["chunk_text"] for row in chunks] == ["JSONL 原文"]


def test_jsonl_duplicate_import_reports_projection_retry_as_failure(tmp_path: Path, monkeypatch) -> None:
    transcript = tmp_path / "events.jsonl"
    payload = {
        "external_id": "message:retry", "session_id": "s", "conversation_id": "c",
        "kind": "chat_message", "role": "user", "content": "等待 JSONL 重试的原文",
    }
    transcript.write_text(json.dumps(payload, ensure_ascii=False) + "\n", encoding="utf-8")
    store = EventStore(tmp_path / "pcltm.db")
    real_apply = TranscriptChunkProjector._apply

    def fail_chunk(self, job, *, now):
        raise RuntimeError("forced chunk failure")

    try:
        monkeypatch.setattr(TranscriptChunkProjector, "_apply", fail_chunk)
        first = JSONLTranscriptImporter(store).import_file(transcript)
        monkeypatch.setattr(TranscriptChunkProjector, "_apply", real_apply)
        duplicate = JSONLTranscriptImporter(store).import_file(transcript)
    finally:
        store.close()

    assert first["ok"] is False
    assert duplicate["ok"] is False
    assert any("projections are not converged" in issue["error"] for issue in duplicate["errors"])
