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
