"""Offline JSONL transcript importer for PCLTM."""

from __future__ import annotations

import json
from collections import Counter
from pathlib import Path
from typing import Any

from .doctor import PersonaLCMDoctor
from .ingest import PCLTMIngestAdapter


class JSONLTranscriptImporter:
    """Import transcript-like JSONL payloads through the PCLTM ingest adapter."""

    def __init__(self, store: Any):
        self.store = store
        self.adapter = PCLTMIngestAdapter(store)

    def import_file(self, path: str | Path) -> dict[str, Any]:
        path = Path(path)
        created = 0
        updated = 0
        skipped_duplicate = 0
        dropped = 0
        by_kind: Counter[str] = Counter()
        by_category: Counter[str] = Counter()
        by_subcategory: Counter[str] = Counter()
        errors: list[dict[str, Any]] = []

        with path.open("r", encoding="utf-8") as handle:
            for line_number, line in enumerate(handle, start=1):
                stripped = line.strip()
                if not stripped:
                    continue
                try:
                    payload = json.loads(stripped)
                    result = self.adapter.ingest(payload)
                    by_kind[payload.get("kind", "chat_message")] += 1
                    if result.get("dropped"):
                        dropped += 1
                        continue
                    if result["created"]:
                        created += 1
                    elif result.get("updated"):
                        updated += 1
                    else:
                        skipped_duplicate += 1
                    event = self.store.get_event(result["event_id"])
                    by_category[event["category"]] += 1
                    by_subcategory[event["subcategory"]] += 1
                except Exception as exc:  # pragma: no cover - exercised by future malformed fixtures
                    errors.append({"line": line_number, "error": str(exc)})

        doctor = PersonaLCMDoctor(self.store).run_checks()
        return {
            "ok": not errors and doctor["ok"],
            "path": str(path),
            "created": created,
            "updated": updated,
            "skipped_duplicate": skipped_duplicate,
            "dropped": dropped,
            "by_kind": dict(by_kind),
            "by_category": dict(by_category),
            "by_subcategory": dict(by_subcategory),
            "errors": errors,
            "doctor_ok": doctor["ok"],
            "doctor_issues": doctor["issues"],
        }
