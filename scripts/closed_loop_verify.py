"""Closed-loop verification: conversation -> classified event -> candidate -> claim -> recall.

Simulates exactly what sync_turn does after our change:
1. Write a Hermes-style state.db with a user/assistant turn
2. HermesHistoryIngestor.ingest(persona_mode="work") -> events classified by EventClassifier
3. PersonaCandidateExtractor.extract() -> candidates (must be non-empty now)
4. CandidatePromotionService.promote() -> claims via guardrails
5. Verify: event has candidate_only + 0.95, candidate extracted, claim active, recall works
"""
from __future__ import annotations

import json
import sqlite3
import tempfile
from pathlib import Path

import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "packages"))

from pcltm.candidate_promotion import CandidatePromotionService
from pcltm.candidates import PersonaCandidateExtractor
from pcltm.hermes_history import HermesHistoryIngestor
from pcltm.memory_retrieval import GovernedMemorySearchRequest, search_governed_memories
from pcltm.memory_contracts import PersonaMode
from pcltm.runtime_paths import resolve_db_path, resolve_memfs_root  # noqa: F401  (ensure env vars)
from pcltm.store import EventStore


def build_hermes_db(path: Path) -> None:
    conn = sqlite3.connect(path)
    conn.execute("CREATE TABLE sessions (id TEXT PRIMARY KEY, source TEXT, parent_session_id TEXT, started_at TEXT, ended_at TEXT, end_reason TEXT, archived INTEGER, rewind_count INTEGER, system_prompt TEXT)")
    conn.execute("CREATE TABLE messages (id INTEGER PRIMARY KEY, session_id TEXT, role TEXT, content TEXT, timestamp TEXT, active INTEGER, compacted INTEGER, observed INTEGER, token_count INTEGER, finish_reason TEXT, platform_message_id TEXT, tool_call_id TEXT, tool_name TEXT, tool_calls TEXT)")
    conn.execute("INSERT INTO sessions VALUES ('s1','desktop',NULL,'2026-08-01T10:00:00Z',NULL,NULL,0,0,'secret system prompt')")
    conn.executemany(
        "INSERT INTO messages (id, session_id, role, content, timestamp, active) VALUES (?,?,?,?,?,1)",
        [
            (1, "s1", "user", "[memory:morning-drink] 我喜欢每天早上喝黑咖啡，不加糖。", "2026-08-01T10:00:01Z"),
            (2, "s1", "assistant", "好的，记住了。", "2026-08-01T10:00:02Z"),
        ],
    )
    conn.commit()
    conn.close()


def report_dict(report) -> dict:
    return {
        "scanned": report.scanned, "activated": report.activated, "pending": report.pending,
        "dropped": report.dropped, "superseded": report.superseded, "rejected": report.rejected,
        "failed": report.failed,
    }


def main() -> None:
    with tempfile.TemporaryDirectory() as tmp:
        tmp_path = Path(tmp)
        hermes_db = tmp_path / "state.db"
        build_hermes_db(hermes_db)

        store = EventStore(tmp_path / "pcltm.db")
        try:
            # --- 1. ingest with persona_mode="work" (same as sync_turn does) ---
            report = HermesHistoryIngestor(store, hermes_db).ingest(persona_mode="work")
            print("ingest report:", report)

            events = store.list_events(limit=20)
            user_events = [e for e in events if e["role"] == "user"]
            assert user_events, "no user events ingested"
            ue = user_events[0]
            print("user event:", {k: ue[k] for k in ("role", "source", "inject_policy", "classification_confidence", "category", "persona_mode")})
            assert ue["inject_policy"] == "candidate_only", f"expected candidate_only, got {ue['inject_policy']}"
            assert ue["classification_confidence"] == 0.95, f"expected 0.95, got {ue['classification_confidence']}"
            assert ue["category"] == "work", f"expected work, got {ue['category']}"

            # --- 2. extract candidates ---
            candidates = PersonaCandidateExtractor(store).extract(scope={"session_id": "s1"}, limit=50)
            print("candidates:", [(c["kind"], c["target_file"], c["confidence"], c["mode"]) for c in candidates])
            assert candidates, "no candidates extracted - pipeline still disconnected"
            assert candidates[0]["kind"] == "system_convention" and candidates[0]["target_file"] == "MEMORY.md"

            # --- 3. promote with guardrails (>=0.85 auto-activate) ---
            promotion = CandidatePromotionService(store).promote(candidates)
            print("promotion:", report_dict(promotion))
            assert promotion.activated == 1, f"expected 1 activated, got {promotion.activated}"

            # --- 3b. drain memory projections so recall sees the claim ---
            from pcltm.projections.memory_runtime import drain_memory_projections
            drain_result = drain_memory_projections(store, memfs_root=tmp_path / "memfs")
            print("drain:", drain_result)

            # --- 4. recall the claim ---
            from pcltm.memory_retrieval import GovernedMemorySearchRequest, search_governed_memories
            result = search_governed_memories(
                store,
                GovernedMemorySearchRequest(query="黑咖啡", persona_mode=PersonaMode.WORK, limit=5),
            )
            print("recall status:", result.status.value, "items:", len(result.items))
            found = [i for i in result.items if "黑咖啡" in i.content]
            assert found, "claim not recallable - promotion did not persist"

            # --- 5. supersede path: same candidate content again (idempotent) ---
            again = dict(candidates[0])
            again["candidate_id"] = "same-key-replay"
            promotion1b = CandidatePromotionService(store).promote([again])
            print("replay same content:", report_dict(promotion1b))
            assert promotion1b.outcomes[0].decision == "duplicate", "identical content should be idempotent"

            # --- 6. pending + dropped guardrails ---
            low = dict(candidates[0]); low["candidate_id"] = "low1"; low["content"] = "低置信内容"; low["confidence"] = 0.5
            mid = dict(candidates[0]); mid["candidate_id"] = "mid1"; mid["content"] = "中置信内容"; mid["confidence"] = 0.7
            promotion3 = CandidatePromotionService(store).promote([low, mid])
            print("promotion3:", report_dict(promotion3))
            assert promotion3.dropped == 1, "low-confidence candidate should be dropped"
            assert promotion3.pending == 1, "mid-confidence candidate should be pending"
            queue = store.list_candidate_queue(status="pending")
            print("pending queue:", [(q["content"], q["status"]) for q in queue])
        finally:
            store.close()
        print("\nCLOSED-LOOP OK")


if __name__ == "__main__":
    main()
