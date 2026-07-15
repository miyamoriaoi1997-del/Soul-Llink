from pathlib import Path

from soul_link import SoulLink
from pcltm import MemoryGovernanceOrchestrator


class FakeEmotionStateManager:
    def __init__(self, *args, **kwargs):
        pass

    def get_current_emotion_state(self):
        return {"emotion_score": 1.5, "current_emotion": 1.5, "mode": "work"}

    def get_tone_modifiers(self):
        return "test modifier"


class FakeEventStore:
    def __init__(self):
        self._records = [
            {
                "record_id": 1,
                "candidate_id": "cand-1",
                "kind": "reflection",
                "target_file": "MEMORY.md",
                "content": "candidate content",
                "status": "pending",
                "metadata": {"source": "reflection"},
            }
        ]
        self.review_calls = []

    def list_memory_records(self, *, status=None, target_file=None):
        rows = list(self._records)
        if status is not None:
            rows = [row for row in rows if row.get("status") == status]
        if target_file is not None:
            rows = [row for row in rows if row.get("target_file") == target_file]
        return rows

    def review_candidate(self, record_id, *, decision, reviewer, decision_reason):
        self.review_calls.append((record_id, decision, reviewer, decision_reason))
        return {"record_id": record_id, "status": decision}


def test_soullink_exports_governance_surface(monkeypatch):
    monkeypatch.setattr(
        "persona_engine.emotion_state_manager.EmotionStateManager",
        FakeEmotionStateManager,
    )
    link = SoulLink(base_dir=(Path.home() / "soul-link" / "packages" / "persona_engine"))

    controller = link.governance(event_store=FakeEventStore())

    assert isinstance(controller, MemoryGovernanceOrchestrator)


def test_soullink_governance_default_memfs_root_is_soullink_var(monkeypatch):
    from soul_link.contracts import SOUL_LINK_ROOT

    monkeypatch.setattr(
        "persona_engine.emotion_state_manager.EmotionStateManager",
        FakeEmotionStateManager,
    )
    link = SoulLink(base_dir=(Path.home() / "soul-link" / "packages" / "persona_engine"))

    controller = link.governance(event_store=FakeEventStore())

    assert controller.memfs_store.root == SOUL_LINK_ROOT / "var" / "memfs"


def test_soullink_governance_is_importable_from_package(monkeypatch):
    from soul_link import MemoryGovernanceOrchestrator as ExportedOrchestrator

    assert ExportedOrchestrator is MemoryGovernanceOrchestrator
