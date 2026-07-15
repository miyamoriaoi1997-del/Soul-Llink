from dataclasses import is_dataclass
from pathlib import Path

from soul_link import (
    SoulLink,
    SoulLinkGovernanceSnapshot,
    SoulLinkMemoryLifecycleLedgerSnapshot,
    SoulLinkSnapshot,
)


class FakeEmotionStateManager:
    update_calls = []

    def __init__(self, *args, **kwargs):
        pass

    def apply_time_decay_if_needed(self):
        raise AssertionError("compose_active_prompt should update emotion from the current turn")

    def update_emotion_state(self, messages):
        type(self).update_calls.append(messages)
        return True

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


def test_soullink_snapshot_returns_structured_object(monkeypatch, tmp_path):
    monkeypatch.setattr(
        "persona_engine.emotion_state_manager.EmotionStateManager",
        FakeEmotionStateManager,
    )
    FakeEmotionStateManager.update_calls = []
    link = SoulLink(base_dir=Path("/example/soul-link/packages/persona_engine"), state_path=tmp_path / "STATE.md")

    snapshot = link.snapshot(
        host_system_prompt="Host Prompt",
        user_message="继续做收口",
        event_store=FakeEventStore(),
    )

    assert FakeEmotionStateManager.update_calls == [[{"role": "user", "content": "继续做收口"}]]
    assert isinstance(snapshot, SoulLinkSnapshot)
    assert isinstance(snapshot.governance_report, SoulLinkGovernanceSnapshot)
    assert is_dataclass(snapshot.governance_report)
    assert snapshot.schema_version == "v1"
    assert snapshot.runtime_target == str(Path("/example/soul-link/packages/persona_engine"))
    assert snapshot.source == "soul_link.snapshot"
    assert snapshot.trace_id.startswith("snap_")
    assert snapshot.warnings == []
    assert snapshot.generated_at
    assert snapshot.governance_report.dry_run is True
    assert snapshot.governance_report.pending_candidates == 1
    assert isinstance(snapshot.governance_report.lifecycle_ledger, SoulLinkMemoryLifecycleLedgerSnapshot)
    assert snapshot.governance_report.lifecycle_ledger.schema_version == 1
    assert snapshot.governance_report.lifecycle_ledger.authority_boundary == "read_only_governance_control_plane"
    assert snapshot.governance_report.lifecycle_ledger.transition_count == len(snapshot.governance_report.actions)
    assert snapshot.governance_report.lifecycle_ledger.risk_level == "medium"
    transition = snapshot.governance_report.lifecycle_ledger.transitions[0]
    assert transition.transition_id.startswith("mlt_")
    assert transition.object_id == "cand-1"
    assert transition.object_kind == "memory_candidate"
    assert transition.current_state == "pending"
    assert transition.proposed_state == "review_required"
    assert transition.trigger == "review_memory_candidate"
    assert transition.authority_boundary == "candidate_memory"
    assert transition.risk_level == "medium"
    assert transition.evidence_refs[0] == {"type": "memory_record", "id": 1}
    assert transition.evidence_refs[1] == {"type": "candidate", "id": "cand-1"}
    assert snapshot.governance_report.risk_level == "medium"
    action = snapshot.governance_report.actions[0]
    assert action.kind == "review_memory_candidate"
    assert action.action_id.startswith("gov_")
    assert action.risk_level == "medium"
    assert action.authority_boundary == "candidate_memory"
    assert action.metadata["candidate_id"] == "cand-1"
    assert snapshot.governance_report.candidate_scores
    assert snapshot.governance_report.top_candidates
    score = snapshot.governance_report.candidate_scores[0]
    assert snapshot.governance_report.top_candidates[0] == score
    assert score.candidate_id == "cand-1"
    assert score.action_id == action.action_id
    assert score.priority in {"medium", "high"}
    assert 0.0 <= score.score <= 1.0
    assert score.persona_relevance >= 0.7
    assert score.stale_after == "unknown"

    serializable = snapshot.to_dict()
    assert serializable["schema_version"] == "v1"
    assert serializable["runtime_target"] == str(Path("/example/soul-link/packages/persona_engine"))
    assert serializable["source"] == "soul_link.snapshot"
    assert serializable["trace_id"] == snapshot.trace_id
    assert serializable["warnings"] == []
    assert serializable["governance_report"]["dry_run"] is True
    assert serializable["governance_report"]["pending_candidates"] == 1
    serialized_ledger = serializable["governance_report"]["lifecycle_ledger"]
    assert serialized_ledger["schema_version"] == 1
    assert serialized_ledger["authority_boundary"] == "read_only_governance_control_plane"
    assert serialized_ledger["transition_count"] == len(serializable["governance_report"]["actions"])
    assert serialized_ledger["risk_level"] == "medium"
    assert serialized_ledger["blocked_count"] == 0
    serialized_transition = serialized_ledger["transitions"][0]
    assert serialized_transition["transition_id"] == transition.transition_id
    assert serialized_transition["object_id"] == "cand-1"
    assert serialized_transition["object_kind"] == "memory_candidate"
    assert serialized_transition["current_state"] == "pending"
    assert serialized_transition["proposed_state"] == "review_required"
    assert serialized_transition["authority_boundary"] == "candidate_memory"
    assert serialized_transition["evidence_refs"][0] == {"type": "memory_record", "id": 1}
    assert serializable["governance_report"]["risk_level"] == "medium"
    assert serializable["governance_report"]["candidate_scores"]
    assert serializable["governance_report"]["top_candidates"]
    serialized_score = serializable["governance_report"]["candidate_scores"][0]
    assert serializable["governance_report"]["top_candidates"][0] == serialized_score
    assert serialized_score["candidate_id"] == "cand-1"
    assert serialized_score["action_id"] == action.action_id
    assert serialized_score["priority"] in {"medium", "high"}
    assert 0.0 <= serialized_score["score"] <= 1.0
    assert serialized_score["persona_relevance"] >= 0.7
    assert serialized_score["stale_after"] == "unknown"
    serialized_action = serializable["governance_report"]["actions"][0]
    assert serialized_action["kind"] == "review_memory_candidate"
    assert serialized_action["action_id"] == action.action_id
    assert serialized_action["risk_level"] == "medium"
    assert serialized_action["authority_boundary"] == "candidate_memory"
    assert serialized_action["metadata"]["candidate_id"] == "cand-1"
