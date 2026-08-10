from pathlib import Path

from pcltm.candidate_promotion import CandidatePromotionService
from pcltm.candidates import PersonaCandidateExtractor
from pcltm.store import EventStore


def test_verbatim_repetition_across_sessions_stays_pending(tmp_path: Path) -> None:
    store = EventStore(tmp_path / "pcltm.db")
    try:
        for session_id in ("repeat-1", "repeat-2"):
            store.append_event(
                session_id=session_id, conversation_id=session_id, platform="test",
                role="user", source="chat", content="我喜欢深色界面。", persona_mode="daily",
            )

        candidate = PersonaCandidateExtractor(store).extract(scope={"session_id": "repeat-2"})[0]

        assert candidate["independent_session_count"] == 2
        assert candidate["admission_tier"] == "pending_review"
        assert candidate["requires_human_confirmation"] is True
        assert CandidatePromotionService(store).promote([candidate]).pending == 1
    finally:
        store.close()
