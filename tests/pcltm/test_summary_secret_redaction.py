from __future__ import annotations

from pcltm.summarizer import FallbackSummarizer, deterministic_summary, redact_secret_assignments
from pcltm.store import EventStore


def test_redact_secret_assignments_uses_shared_secret_policy() -> None:
    text = "auth Bearer abcdefghijklmnopqrstuvwxyz123456 and db postgres://example-user:secretpw@db.example.com/app"

    redacted = redact_secret_assignments(text)

    assert "abcdefghijklmnopqrstuvwxyz123456" not in redacted
    assert "secretpw" not in redacted
    assert "[REDACTED_SECRET]" in redacted


def test_deterministic_summary_redacts_secret_like_values() -> None:
    summary = deterministic_summary(
        [
            {
                "event_id": 1,
                "role": "tool",
                "source": "terminal",
                "persona_mode": "work",
                "content": "command output included Bearer abcdefghijklmnopqrstuvwxyz123456",
            }
        ],
        max_chars=400,
    )

    assert "abcdefghijklmnopqrstuvwxyz123456" not in summary
    assert "[REDACTED_SECRET]" in summary


def test_fallback_summarizer_redacts_external_summary_output() -> None:
    summarizer = FallbackSummarizer(
        detailed=lambda events, max_chars: "model summary leaked PASSWORD=hunter2",
    )

    summary = summarizer.summarize([], max_chars=200)

    assert "hunter2" not in summary
    assert "[REDACTED_SECRET]" in summary


def test_create_summary_node_redacts_direct_summary_writes(tmp_path) -> None:
    store = EventStore(tmp_path / "pcltm.db")

    node_id = store.create_summary_node(depth=0, summary="direct summary leaked PASSWORD=hunter2")
    node = store.get_summary_node(node_id)

    assert "hunter2" not in node["summary"]
    assert "[REDACTED_SECRET]" in node["summary"]
