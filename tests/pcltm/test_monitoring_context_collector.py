from __future__ import annotations

import json
from datetime import datetime, timedelta, timezone

from pcltm.monitoring import collectors


NOW = datetime(2026, 7, 13, 22, 30, tzinfo=timezone.utc)
OPAQUE_SENTINEL = "opaque-context-sentinel-84c1"


def test_context_collector_separates_config_from_last_telemetry(monkeypatch) -> None:
    monkeypatch.setattr(
        collectors,
        "last_live_context_telemetry",
        lambda: {
            "within_budget": True,
            "total_chars": 1200,
            "limit_chars": 3600,
            "omitted_chars": 40,
            "actions": ["omitted"],
            "capsules": {"continuation": 1, "tool_evidence": 2},
            "observed_at": (NOW - timedelta(seconds=4)).isoformat(),
            "prompt": OPAQUE_SENTINEL,
        },
    )

    report = collectors.collect_context_budget(
        config={
            "model": {"context_length": 400_000},
            "context": {"engine": "pcltm-context", "budget_tokens": 256_000},
            "compression": {"threshold": 0.5},
        },
        now=NOW,
    )

    assert report == {
        "model_context_tokens": 400_000,
        "budget_tokens": 256_000,
        "trigger_threshold": 0.5,
        "engine": "pcltm-context",
        "budget_buckets": {
            "active_frame": 179_200,
            "continuity": 12_800,
            "tool_evidence": 12_800,
            "recall": 20_480,
            "reserve": 30_720,
        },
        "within_budget": True,
        "rendered_chars": 1200,
        "limit_chars": 3600,
        "usage_ratio": 1 / 3,
        "omitted_chars": 40,
        "truncated_count": 1,
        "capsule_count": 3,
        "source": "config+last_telemetry",
        "observed_at": "2026-07-13T22:29:56+00:00",
        "age_seconds": 4.0,
        "stale": False,
        "issues": [],
    }
    assert OPAQUE_SENTINEL not in repr(report)


def test_context_collector_marks_old_telemetry_stale(monkeypatch) -> None:
    monkeypatch.setattr(
        collectors,
        "last_live_context_telemetry",
        lambda: {
            "within_budget": False,
            "total_chars": 3600,
            "limit_chars": 3600,
            "omitted_chars": 0,
            "actions": [],
            "capsules": {},
            "observed_at": (NOW - timedelta(seconds=31)).isoformat(),
        },
    )

    report = collectors.collect_context_budget(
        config={"context": {"budget_tokens": 256_000}},
        now=NOW,
        stale_after_seconds=30,
    )

    assert report["source"] == "config+last_telemetry"
    assert report["age_seconds"] == 31.0
    assert report["stale"] is True
    assert report["issues"][0]["code"] == "CONTEXT_TELEMETRY_STALE"


def test_context_collector_uses_config_only_when_telemetry_has_no_timestamp(
    monkeypatch,
) -> None:
    monkeypatch.setattr(
        collectors,
        "last_live_context_telemetry",
        lambda: {"within_budget": True, "total_chars": 100, "limit_chars": 200},
    )

    report = collectors.collect_context_budget(
        config={"context": {"engine": "pcltm-context", "budget_tokens": 1000}},
        now=NOW,
    )

    assert report["source"] == "config"
    assert report["observed_at"] is None
    assert report["age_seconds"] is None
    assert report["within_budget"] is None
    assert report["issues"][0]["code"] == "CONTEXT_TELEMETRY_UNTIMESTAMPED"


def test_context_collector_budget_boundary_is_not_rounded_up(monkeypatch) -> None:
    monkeypatch.setattr(collectors, "last_live_context_telemetry", lambda: {})

    report = collectors.collect_context_budget(
        config={"context": {"budget_tokens": 255_999}},
        now=NOW,
    )

    assert report["budget_tokens"] == 255_999
    assert sum(report["budget_buckets"].values()) == 255_999
    assert report["source"] == "config"


def test_context_collector_reads_exact_cross_process_host_usage(tmp_path, monkeypatch) -> None:
    monkeypatch.setattr(collectors, "last_live_context_telemetry", lambda: {})
    telemetry_path = tmp_path / "soullink-context-telemetry.json"
    telemetry_path.write_text(
        json.dumps(
            {
                "source": "exact_host_context_usage",
                "observed_at": (NOW - timedelta(seconds=2)).isoformat(),
                "session_id": "session-live",
                "engine": "pcltm-context",
                "prompt_tokens": 64_000,
                "completion_tokens": 1_200,
                "total_tokens": 65_200,
                "context_length": 400_000,
                "budget_tokens": 256_000,
                "compression_count": 3,
                "cache_read_tokens": 48_000,
            }
        ),
        encoding="utf-8",
    )

    report = collectors.collect_context_budget(
        config={
            "model": {"context_length": 400_000},
            "context": {"engine": "pcltm-context", "budget_tokens": 256_000},
        },
        telemetry_path=telemetry_path,
        now=NOW,
    )

    assert report["source"] == "exact_host_context_usage"
    assert report["observed_at"] == "2026-07-13T22:29:58+00:00"
    assert report["age_seconds"] == 2.0
    assert report["prompt_tokens"] == 64_000
    assert report["completion_tokens"] == 1_200
    assert report["total_tokens"] == 65_200
    assert report["usage_ratio"] == 0.25
    assert report["within_budget"] is True
    assert report["compression_count"] == 3
    assert report["cache_read_tokens"] == 48_000
    assert report["session_id"] == "session-live"


def test_exact_host_usage_supplies_live_budget_and_model_window_when_config_is_empty(
    tmp_path, monkeypatch
) -> None:
    monkeypatch.setattr(collectors, "last_live_context_telemetry", lambda: {})
    telemetry_path = tmp_path / "soullink-context-telemetry.json"
    telemetry_path.write_text(
        json.dumps({
            "source": "exact_host_context_usage",
            "observed_at": (NOW - timedelta(seconds=2)).isoformat(),
            "session_id": "session-live",
            "prompt_tokens": 100_000,
            "completion_tokens": 300,
            "total_tokens": 100_300,
            "context_length": 400_000,
            "budget_tokens": 200_000,
        }),
        encoding="utf-8",
    )

    report = collectors.collect_context_budget(config={}, telemetry_path=telemetry_path, now=NOW)

    assert report["source"] == "exact_host_context_usage"
    assert report["budget_tokens"] == 200_000
    assert report["model_context_tokens"] == 400_000
    assert report["usage_ratio"] == 0.5
    assert sum(report["budget_buckets"].values()) == 200_000


def test_exact_host_usage_supplies_live_budget_and_model_window_when_sidecar_config_is_empty(
    tmp_path, monkeypatch
) -> None:
    monkeypatch.setattr(collectors, "last_live_context_telemetry", lambda: {})
    telemetry_path = tmp_path / "soullink-context-telemetry.json"
    telemetry_path.write_text(
        json.dumps(
            {
                "source": "exact_host_context_usage",
                "observed_at": (NOW - timedelta(seconds=2)).isoformat(),
                "session_id": "session-live",
                "prompt_tokens": 107_417,
                "completion_tokens": 293,
                "total_tokens": 107_710,
                "context_length": 400_000,
                "budget_tokens": 200_000,
            }
        ),
        encoding="utf-8",
    )

    report = collectors.collect_context_budget(
        config={}, telemetry_path=telemetry_path, now=NOW
    )

    assert report["source"] == "exact_host_context_usage"
    assert report["prompt_tokens"] == 107_417
    assert report["budget_tokens"] == 200_000
    assert report["model_context_tokens"] == 400_000
    assert report["usage_ratio"] == 107_417 / 200_000
    assert sum(report["budget_buckets"].values()) == 200_000
