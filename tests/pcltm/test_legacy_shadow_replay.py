from __future__ import annotations

import json
import hashlib

from pcltm.legacy_shadow_migration import compare_shadow_recall


def test_shadow_recall_compares_shared_content_commitments_not_numeric_id_spaces() -> None:
    shared = "c" * 64
    queries = [
        {
            "query_id": "q1",
            "query_sha256": hashlib.sha256(b"q1").hexdigest(),
            "legacy": {
                "status": "ok", "reason_codes": ["legacy_allowed"],
                "result_commitments": [shared],
            },
            "governed": {
                "status": "ok", "reason_codes": ["access_allowed"],
                "result_commitments": [shared],
            },
        },
        {
            "query_id": "q2",
            "query_sha256": hashlib.sha256(b"q2").hexdigest(),
            "legacy": {
                "status": "ok", "reason_codes": ["legacy_allowed"],
                "result_commitments": ["d" * 64],
            },
            "governed": {
                "status": "abstained", "reason_codes": ["no_answer"],
                "result_commitments": [],
            },
        },
    ]

    report = compare_shadow_recall(queries, query_bindings={"q1": "q1", "q2": "q2"})
    rendered = json.dumps(report, sort_keys=True)

    assert report["bodyless"] is True
    assert report["runtime_authority_changed"] is False
    assert report["fallback_used"] is False
    assert report["counts"] == {"different": 1, "same": 1}
    assert [item["verdict"] for item in report["diffs"]] == ["same", "different"]
    assert report["diffs"][0]["legacy_result_count"] == 1
    assert report["diffs"][0]["governed_result_count"] == 1
    assert '"result_commitments":' not in rendered
    assert "record_ids" not in rendered
    assert "claim_ids" not in rendered
    assert shared not in rendered


def test_shadow_recall_diff_rejects_bodies_and_untyped_statuses() -> None:
    bad_body = [{
        "query_id": "q1", "query_sha256": hashlib.sha256(b"q1").hexdigest(),
        "query": "private query body",
        "legacy": {"status": "ok", "reason_codes": [], "result_commitments": []},
        "governed": {"status": "ok", "reason_codes": [], "result_commitments": []},
    }]
    try:
        compare_shadow_recall(bad_body, query_bindings={"q1": "q1"})
    except ValueError as exc:
        assert str(exc) == "shadow_input_contains_body"
    else:
        raise AssertionError("body-bearing shadow input was accepted")

    bad_status = [{
        "query_id": "q1", "query_sha256": hashlib.sha256(b"q1").hexdigest(),
        "legacy": {"status": "fallback", "reason_codes": [], "result_commitments": []},
        "governed": {"status": "ok", "reason_codes": [], "result_commitments": []},
    }]
    try:
        compare_shadow_recall(bad_status, query_bindings={"q1": "q1"})
    except ValueError as exc:
        assert str(exc) == "shadow_status_invalid"
    else:
        raise AssertionError("untyped shadow status was accepted")
