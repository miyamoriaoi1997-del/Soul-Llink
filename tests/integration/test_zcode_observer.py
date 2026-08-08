from __future__ import annotations

import json
from pathlib import Path

from soul_link.zcode_observer import verify_injection


def _write_model_io(root: Path, *, session_id: str = "sess-1") -> Path:
    rollout = root / "rollout"
    rollout.mkdir(parents=True, exist_ok=True)
    record = {
        "sessionId": session_id,
        "type": "model_io",
        "request": {
            "body": {
                "model": "m",
                "input": [
                    {"role": "system", "content": "base instructions"},
                    {"role": "user", "content": "remember: pineapple on pizza"},
                ],
            }
        },
    }
    path = rollout / f"model-io-sess_{session_id}.jsonl"
    path.write_text(json.dumps(record, ensure_ascii=False) + "\n", encoding="utf-8")
    return path


def test_verify_injection_finds_excerpt_in_request_body(tmp_path: Path) -> None:
    _write_model_io(tmp_path)
    report = verify_injection(injected=["pineapple on pizza"], session_id="sess-1", root=tmp_path)
    assert report["observed"] is True
    assert "pineapple on pizza" in report["matches"]
    assert report["boundary"] == "model_io_log_observation"


def test_verify_injection_reports_missing_honestly(tmp_path: Path) -> None:
    _write_model_io(tmp_path)
    report = verify_injection(injected=["never injected"], session_id="sess-1", root=tmp_path)
    assert report["observed"] is False
    assert report["matches"] == []
    assert report["missing"] == ["never injected"]


def test_verify_injection_filters_by_session(tmp_path: Path) -> None:
    _write_model_io(tmp_path)
    report = verify_injection(injected=["pineapple on pizza"], session_id="other-session", root=tmp_path)
    assert report["observed"] is False
    assert report["reason"] == "no_model_io_records"


def test_verify_injection_empty_input_verifies_nothing(tmp_path: Path) -> None:
    report = verify_injection(injected=[], root=tmp_path)
    assert report["observed"] is False
    assert report["reason"] == "no_injected_content"


def test_verify_injection_without_rollout_dir(tmp_path: Path) -> None:
    report = verify_injection(injected=["x"], root=tmp_path)
    assert report["observed"] is False
    assert report["reason"] == "no_model_io_records"


def test_verify_injection_ignores_corrupt_lines(tmp_path: Path) -> None:
    rollout = tmp_path / "rollout"
    rollout.mkdir(parents=True, exist_ok=True)
    path = rollout / "model-io-sess_1.jsonl"
    path.write_text("not json\n", encoding="utf-8")
    report = verify_injection(injected=["x"], session_id="sess-1", root=tmp_path)
    assert report["observed"] is False
    assert report["reason"] == "no_model_io_records"
