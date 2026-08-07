from __future__ import annotations

import json
import os
import subprocess
import sys

from pcltm.memory_adapter import load_entries
from pcltm.memory_selection_probe import build_probe_report, explain_memory_records


EXPECTED = {
    "status": "retired",
    "bodyless": True,
    "reason": "legacy_memory_selection_probe_not_runtime_authority",
    "target": "user",
    "mode": "work",
}


def test_legacy_selection_probe_is_bodyless_and_retired() -> None:
    assert explain_memory_records(
        "user", mode="work", emotion_axes={"focus"}, budget_available=1.0,
    ) == []
    assert load_entries("user") == []
    assert build_probe_report(
        "user", mode="work", emotion_axes={"focus"}, budget_available=1.0,
    ) == EXPECTED


def test_memory_selection_probe_cli_emits_only_retirement_receipt() -> None:
    env = os.environ.copy()
    env["PYTHONPATH"] = os.pathsep.join(("packages", "adapters"))
    result = subprocess.run(
        [sys.executable, "-m", "pcltm.memory_selection_probe", "--target", "user",
         "--mode", "work", "--budget", "1.0", "--json"],
        check=True, capture_output=True, env=env, text=True,
    )
    report = json.loads(result.stdout)

    assert report == EXPECTED
    assert "selected" not in report
    assert "skipped" not in report
    assert "load_entries_baseline" not in report
