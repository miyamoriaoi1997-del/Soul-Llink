from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path


def test_public_release_audit_accepts_repository() -> None:
    root = Path(__file__).resolve().parents[1]
    script = root / "scripts" / "public_release_audit.py"

    completed = subprocess.run(
        [sys.executable, str(script), "--root", str(root), "--json"],
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        check=False,
    )

    assert completed.returncode == 0, completed.stderr or completed.stdout
    report = json.loads(completed.stdout)
    assert report["ok"] is True
    assert report["private_marker_hits"] == []
    assert report["forbidden_files"] == []
    assert report["missing_required_files"] == []
