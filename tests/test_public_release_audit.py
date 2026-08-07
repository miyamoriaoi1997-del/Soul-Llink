from __future__ import annotations

import importlib.util
import json
import shutil
import subprocess
import sys
from pathlib import Path

_AUDIT_PATH = Path(__file__).resolve().parents[1] / "scripts" / "public_release_audit.py"
_AUDIT_SPEC = importlib.util.spec_from_file_location("soullink_public_release_audit", _AUDIT_PATH)
assert _AUDIT_SPEC is not None and _AUDIT_SPEC.loader is not None
_AUDIT_MODULE = importlib.util.module_from_spec(_AUDIT_SPEC)
_AUDIT_SPEC.loader.exec_module(_AUDIT_MODULE)
audit = _AUDIT_MODULE.audit


def test_public_release_audit_accepts_repository() -> None:
    root = Path(__file__).resolve().parents[1]
    script = root / "scripts" / "public_release_audit.py"

    # The public audit is a release-artifact check. Remove ignored runtime logs
    # that earlier tests may have generated; the audit still fails on any
    # tracked or otherwise present release-tree residue.
    for runtime_logs in (
        root / "logs",
        root / "packages" / "persona_engine" / "logs",
    ):
        shutil.rmtree(runtime_logs, ignore_errors=True)
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


def test_public_release_audit_allows_private_marker_only_inside_declared_project_urls(tmp_path: Path) -> None:
    for name in (
        ".gitignore", "CONTRIBUTING.md", "LICENSE", "MANIFEST.in", "README.md",
        "RELEASE_CHECKLIST.md", "SECURITY.md",
    ):
        (tmp_path / name).write_text("public\n", encoding="utf-8")
    (tmp_path / "pyproject.toml").write_text(
        '[project]\nname = "example"\n\n[project.urls]\n'
        'Repository = "https://github.com/miyamoriaoi1997-del/Soul-Llink"\n',
        encoding="utf-8",
    )

    assert audit(tmp_path)["ok"] is True

    private_marker = "miyamori" + "aoi"
    (tmp_path / "README.md").write_text(
        private_marker + " private persona\n", encoding="utf-8"
    )
    report = audit(tmp_path)
    assert report["ok"] is False
    assert report["private_marker_hits"] == [
        {"path": "README.md", "categories": ["private-identity"]}
    ]
