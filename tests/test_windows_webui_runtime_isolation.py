from __future__ import annotations

from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
START_SCRIPT = ROOT / "scripts/windows/start-webui.ps1"


def test_webui_uses_soullink_owned_python_not_hermes_venv() -> None:
    text = START_SCRIPT.read_text(encoding="utf-8")

    assert 'Join-Path $repoRoot ".venv\\Scripts\\python.exe"' in text
    assert 'hermes\\hermes-agent\\venv\\Scripts\\python.exe' not in text
    assert "SoulLink Python was not found" in text
