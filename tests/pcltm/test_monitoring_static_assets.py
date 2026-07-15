from __future__ import annotations

from importlib import resources


def test_private_monitor_static_assets_expose_live_capture_and_soul_views() -> None:
    package = resources.files("pcltm.monitoring.static")
    html = package.joinpath("index.html").read_text(encoding="utf-8")
    js = package.joinpath("app.js").read_text(encoding="utf-8")

    for expected in (
        'id="runtime-injection"',
        'id="emotion-modifier-body"',
        'id="state-machine-details"',
        'id="runtime-injection-body"',
        'id="soul"',
        'id="soul-body"',
    ):
        assert expected in html
    assert "renderRuntimeCapture(s.runtime_capture)" in js
    assert "renderSoul(s.soul)" in js
    assert "snapshot?.soul" in js
    assert "上一模型请求实际输入 tokens" in js
    assert "c.prompt_tokens" in js
    assert "exact_host_context_usage" in js
    assert "PCLTM Mode" in js
    assert "Mode Sync" in js
