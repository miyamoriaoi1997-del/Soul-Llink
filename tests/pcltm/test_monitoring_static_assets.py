from __future__ import annotations

from importlib import resources


def test_v124_static_assets_expose_five_live_views_without_mock_claims() -> None:
    package = resources.files("pcltm.monitoring.static")
    html = package.joinpath("index.html").read_text(encoding="utf-8")
    js = package.joinpath("app.js").read_text(encoding="utf-8")
    css = package.joinpath("styles.css").read_text(encoding="utf-8")

    for expected in (
        'data-view="now"',
        'data-view="memory"',
        'data-view="injection"',
        'data-view="identity"',
        'data-view="evidence"',
        'id="memory-stories"',
        'id="memory-path"',
        'id="chain-stages"',
        'id="layers"',
        'id="evidence-table"',
        'id="rawBlock"',
    ):
        assert expected in html
    assert "MOCK SNAPSHOT" not in html
    assert "CONCEPT PROTOTYPE" not in html
    assert "fetch('/api/v1/snapshot'" in js
    assert "exact_host_capture" in js
    assert "memory_selection" in js
    assert "selected_records" in js
    assert "FINAL FORWARD" in js
    assert "SHA-256" in js
    assert "sidecar reconstruction" in js
    assert "persistent_memory_total" in js
    assert "event_chunks excluded" in js
    assert "renderMemoryPath" in js
    assert "candidate_records" in js
    assert "judgment_workset" in js
    assert 'id="memory-orb"' not in html
    assert "renderOrb" not in js
    assert "requestAnimationFrame" not in js
    assert "not inferred" in js
    assert ".memory-path" in css
    assert ".record-routes" in css
    assert "emotion_score" in js
    assert "[-5, +5]" in js
    assert "capture?.emotion_modifier" in js
    assert "当前情绪状态" in js
    assert "情绪注入词" in js
    assert 'class="bar" value="${safeV}" max="120"' in js
    assert 'class="score-range" type="range" min="-5" max="5"' in js
    assert ".bar::-webkit-progress-value" in css
    assert ".score-range::-webkit-slider-thumb" in css
    assert "white-space:nowrap" in css
    assert "context.budget_tokens" in js
    assert "当前用量" in js
    assert "观测冲突" in js
    assert "selectedRecords(capture).length" in js
