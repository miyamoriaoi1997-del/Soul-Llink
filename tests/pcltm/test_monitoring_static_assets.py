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
        'id="context-board"',
        'id="emotion-board"',
        'id="chain-stages"',
        'id="layers"',
        'id="evidence-table"',
        'id="rawBlock"',
    ):
        assert expected in html
    assert "MOCK SNAPSHOT" not in html
    assert "CONCEPT PROTOTYPE" not in html
    assert "Soul 状态与决策权威" in html
    assert "对话处理中代表本轮实时数据；回复完成后代表最近已完成轮次" in html
    assert "此刻的状态，以及它如何形成。" not in html
    assert "上一轮对话的真实权威快照，以及它如何形成。" not in html
    assert "LATEST AUTHORITATIVE TURN" in html
    assert "LATEST AUTHORITATIVE TURN" in js
    assert "LAST COMPLETED TURN" not in html
    assert "LAST COMPLETED TURN" not in js
    assert "LIVE SNAPSHOT" not in html
    assert "LIVE SNAPSHOT" not in js
    assert "fetch('/api/v1/snapshot'" in js
    assert "exact_host_capture" in js
    assert "memory_selection" in js
    assert "selected_records" in js
    assert "FINAL FORWARD" in js
    assert "SHA-256" in js
    assert "sidecar reconstruction" in js
    assert "当前生效记忆" in js
    assert "active_memory_count" in js
    assert "active_event_derived_count" in js
    assert "生效来源拆分" in js
    assert "当前旧记录列表" in js
    assert "原始事件证据" in js
    assert "来源证据，不计入生效记忆" in js
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
    assert "上下文构成" in js
    assert "CONTEXT OBSERVATION" in js
    assert "最近一次宿主用量快照" in js
    assert "非连续实时计数器" in js
    assert "上下文架构地图" in js
    assert "策略容量配置" in js
    assert "静态/请求配置 · 非实际使用" in js
    assert "SoulLink 最终载荷证据" in js
    assert "系统与开发者指令" in js
    assert "对话与当前消息" in js
    assert "工具定义" in js
    assert "连续性层" in js
    assert "工具证据层" in js
    assert "核心人格层" in js
    assert "响应预留" in js
    assert "存在已确认 · 用量未分层" in js
    assert "情绪注入层" in js
    assert "模式人格层" in js
    assert "记忆召回层" in js
    assert "turn_injection 容器还含状态机标记和包装文本" in js
    assert "final model forward 独立边界" in js
    assert "context.budget_buckets" in js
    assert "context.stale===true" in js
    assert "policyCard('continuity','Continuity'" in js
    assert "policyCard('tool_evidence','Tool evidence'" in js
    assert "turn_injection" in js
    assert ".context-composition" in css
    assert ".usage-grid" in css
    assert ".architecture-map" in css
    assert ".policy-grid" in css
    assert ".payload-grid" in css
    assert ".emotion-axes{display:grid" in css
    assert "$('context-board').innerHTML=renderContextComposition" in js
    assert "$('emotion-board').innerHTML" in js
    assert '<details class="context-section' not in js
    assert '<details class="emotion-copy' not in js
    assert ".context-board .context-sections" in css
    assert ".emotion-board{display:grid" in css
    assert "观测冲突" in js
    assert "selectedRecords(capture).length" in js
    assert 'id="status-rail"' in html
    assert 'id="posture-core"' in html
    assert 'id="authority-core"' in html
    assert 'id="semantic-core"' in html
    assert "renderDecisionSurface" in js
    assert "capture?.host_turn_count ?? capture?.turn_number" in js
    assert "semantic_shadow" in js
    assert "semantic_fusion" in js
    assert "authority_source" in js
    assert ".command-grid" in css
    assert ".authority-track" in css
    assert ".semantic-bar" in css
    assert 'style="--confidence:' not in js
    assert "mode-ring" in js
