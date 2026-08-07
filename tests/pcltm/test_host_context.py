from __future__ import annotations

from pcltm.host_context import PCLTMContextPort, conservative_mode_hint


def test_conservative_mode_hint_separates_obvious_queries_without_forcing_ambiguous_turns() -> None:
    assert conservative_mode_hint("检查 SoulLink 运行情况") == "work"
    assert conservative_mode_hint("我爱你") == "daily"
    assert conservative_mode_hint("我们做爱") == "sex"
    assert conservative_mode_hint("那你做吧") is None


def test_context_port_passes_query_and_hint_to_injected_loader() -> None:
    calls = []
    port = PCLTMContextPort(
        loader=lambda **kwargs: calls.append(kwargs) or "<memory_context>alpha</memory_context>"
    )

    assert port.prefetch("检查数据库") == "<memory_context>alpha</memory_context>"
    assert calls == [{"mode": "work", "query": "检查数据库"}]


def test_context_port_prefers_authoritative_state_machine_mode_over_query_hint() -> None:
    calls = []
    port = PCLTMContextPort(loader=lambda **kwargs: calls.append(kwargs) or "ok")

    assert port.prefetch("我爱你", active_mode="work") == "ok"
    assert calls == [{"mode": "work", "query": "我爱你"}]


def test_context_port_forwards_session_scoped_continuity_without_inventing_it() -> None:
    calls = []
    evidence = object()
    port = PCLTMContextPort(loader=lambda **kwargs: calls.append(kwargs) or "ok")

    assert port.prefetch(
        "也就是说现在达到预期了吗",
        active_mode="work",
        session_id="session-a",
        continuity_evidence=evidence,
    ) == "ok"
    assert calls == [{
        "mode": "work",
        "query": "也就是说现在达到预期了吗",
        "session_id": "session-a",
        "continuity_evidence": evidence,
    }]
