from __future__ import annotations

from types import SimpleNamespace

from model_router import app


class _FakeConfig:
    routing = {"enabled": False, "default_model": "default-model"}

    def upstream(self):
        return app.UpstreamConfig(
            base_url="http://upstream.test/v1/chat/completions",
            api_key="test-key",
            timeout_seconds=1,
        )


class _FakeHandler:
    _call_upstream = app.Handler._call_upstream
    _forward = app.Handler._forward
    _audit = app.Handler._audit

    def __init__(self, status_code: int):
        self.server = SimpleNamespace(cfg=_FakeConfig())
        self.status_code = status_code
        self.audit_events = []
        self.sent_errors = []

    def _call_upstream(self, payload, upstream):
        return self.status_code, b"{}", "application/json", {}

    def _audit(self, request_hash, payload, decision, selected_model, fallback_used, status, ok, started, error):
        self.audit_events.append(
            {
                "selected_model": selected_model,
                "status": status,
                "ok": ok,
                "error": error,
            }
        )

    def _send_json(self, code, payload):
        self.sent_errors.append((code, payload))


def test_forward_audit_marks_upstream_non_2xx_as_not_ok():
    handler = _FakeHandler(500)

    result = handler._forward({"messages": []}, b"{}", "hash", 0.0)

    assert result == (500, b"{}", "application/json", {})
    assert handler.audit_events == [
        {
            "selected_model": "default-model",
            "status": 500,
            "ok": False,
            "error": None,
        }
    ]
    assert handler.sent_errors == []


def test_forward_audit_marks_upstream_2xx_as_ok():
    handler = _FakeHandler(200)

    result = handler._forward({"messages": []}, b"{}", "hash", 0.0)

    assert result == (200, b"{}", "application/json", {})
    assert handler.audit_events[0]["ok"] is True
