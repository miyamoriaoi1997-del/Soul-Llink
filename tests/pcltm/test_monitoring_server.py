from __future__ import annotations

import json
import threading
from urllib import error, request

import pytest

from pcltm.monitoring.server import create_server


@pytest.fixture
def running_server():
    service = type("Service", (), {"get": lambda self: {"ok": True, "issues": [], "runtime": {"status": "healthy"}}})()
    server = create_server("127.0.0.1", 0, service)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        yield f"http://127.0.0.1:{server.server_port}"
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=2)


def test_health_snapshot_404_and_security_headers(running_server: str) -> None:
    with request.urlopen(running_server + "/api/v1/health") as response:
        payload = json.load(response)
        assert payload["service"] == "soullink-monitor"
        assert response.headers["X-Content-Type-Options"] == "nosniff"
        assert "default-src 'self'" in response.headers["Content-Security-Policy"]
    with request.urlopen(running_server + "/api/v1/snapshot") as response:
        assert json.load(response)["runtime"]["status"] == "healthy"
    with pytest.raises(error.HTTPError) as exc:
        request.urlopen(running_server + "/unknown")
    assert exc.value.code == 404


def test_writes_and_non_loopback_binding_are_rejected(running_server: str) -> None:
    req = request.Request(running_server + "/api/v1/snapshot", data=b"{}", method="POST")
    with pytest.raises(error.HTTPError) as exc:
        request.urlopen(req)
    assert exc.value.code == 405
    with pytest.raises(ValueError, match="loopback"):
        create_server("0.0.0.0", 0, object())


def test_dns_rebinding_host_is_rejected(running_server: str) -> None:
    req = request.Request(running_server + "/api/v1/snapshot")
    req.add_header("Host", "attacker.example")
    with pytest.raises(error.HTTPError) as exc:
        request.urlopen(req)
    assert exc.value.code == 421
    assert json.load(exc.value)["error"] == "untrusted_host"
