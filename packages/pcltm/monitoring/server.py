"""Localhost-only read-only HTTP server for SoulLink monitoring."""

from __future__ import annotations

import ipaddress
import json
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from importlib.resources import files
from typing import Any
from urllib.parse import urlsplit


class MonitorServer(ThreadingHTTPServer):
    daemon_threads = True
    allow_reuse_address = True

    def __init__(self, address, handler, service):
        self.snapshot_service = service
        super().__init__(address, handler)

    @property
    def allowed_hosts(self) -> set[str]:
        port = int(self.server_port)
        host = str(self.server_address[0]).lower()
        allowed = {host, f"{host}:{port}", "localhost", f"localhost:{port}"}
        if host == "::1":
            allowed.update({"[::1]", f"[::1]:{port}"})
        return allowed


class MonitorHandler(BaseHTTPRequestHandler):
    server: MonitorServer

    def log_message(self, format: str, *args: Any) -> None:
        return

    def _headers(self, content_type: str, length: int) -> None:
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(length))
        self.send_header("Cache-Control", "no-store")
        self.send_header("X-Content-Type-Options", "nosniff")
        self.send_header("Content-Security-Policy", "default-src 'self'; connect-src 'self'; style-src 'self'; script-src 'self'")
        self.send_header("Referrer-Policy", "no-referrer")

    def _send(self, code: int, body: bytes, content_type: str) -> None:
        self.send_response(code)
        self._headers(content_type, len(body))
        self.end_headers()
        self.wfile.write(body)

    def _json(self, code: int, payload: Any) -> None:
        self._send(code, json.dumps(payload, ensure_ascii=False).encode("utf-8"), "application/json; charset=utf-8")

    def _trusted_host(self) -> bool:
        # A loopback bind alone does not prevent browser DNS rebinding.
        host = str(self.headers.get("Host") or "").strip().lower().rstrip(".")
        return host in self.server.allowed_hosts

    def do_GET(self) -> None:
        if not self._trusted_host():
            self._json(421, {"ok": False, "error": "untrusted_host"})
            return
        path = urlsplit(self.path).path
        if len(self.path) > 2048:
            self._json(414, {"ok": False, "error": "uri_too_long"})
            return
        snapshot = self.server.snapshot_service.get
        if path == "/api/v1/health":
            self._json(200, {"ok": True, "service": "soullink-monitor", "version": "1"})
        elif path == "/api/v1/snapshot":
            self._json(200, snapshot())
        elif path == "/api/v1/issues":
            self._json(200, {"issues": snapshot().get("issues", [])})
        elif path in {"/", "/assets/app.js", "/assets/styles.css"}:
            names = {"/": "index.html", "/assets/app.js": "app.js", "/assets/styles.css": "styles.css"}
            name = names[path]
            try:
                body = files("pcltm.monitoring.static").joinpath(name).read_bytes()
            except (FileNotFoundError, ModuleNotFoundError):
                self._json(404, {"ok": False, "error": "asset_missing"})
                return
            ctype = "text/html; charset=utf-8" if name.endswith("html") else "text/javascript; charset=utf-8" if name.endswith("js") else "text/css; charset=utf-8"
            self._send(200, body, ctype)
        else:
            self._json(404, {"ok": False, "error": "not_found"})

    def _reject_write(self) -> None:
        if not self._trusted_host():
            self._json(421, {"ok": False, "error": "untrusted_host"})
            return
        self._json(405, {"ok": False, "error": "method_not_allowed"})

    do_POST = do_PUT = do_PATCH = do_DELETE = _reject_write


def create_server(host: str, port: int, snapshot_service: Any) -> MonitorServer:
    try:
        address = ipaddress.ip_address(host)
    except ValueError as exc:
        raise ValueError("monitor host must be a loopback IP address") from exc
    if not address.is_loopback:
        raise ValueError("monitor host must be loopback")
    return MonitorServer((host, int(port)), MonitorHandler, snapshot_service)


__all__ = ["MonitorServer", "create_server"]
