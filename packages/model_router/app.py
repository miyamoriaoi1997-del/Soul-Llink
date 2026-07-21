#!/usr/bin/env python3
"""OpenAI-compatible local model router proxy for Hermes.

Design constraints:
- No raw prompts, API keys, or headers in audit logs.
- Request-level model routing: every chat/completions request is routed fresh.
- Preserve OpenAI-compatible payload fields, including tools/tool_choice.
- Support non-streaming and streaming passthrough.
- Route only from Hermes/persona state-machine metadata; upstream fallback remains outside this proxy.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import pathlib
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Any, Dict, Iterable, Tuple
from urllib import error as urlerror
from urllib import request as urlrequest
from urllib.parse import urlsplit

import yaml

PROJECT_DIR = pathlib.Path(__file__).resolve().parent
DEFAULT_CONFIG = PROJECT_DIR / "config.yaml"


@dataclass
class UpstreamConfig:
    base_url: str
    api_key: str
    timeout_seconds: int


@dataclass
class RouteDecision:
    route: str
    selected_model: str
    reason: str


class RouterConfig:
    def __init__(self, path: pathlib.Path):
        self.path = path
        self.data = self._load_yaml(path)

    @staticmethod
    def _load_yaml(path: pathlib.Path) -> dict:
        if not path.exists():
            raise FileNotFoundError(f"config not found: {path}")
        return yaml.safe_load(path.read_text(encoding="utf-8")) or {}

    def reload(self) -> None:
        self.data = self._load_yaml(self.path)

    @property
    def listen_host(self) -> str:
        return str((self.data.get("listen") or {}).get("host") or "127.0.0.1")

    @property
    def listen_port(self) -> int:
        return int((self.data.get("listen") or {}).get("port") or 18080)

    @property
    def audit_path(self) -> pathlib.Path:
        p = (self.data.get("audit") or {}).get("path") or str(PROJECT_DIR / "logs" / "audit.jsonl")
        return pathlib.Path(p)

    @property
    def routing(self) -> dict:
        return self.data.get("routing") or {}

    @property
    def model_metadata(self) -> dict:
        return dict(self.data.get("model_metadata") or {})

    def upstream(self) -> UpstreamConfig:
        upstream = self.data.get("upstream") or {}
        hermes_config = pathlib.Path(upstream.get("hermes_config_path") or str(Path.home() / ".hermes" / "config.yaml"))
        hcfg = self._load_yaml(hermes_config) if hermes_config.exists() else {}
        model_cfg = hcfg.get("model") or {}
        # Important: once Hermes itself points at this local proxy, reading model.base_url
        # back from Hermes would create a self-forwarding loop. Keep the true upstream
        # endpoint pinned in this proxy config.
        base = str(upstream.get("base_url") or model_cfg.get("base_url") or "").rstrip("/")
        if not base:
            raise RuntimeError("upstream base_url is empty")
        parsed = urlsplit(base)
        if parsed.scheme not in {"http", "https"}:
            raise RuntimeError("upstream base_url must use http or https")
        if not parsed.hostname:
            raise RuntimeError("upstream base_url must include a hostname")
        try:
            parsed.port
        except ValueError as exc:
            raise RuntimeError("upstream base_url has an invalid port") from exc
        if parsed.username is not None or parsed.password is not None:
            raise RuntimeError("upstream base_url must not embed credentials")
        if parsed.query or parsed.fragment:
            raise RuntimeError("upstream base_url must not include query or fragment data")
        if not base.endswith("/v1"):
            base += "/v1"
        api_key = str(upstream.get("api_key") or model_cfg.get("api_key") or "")
        timeout = int(upstream.get("timeout_seconds") or 180)
        return UpstreamConfig(base_url=base, api_key=api_key, timeout_seconds=timeout)


def _truthy(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, float)):
        return value != 0
    if isinstance(value, str):
        return value.strip().lower() in {"1", "true", "yes", "y", "on", "allowed", "allow"}
    return False


def _configured_model(routing: dict, *keys: str, required: bool = True) -> str:
    for key in keys:
        value = routing.get(key)
        if value:
            return str(value)
    if required:
        joined = " or ".join(f"routing.{key}" for key in keys)
        raise RuntimeError(f"{joined} is required")
    return ""


def _state_machine_signal(payload: dict) -> dict:
    metadata = payload.get("metadata")
    if not isinstance(metadata, dict):
        metadata = {}
    signal = {}
    for key in (
        "hermes_route_bucket",
        "route_bucket",
        "hermes_model_hint",
        "model_hint",
        "hermes_selected_model",
        "selected_model",
        "hermes_model_override",
        "model_override",
        "hermes_switch_allowed",
        "switch_allowed",
        "hermes_switch_reason",
        "switch_reason",
        "hermes_turn_correlation_id",
    ):
        if key in metadata:
            signal[key] = metadata[key]
    return signal


def _strip_hermes_routing_metadata(payload: dict) -> dict:
    metadata = payload.get("metadata")
    if not isinstance(metadata, dict):
        return payload
    cleaned_metadata = {
        k: v for k, v in metadata.items()
        if not (isinstance(k, str) and (k.startswith("hermes_") or k in {"route_bucket", "model_hint", "switch_allowed", "switch_reason"}))
    }
    cleaned = dict(payload)
    if cleaned_metadata:
        cleaned["metadata"] = cleaned_metadata
    else:
        cleaned.pop("metadata", None)
    return cleaned


def extract_text_for_routing(payload: dict) -> str:
    """Extract text snippets for routing without returning them to logs."""
    parts: list[str] = []
    for msg in payload.get("messages") or []:
        content = msg.get("content")
        if isinstance(content, str):
            parts.append(content)
        elif isinstance(content, list):
            for item in content:
                if isinstance(item, dict):
                    if item.get("type") == "text" and isinstance(item.get("text"), str):
                        parts.append(item["text"])
                    elif isinstance(item.get("content"), str):
                        parts.append(item["content"])
    return "\n".join(parts)


def decide_route(payload: dict, cfg: RouterConfig) -> RouteDecision:
    routing = cfg.routing
    enabled = bool(routing.get("enabled", True))
    original_model = str(payload.get("model") or "")
    virtual_models = set(routing.get("virtual_models") or ["persona-auto", "persona-auto-technical"])
    default_model = _configured_model(routing, "default_model")
    work_model = _configured_model(routing, "work_model", required=False) or default_model

    if not enabled:
        return RouteDecision("default", default_model, "kill_switch_disabled")

    signal = _state_machine_signal(payload)
    route_bucket = str(signal.get("hermes_route_bucket") or signal.get("route_bucket") or "").strip().lower()
    model_hint = str(signal.get("hermes_model_hint") or signal.get("model_hint") or "").strip().lower()
    explicit_selected_model = str(
        signal.get("hermes_selected_model")
        or signal.get("selected_model")
        or signal.get("hermes_model_override")
        or signal.get("model_override")
        or ""
    ).strip()

    if explicit_selected_model and explicit_selected_model not in virtual_models:
        return RouteDecision("selected", explicit_selected_model, "state_machine:selected_model")

    if route_bucket in {"task", "work", "system_maintenance", "technical"} or model_hint in {"work", "technical"}:
        return RouteDecision("technical", work_model, "state_machine:task")
    if route_bucket in {"relationship", "daily", "intimacy", "repair", "conflict", "default"} or model_hint == "default":
        return RouteDecision("default", default_model, f"state_machine:{route_bucket or 'default'}")
    if route_bucket == "sex" or model_hint == "sex":
        sex_model = str(routing.get("sex_model") or default_model)
        return RouteDecision("sex", sex_model, "state_machine:sex")

    if original_model in virtual_models or not original_model:
        return RouteDecision("default", default_model, "virtual_default")

    # Non-virtual explicit model: preserve caller's model unless keyword route overrides.
    return RouteDecision("explicit", original_model, "explicit_passthrough")


def json_bytes(obj: Any) -> bytes:
    return json.dumps(obj, ensure_ascii=False, separators=(",", ":")).encode("utf-8")


def make_request_hash(raw: bytes) -> str:
    return hashlib.sha256(raw).hexdigest()[:16]


def safe_error_text(exc: BaseException) -> str:
    msg = str(exc)
    if len(msg) > 300:
        msg = msg[:300] + "..."
    return f"{type(exc).__name__}: {msg}"


class RouterServer(ThreadingHTTPServer):
    def __init__(self, server_address: tuple[str, int], RequestHandlerClass, cfg: RouterConfig):
        super().__init__(server_address, RequestHandlerClass)
        self.cfg = cfg


class Handler(BaseHTTPRequestHandler):
    server: RouterServer
    protocol_version = "HTTP/1.1"

    def log_message(self, fmt: str, *args: Any) -> None:
        # Avoid stdout spam under systemd. Audit JSONL is the source of truth.
        return

    def _send_json(self, code: int, obj: Any) -> None:
        body = json_bytes(obj)
        self.send_response(code)
        self.send_header("content-type", "application/json")
        self.send_header("content-length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def _send_raw(self, code: int, body: bytes, content_type: str = "application/json", extra_headers: dict | None = None) -> None:
        self.send_response(code)
        self.send_header("content-type", content_type)
        self.send_header("content-length", str(len(body)))
        if extra_headers:
            for k, v in extra_headers.items():
                lk = k.lower()
                if lk in {"content-length", "content-encoding", "transfer-encoding", "connection"}:
                    continue
                self.send_header(k, v)
        self.end_headers()
        self.wfile.write(body)

    def _health_payload(self) -> dict:
        self.server.cfg.reload()
        upstream = self.server.cfg.upstream()
        routing = self.server.cfg.routing
        return {
            "ok": True,
            "upstream_base_url": upstream.base_url,
            "default_model": routing.get("default_model"),
            "work_model": routing.get("work_model"),
            "sex_model": routing.get("sex_model"),
            "routing_enabled": bool(routing.get("enabled", True)),
        }

    def _models_payload(self) -> dict:
        self.server.cfg.reload()
        routing = self.server.cfg.routing
        model_ids: list[str] = []
        for item in routing.get("virtual_models") or ["persona-auto", "persona-auto-technical"]:
            if item and item not in model_ids:
                model_ids.append(str(item))
        for key in ("default_model", "work_model", "sex_model"):
            item = routing.get(key)
            if item and item not in model_ids:
                model_ids.append(str(item))
        model_metadata = self.server.cfg.model_metadata
        return {
            "object": "list",
            "data": [
                {"id": model_id, "object": "model", "owned_by": "model-router", **model_metadata}
                for model_id in model_ids
            ],
        }

    def do_GET(self) -> None:
        path = self.path.rstrip("/") or "/"
        if path in {"/healthz", "/health"}:
            try:
                self._send_json(200, self._health_payload())
            except Exception as exc:
                self._send_json(500, {"ok": False, "error": safe_error_text(exc)})
            return
        if path == "/v1/models":
            try:
                self._send_json(200, self._models_payload())
            except Exception as exc:
                self._send_json(500, {"error": {"message": safe_error_text(exc), "type": "router_error"}})
            return
        self._send_json(404, {"error": {"message": "not found", "type": "not_found"}})

    def do_POST(self) -> None:
        if self.path.rstrip("/") != "/v1/chat/completions":
            self._send_json(404, {"error": {"message": "not found", "type": "not_found"}})
            return
        raw = self.rfile.read(int(self.headers.get("content-length") or 0))
        request_hash = make_request_hash(raw)
        started = time.time()
        try:
            payload = json.loads(raw.decode("utf-8"))
        except Exception as exc:
            self._send_json(400, {"error": {"message": safe_error_text(exc), "type": "bad_request"}})
            return

        try:
            self.server.cfg.reload()
            response = self._forward(payload, raw, request_hash, started)
            if response is not None:
                code, body, ctype, headers = response
                self._send_raw(code, body, ctype, headers)
        except Exception as exc:
            self._audit(request_hash, payload, None, None, False, None, False, started, safe_error_text(exc))
            self._send_json(502, {"error": {"message": safe_error_text(exc), "type": "router_error"}})

    def _forward(self, payload: dict, raw: bytes, request_hash: str, started: float):
        cfg = self.server.cfg
        upstream = cfg.upstream()
        decision = decide_route(payload, cfg)

        last_error: str | None = None
        fallback_used = False
        out = _strip_hermes_routing_metadata(dict(payload))
        out["model"] = decision.selected_model
        try:
            code, body, ctype, headers = self._call_upstream(out, upstream)
            ok = 200 <= code < 300
            self._audit(request_hash, payload, decision, decision.selected_model, fallback_used, code, ok, started, None)
            return code, body, ctype, headers
        except Exception as exc:
            last_error = safe_error_text(exc)
            self._audit(request_hash, payload, decision, decision.selected_model, fallback_used, None, False, started, last_error)
            self._send_json(502, {"error": {"message": last_error, "type": "upstream_error"}})
            return None

        self._send_json(502, {"error": {"message": last_error or "upstream failed", "type": "upstream_error"}})
        return None

    def _call_upstream(self, payload: dict, upstream: UpstreamConfig) -> Tuple[int, bytes, str, dict]:
        body = json_bytes(payload)
        headers = {
            "content-type": "application/json",
            "authorization": "Bearer " + upstream.api_key,
        }
        req = urlrequest.Request(upstream.base_url + "/chat/completions", data=body, headers=headers, method="POST")
        try:
            with urlrequest.urlopen(req, timeout=upstream.timeout_seconds) as resp:
                data = resp.read()
                ctype = resp.headers.get("content-type", "application/json")
                return resp.status, data, ctype, dict(resp.headers.items())
        except urlerror.HTTPError as exc:
            data = exc.read()
            ctype = exc.headers.get("content-type", "application/json") if exc.headers else "application/json"
            return exc.code, data, ctype, dict(exc.headers.items()) if exc.headers else {}

    def _audit(
        self,
        request_hash: str,
        payload: dict,
        decision: RouteDecision | None,
        forwarded_model: str | None,
        fallback_used: bool,
        upstream_status: int | None,
        ok: bool,
        started: float,
        error: str | None,
    ) -> None:
        cfg = self.server.cfg
        audit_path = cfg.audit_path
        audit_path.parent.mkdir(parents=True, exist_ok=True)
        original_model = str(payload.get("model") or "")
        row = {
            "ts": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
            "request_hash": request_hash,
            "turn_correlation_id": str(
                ((payload.get("metadata") or {}).get("hermes_turn_correlation_id") or "")
            ) or None,
            "original_model": original_model,
            "route": decision.route if decision else None,
            "selected_model": decision.selected_model if decision else None,
            "forwarded_model": forwarded_model,
            "fallback_used": bool(fallback_used),
            "reason": decision.reason if decision else None,
            "upstream_status": upstream_status,
            "ok": bool(ok),
            "latency_ms": round((time.time() - started) * 1000),
            "error_set": bool(error),
            "error_type": error.split(":", 1)[0] if error else None,
        }
        with audit_path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default=str(DEFAULT_CONFIG))
    args = ap.parse_args(argv)
    cfg = RouterConfig(pathlib.Path(args.config))
    server = RouterServer((cfg.listen_host, cfg.listen_port), Handler, cfg)
    print(f"model-router-proxy listening on http://{cfg.listen_host}:{cfg.listen_port}", flush=True)
    server.serve_forever()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
