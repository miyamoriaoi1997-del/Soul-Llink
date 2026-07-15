"""Thread-safe cached composition of monitoring collectors."""

from __future__ import annotations

import threading
import time
from datetime import datetime, timezone
from typing import Any, Callable, Mapping


class SnapshotService:
    def __init__(self, collectors: Mapping[str, Callable[[], dict[str, Any]]], *, ttl_seconds: float = 2.0):
        self.collectors = dict(collectors)
        self.ttl_seconds = max(0.0, float(ttl_seconds))
        self._lock = threading.Lock()
        self._cached: dict[str, Any] | None = None
        self._cached_at = 0.0

    def get(self) -> dict[str, Any]:
        now = time.monotonic()
        if self._cached is not None and now - self._cached_at <= self.ttl_seconds:
            return self._cached
        with self._lock:
            now = time.monotonic()
            if self._cached is not None and now - self._cached_at <= self.ttl_seconds:
                return self._cached
            started = time.perf_counter()
            result: dict[str, Any] = {
                "schema_version": 2,
                "runtime": {}, "context": {}, "memory": {}, "persona": {}, "router": {},
                "emotion": {}, "memory_bodies": {}, "injection": {},
                "runtime_capture": {}, "soul": {},
                "issues": [],
            }
            for name, collector in self.collectors.items():
                try:
                    payload = collector()
                    if isinstance(payload, dict):
                        for section in (
                            "runtime", "context", "memory", "persona", "router",
                            "emotion", "memory_bodies", "injection",
                            "runtime_capture", "soul",
                        ):
                            if isinstance(payload.get(section), dict):
                                result[section] = payload[section]
                        if isinstance(payload.get("issues"), list):
                            result["issues"].extend(payload["issues"])
                except Exception as exc:
                    result["issues"].append({
                        "severity": "error", "code": f"{name.upper()}_COLLECTOR_FAILED", "source": name,
                        "message": f"{type(exc).__name__}: collector failed", "timestamp": datetime.now(timezone.utc).isoformat(),
                        "remediation": f"Inspect the {name} collector.",
                    })
            result["generated_at"] = datetime.now(timezone.utc).isoformat()
            result["duration_ms"] = round((time.perf_counter() - started) * 1000)
            result["ok"] = not any(issue.get("severity") == "error" for issue in result["issues"])
            result["issues"].sort(key=lambda item: {"error": 0, "warning": 1, "info": 2}.get(item.get("severity"), 3))
            self._cached = result
            self._cached_at = time.monotonic()
            return result


__all__ = ["SnapshotService"]
