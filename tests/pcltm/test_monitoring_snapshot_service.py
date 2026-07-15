from __future__ import annotations

import threading
import time

from pcltm.monitoring.snapshot import SnapshotService


def test_snapshot_service_caches_and_shares_concurrent_refresh() -> None:
    calls = 0
    lock = threading.Lock()

    def collector():
        nonlocal calls
        with lock:
            calls += 1
        time.sleep(0.03)
        return {"runtime": {"status": "healthy"}, "issues": []}

    service = SnapshotService({"core": collector}, ttl_seconds=2)
    results = []
    threads = [threading.Thread(target=lambda: results.append(service.get())) for _ in range(20)]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()

    assert calls == 1
    assert len(results) == 20
    assert all(result["runtime"]["status"] == "healthy" for result in results)


def test_snapshot_service_keeps_other_sections_when_collector_fails() -> None:
    service = SnapshotService(
        {
            "runtime": lambda: {"runtime": {"status": "healthy"}},
            "context": lambda: (_ for _ in ()).throw(RuntimeError("opaque failure")),
        }
    )

    result = service.get()

    assert result["runtime"]["status"] == "healthy"
    assert result["context"] == {}
    assert result["issues"][0]["code"] == "CONTEXT_COLLECTOR_FAILED"
    assert result["ok"] is False


def test_snapshot_service_exposes_runtime_capture_and_soul_sections() -> None:
    service = SnapshotService(
        {
            "capture": lambda: {"runtime_capture": {"source": "exact_host_capture"}},
            "soul": lambda: {"soul": {"source": "runtime_soul_files"}},
        },
        ttl_seconds=0,
    )

    result = service.get()

    assert result["runtime_capture"]["source"] == "exact_host_capture"
    assert result["soul"]["source"] == "runtime_soul_files"
