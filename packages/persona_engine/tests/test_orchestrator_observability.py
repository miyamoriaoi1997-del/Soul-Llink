from __future__ import annotations

import json
import multiprocessing
import os
import threading
from pathlib import Path

from persona_engine.persona_orchestrator.observability import OrchestratorLogger


def _write_from_process(path: str, worker: int, result_queue) -> None:
    logger = OrchestratorLogger(path, max_bytes=512, backup_count=100)
    for sequence in range(50):
        logger.log({"worker": worker, "sequence": sequence})
    result_queue.put(logger.status()["write_failures"])


def test_orchestrator_logger_preserves_every_concurrent_jsonl_record(tmp_path: Path) -> None:
    path = tmp_path / "orchestrator.jsonl"
    logger = OrchestratorLogger(path, max_bytes=0)
    writes_per_thread = 150
    thread_count = 8

    def write_records(worker: int) -> None:
        for sequence in range(writes_per_thread):
            logger.log({"worker": worker, "sequence": sequence, "payload": "x" * 4096})

    threads = [threading.Thread(target=write_records, args=(worker,)) for worker in range(thread_count)]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()

    rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]
    observed = {(row["packet"]["worker"], row["packet"]["sequence"]) for row in rows}

    assert len(rows) == thread_count * writes_per_thread
    assert len(observed) == thread_count * writes_per_thread
    assert logger.status()["write_failures"] == 0


def test_orchestrator_logger_rotates_without_losing_current_log_validity(tmp_path: Path) -> None:
    path = tmp_path / "orchestrator.jsonl"
    logger = OrchestratorLogger(path, max_bytes=1024, backup_count=2)

    for sequence in range(30):
        logger.log({"sequence": sequence, "payload": "x" * 300})

    files = [candidate for candidate in (path, path.with_suffix(path.suffix + ".1"), path.with_suffix(path.suffix + ".2")) if candidate.exists()]

    assert len(files) == 3
    assert all(json.loads(line) for file in files for line in file.read_text(encoding="utf-8").splitlines())
    assert logger.status()["rotation_count"] > 0


def test_orchestrator_logger_exposes_write_failures(tmp_path: Path) -> None:
    directory_as_log = tmp_path / "not-a-file"
    directory_as_log.mkdir()
    logger = OrchestratorLogger(directory_as_log)

    logger.log({"mode": "work"})

    status = logger.status()
    assert status["write_failures"] == 1
    assert status["last_error_type"] in {"IsADirectoryError", "PermissionError"}
    assert status["healthy"] is False


def test_multiple_logger_instances_share_one_lock(tmp_path: Path) -> None:
    path = tmp_path / "shared.jsonl"
    loggers = [OrchestratorLogger(path, max_bytes=0) for _ in range(4)]
    threads = [
        threading.Thread(
            target=lambda logger=logger, worker=worker: [
                logger.log({"worker": worker, "sequence": sequence}) for sequence in range(100)
            ]
        )
        for worker, logger in enumerate(loggers)
    ]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()

    rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]
    assert len(rows) == 400
    assert sum(logger.status()["write_failures"] for logger in loggers) == 0


def test_unserializable_packet_is_counted_without_escaping(tmp_path: Path) -> None:
    logger = OrchestratorLogger(tmp_path / "o.jsonl")

    assert logger.log({"bad": object()}) is False
    assert logger.status()["write_failures"] == 1
    assert logger.status()["last_error_type"] == "TypeError"


def test_multiple_processes_rotate_without_loss(tmp_path: Path) -> None:
    path = tmp_path / "multiprocess.jsonl"
    context = multiprocessing.get_context("spawn")
    result_queue = context.Queue()
    processes = [context.Process(target=_write_from_process, args=(str(path), worker, result_queue)) for worker in range(4)]
    for process in processes:
        process.start()
    for process in processes:
        process.join(timeout=30)
        assert process.exitcode == 0

    assert [result_queue.get(timeout=5) for _ in processes] == [0, 0, 0, 0]
    rows = []
    for candidate in [path, *sorted(tmp_path.glob("multiprocess.jsonl.*"))]:
        if candidate.name.endswith(".lock"):
            continue
        rows.extend(json.loads(line) for line in candidate.read_text(encoding="utf-8").splitlines())
    assert len(rows) == 200
    assert len({(row["packet"]["worker"], row["packet"]["sequence"]) for row in rows}) == 200


def test_windows_lock_retries_transient_contention(tmp_path: Path, monkeypatch) -> None:
    if os.name != "nt":
        return
    import msvcrt

    real_locking = msvcrt.locking
    attempts = 0

    def transient_locking(fd, mode, size):
        nonlocal attempts
        if mode == msvcrt.LK_NBLCK and attempts < 2:
            attempts += 1
            raise OSError("simulated contention")
        return real_locking(fd, mode, size)

    monkeypatch.setattr(msvcrt, "locking", transient_locking)
    logger = OrchestratorLogger(tmp_path / "retry.jsonl")

    assert logger.log({"sequence": 1}) is True
    assert attempts == 2
    assert logger.status()["write_failures"] == 0
