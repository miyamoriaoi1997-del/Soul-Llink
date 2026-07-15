"""Shadow-mode observability for persona orchestration decisions."""

from __future__ import annotations

import json
import os
import threading
import time
from contextlib import contextmanager
from dataclasses import asdict, is_dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Iterator


_PATH_LOCKS: dict[str, threading.Lock] = {}
_PATH_LOCKS_GUARD = threading.Lock()


def _shared_thread_lock(path: Path) -> threading.Lock:
    key = os.path.normcase(str(path.resolve()))
    with _PATH_LOCKS_GUARD:
        return _PATH_LOCKS.setdefault(key, threading.Lock())


@contextmanager
def _interprocess_lock(path: Path, *, timeout_seconds: float = 15.0) -> Iterator[None]:
    """Serialize rotation and append across processes on Windows and POSIX."""
    lock_path = path.with_suffix(path.suffix + ".lock")
    lock_path.parent.mkdir(parents=True, exist_ok=True)
    with lock_path.open("a+b") as stream:
        stream.seek(0, os.SEEK_END)
        if stream.tell() == 0:
            stream.write(b"0")
            stream.flush()
        stream.seek(0)
        if os.name == "nt":
            import msvcrt

            deadline = time.monotonic() + max(0.0, timeout_seconds)
            while True:
                try:
                    stream.seek(0)
                    msvcrt.locking(stream.fileno(), msvcrt.LK_NBLCK, 1)
                    break
                except OSError:
                    if time.monotonic() >= deadline:
                        raise
                    time.sleep(0.05)
            try:
                yield
            finally:
                stream.seek(0)
                msvcrt.locking(stream.fileno(), msvcrt.LK_UNLCK, 1)
        else:
            import fcntl

            fcntl.flock(stream.fileno(), fcntl.LOCK_EX)
            try:
                yield
            finally:
                fcntl.flock(stream.fileno(), fcntl.LOCK_UN)


class OrchestratorLogger:
    """Append orchestration decisions as bounded, thread-safe JSONL."""

    def __init__(
        self,
        log_path: str | Path,
        *,
        max_bytes: int = 2 * 1024 * 1024,
        backup_count: int = 2,
    ):
        self.log_path = Path(log_path)
        self.max_bytes = max(0, int(max_bytes))
        self.backup_count = max(0, int(backup_count))
        self._lock = _shared_thread_lock(self.log_path)
        self._write_failures = 0
        self._rotation_count = 0
        self._last_error_type = ""
        self._last_error = ""

    def log(self, packet: Any, extra: dict | None = None) -> bool:
        try:
            record = {
                "timestamp": datetime.now().isoformat(),
                "packet": asdict(packet) if is_dataclass(packet) else packet,
                "extra": extra or {},
            }
            line = json.dumps(record, ensure_ascii=False, sort_keys=True) + "\n"
            with self._lock:
                with _interprocess_lock(self.log_path):
                    self.log_path.parent.mkdir(parents=True, exist_ok=True)
                    self._rotate_if_needed(len(line.encode("utf-8")))
                    with self.log_path.open("a", encoding="utf-8") as stream:
                        stream.write(line)
            return True
        except Exception as exc:
            with self._lock:
                self._write_failures += 1
                self._last_error_type = type(exc).__name__
                self._last_error = str(exc)[:240]
            return False

    def status(self) -> dict[str, Any]:
        with self._lock:
            return {
                "healthy": self._write_failures == 0,
                "write_failures": self._write_failures,
                "rotation_count": self._rotation_count,
                "last_error_type": self._last_error_type,
                "last_error": self._last_error,
                "log_path": str(self.log_path),
            }

    def _rotate_if_needed(self, incoming_bytes: int) -> None:
        if self.max_bytes <= 0 or self.backup_count <= 0 or not self.log_path.exists():
            return
        if self.log_path.stat().st_size + incoming_bytes <= self.max_bytes:
            return
        oldest = self.log_path.with_suffix(self.log_path.suffix + f".{self.backup_count}")
        if oldest.exists():
            oldest.unlink()
        for index in range(self.backup_count - 1, 0, -1):
            source = self.log_path.with_suffix(self.log_path.suffix + f".{index}")
            if source.exists():
                source.replace(self.log_path.with_suffix(self.log_path.suffix + f".{index + 1}"))
        self.log_path.replace(self.log_path.with_suffix(self.log_path.suffix + ".1"))
        self._rotation_count += 1
