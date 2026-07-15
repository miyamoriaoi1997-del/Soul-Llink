"""Stable filesystem snapshots for read-only monitoring of live SQLite WAL databases."""

from __future__ import annotations

from contextlib import contextmanager
import hashlib
from pathlib import Path
import shutil
import tempfile
import time
from typing import Iterator


def _digest(path: Path) -> tuple[int, str] | None:
    if not path.is_file():
        return None
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return path.stat().st_size, digest.hexdigest()


def _source_fingerprint(db: Path) -> tuple[tuple[int, str] | None, tuple[int, str] | None]:
    return _digest(db), _digest(Path(str(db) + "-wal"))


@contextmanager
def stable_sqlite_snapshot(
    db_path: str | Path,
    *,
    max_attempts: int = 5,
    retry_delay_seconds: float = 0.01,
) -> Iterator[Path]:
    """Yield a consistent temporary DB+WAL copy without opening the source DB.

    The source DB and WAL are hashed before and after copying. A checkpoint or
    commit racing the copy changes at least one digest and causes a retry. SHM
    is deliberately not copied: SQLite reconstructs lock/index state beside the
    temporary database, so monitoring never opens or mutates production SHM.
    """
    source = Path(db_path).resolve()
    if not source.is_file():
        raise FileNotFoundError(source)
    attempts = max(1, int(max_attempts))
    with tempfile.TemporaryDirectory(prefix="soullink-monitor-") as directory:
        target = Path(directory) / source.name
        for attempt in range(attempts):
            before = _source_fingerprint(source)
            shutil.copy2(source, target)
            source_wal = Path(str(source) + "-wal")
            target_wal = Path(str(target) + "-wal")
            target_wal.unlink(missing_ok=True)
            if source_wal.is_file():
                shutil.copy2(source_wal, target_wal)
            after = _source_fingerprint(source)
            if before == after:
                yield target
                return
            target.unlink(missing_ok=True)
            target_wal.unlink(missing_ok=True)
            if attempt + 1 < attempts:
                time.sleep(max(0.0, float(retry_delay_seconds)))
    raise RuntimeError("live SQLite database changed during every snapshot attempt")


__all__ = ["stable_sqlite_snapshot"]
