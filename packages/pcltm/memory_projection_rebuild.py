"""Destructive rebuild of derived governed-memory projections only."""

from __future__ import annotations

import json
import os
import stat
from pathlib import Path
from typing import Callable

from .projection_outbox import enqueue_memory_projections
from .projections.memory_runtime import drain_memory_projections
from .store import EventStore


FileIdentity = tuple[int, int, int]


def _path_identity(path: Path) -> FileIdentity:
    info = path.stat(follow_symlinks=False)
    if path.is_symlink() or int(getattr(info, "st_file_attributes", 0)) & 0x400:
        raise RuntimeError("unmanaged_memfs_projection_file")
    return int(info.st_dev), int(info.st_ino), int(info.st_mode)


def _validate_root_path(root: Path) -> None:
    """Reject every existing or dangling reparse component without resolving it."""
    for path in (root, *root.parents):
        try:
            _path_identity(path)
        except FileNotFoundError:
            # ``exists()`` is false for dangling symlinks, but ``is_symlink()``
            # still exposes them and they must remain fail-closed.
            if path.is_symlink():
                raise RuntimeError("unmanaged_memfs_projection_file")


def _require_identity(path: Path, expected: FileIdentity) -> None:
    try:
        actual = _path_identity(path)
    except (FileNotFoundError, OSError) as exc:
        raise RuntimeError("memfs_projection_path_changed") from exc
    if actual != expected:
        raise RuntimeError("memfs_projection_path_changed")


def _delete_verified_file(
    path: Path,
    expected: FileIdentity,
    *,
    fault_hook: Callable[[str], None] | None = None,
) -> None:
    """Delete the verified file object, not a later replacement at its path."""
    if os.name == "nt":
        import ctypes
        import msvcrt

        kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)
        create_file = kernel32.CreateFileW
        create_file.argtypes = [
            ctypes.c_wchar_p, ctypes.c_uint32, ctypes.c_uint32,
            ctypes.c_void_p, ctypes.c_uint32, ctypes.c_uint32, ctypes.c_void_p,
        ]
        create_file.restype = ctypes.c_void_p
        handle = create_file(
            str(path),
            0x80000000 | 0x00010000,  # GENERIC_READ | DELETE
            0x00000001 | 0x00000002 | 0x00000004,  # share read/write/delete
            None, 3, 0x00200000, None,  # OPEN_EXISTING | OPEN_REPARSE_POINT
        )
        if handle == ctypes.c_void_p(-1).value:
            raise OSError(ctypes.get_last_error(), "CreateFileW failed")
        descriptor = -1
        try:
            descriptor = msvcrt.open_osfhandle(int(handle), os.O_RDONLY)
            info = os.fstat(descriptor)
            actual = int(info.st_dev), int(info.st_ino), int(info.st_mode)
            if actual != expected or not stat.S_ISREG(actual[2]):
                raise RuntimeError("memfs_projection_path_changed")
            if fault_hook is not None:
                fault_hook("stale_after_open")

            class FileDispositionInfo(ctypes.Structure):
                _fields_ = [("DeleteFile", ctypes.c_bool)]

            disposition = FileDispositionInfo(True)
            set_information = kernel32.SetFileInformationByHandle
            set_information.argtypes = [
                ctypes.c_void_p, ctypes.c_int, ctypes.c_void_p, ctypes.c_uint32,
            ]
            set_information.restype = ctypes.c_int
            if not set_information(
                handle, 4, ctypes.byref(disposition), ctypes.sizeof(disposition),
            ):
                raise OSError(
                    ctypes.get_last_error(), "SetFileInformationByHandle failed",
                )
        finally:
            if descriptor >= 0:
                os.close(descriptor)
            else:
                kernel32.CloseHandle(handle)
        return

    descriptor = os.open(path, os.O_RDONLY)
    try:
        info = os.fstat(descriptor)
        actual = int(info.st_dev), int(info.st_ino), int(info.st_mode)
        if actual != expected or not stat.S_ISREG(actual[2]):
            raise RuntimeError("memfs_projection_path_changed")
        if fault_hook is not None:
            fault_hook("stale_after_open")
        _require_identity(path, expected)
        path.unlink()
    finally:
        os.close(descriptor)


def _validate_claims_directory(
    root: Path,
) -> tuple[Path | None, FileIdentity | None, list[tuple[Path, FileIdentity]]]:
    _validate_root_path(root)
    claims = root / "claims"
    try:
        claims_identity = _path_identity(claims)
    except FileNotFoundError:
        if claims.is_symlink():
            raise RuntimeError("unmanaged_memfs_projection_file")
        return None, None, []
    if not stat.S_ISDIR(claims_identity[2]):
        raise RuntimeError("unmanaged_memfs_projection_file")
    managed: list[tuple[Path, FileIdentity]] = []
    for path in claims.iterdir():
        identity = _path_identity(path)
        if (
            not stat.S_ISREG(identity[2])
            or path.suffix != ".md"
            or len(path.stem) != 16
            or not path.stem.isascii()
            or not path.stem.isdecimal()
            or int(path.stem) <= 0
        ):
            raise RuntimeError("unmanaged_memfs_projection_file")
        managed.append((path, identity))
    return claims, claims_identity, managed


def _active_claim_rows(conn) -> tuple[dict[int, list], list]:
    rows = conn.execute(
        """
        SELECT mc.claim_id, v.version, v.content_sha256, s.source_kind,
               s.event_id, s.legacy_record_id
        FROM memory_current mc
        JOIN memory_claim_versions v ON v.claim_version_id = mc.claim_version_id
        JOIN memory_claim_sources s ON s.claim_version_id = mc.claim_version_id
        WHERE mc.lifecycle_state = 'active'
        ORDER BY mc.claim_id, s.claim_source_id
        """
    ).fetchall()
    by_claim: dict[int, list] = {}
    for row in rows:
        by_claim.setdefault(int(row["claim_id"]), []).append(row)
    for claim_rows in by_claim.values():
        if len(claim_rows) != 1:
            raise RuntimeError("projection_rebuild_source_ambiguous")
    return by_claim, rows


def _require_no_projection_guards(conn) -> None:
    active_guards = int(conn.execute(
        "SELECT count(*) FROM memory_projection_guards"
    ).fetchone()[0])
    if active_guards:
        raise RuntimeError("projection_rebuild_guard_active")


def _cleanup_stale_memfs_files(
    store: EventStore,
    *,
    root: Path,
    fault_hook: Callable[[str], None] | None,
) -> None:
    """Delete only non-active claim files after active projections converged."""
    conn = store._conn
    try:
        conn.execute("BEGIN IMMEDIATE")
        _require_no_projection_guards(conn)
        active_ids = {
            int(row[0]) for row in conn.execute(
                "SELECT claim_id FROM memory_current WHERE lifecycle_state = 'active'"
            )
        }
        claims, claims_identity, managed_files = _validate_claims_directory(root)
        if claims is not None and claims_identity is not None:
            _require_identity(claims, claims_identity)
        for path, identity in managed_files:
            if int(path.stem) in active_ids:
                continue
            if claims is None or claims_identity is None or path.parent != claims:
                raise RuntimeError("memfs_projection_path_changed")
            _require_identity(claims, claims_identity)
            _delete_verified_file(path, identity, fault_hook=fault_hook)
        if fault_hook is not None:
            fault_hook("memfs_after_delete")
        conn.commit()
    except BaseException:
        conn.rollback()
        raise


def rebuild_all_memory_projections(
    store: EventStore,
    *,
    memfs_root: Path,
    fault_hook: Callable[[str], None] | None = None,
) -> dict[str, int]:
    """Rebuild active projections, then remove only stale MemFS claim files."""
    conn = store._conn
    if conn.in_transaction:
        raise RuntimeError("projection_rebuild_requires_transaction_ownership")
    root = Path(memfs_root)
    try:
        conn.execute("BEGIN IMMEDIATE")
        by_claim, _ = _active_claim_rows(conn)
        _require_no_projection_guards(conn)
        _validate_claims_directory(root)
        exists = conn.execute(
            "SELECT 1 FROM sqlite_master WHERE type='table' AND name='memory_fts'"
        ).fetchone()
        if exists is not None:
            conn.execute("DELETE FROM memory_fts")
        conn.execute(
            """
            DELETE FROM projection_outbox
            WHERE projection_kind IN ('memory_fts', 'memory_memfs')
            """
        )
        for claim_id, claim_rows in by_claim.items():
            row = claim_rows[0]
            source_kind = str(row["source_kind"])
            if source_kind == "event":
                event_id = int(row["event_id"])
                authority_kind, authority_id = "event", str(event_id)
            elif source_kind == "legacy_record":
                event_id = None
                authority_kind = "legacy_record"
                authority_id = str(int(row["legacy_record_id"]))
            else:
                raise RuntimeError("projection_rebuild_source_unsupported")
            enqueue_memory_projections(
                conn,
                event_id=event_id,
                authority_kind=authority_kind,
                authority_id=authority_id,
                aggregate_id=f"memory:{claim_id}",
                aggregate_version=int(row["version"]),
                payload_sha256=str(row["content_sha256"]),
            )
        if fault_hook is not None:
            fault_hook("authority_before_commit")
        conn.commit()
    except BaseException as operation_exc:
        if conn.in_transaction:
            conn.rollback()
        else:
            try:
                drain_memory_projections(store, memfs_root=root)
            except BaseException as convergence_exc:
                raise RuntimeError(
                    "projection_rebuild_post_commit_convergence_failed"
                ) from convergence_exc
            raise operation_exc
        raise

    if fault_hook is not None:
        fault_hook("authority_after_commit")

    applied = drain_memory_projections(store, memfs_root=root)
    _cleanup_stale_memfs_files(store, root=root, fault_hook=fault_hook)
    return {
        "claims": len(by_claim),
        "memory_fts": applied["memory_fts"],
        "memory_memfs": applied["memory_memfs"],
    }
