"""Bodyless backup bundle and empty-directory restore rehearsal for governed memory."""

from __future__ import annotations

import hashlib
import json
import shutil
import sqlite3
import subprocess
from pathlib import Path, PurePosixPath
from typing import Callable, Mapping

from .legacy_shadow_migration import create_readonly_sqlite_snapshot
from .memory_projection_rebuild import rebuild_all_memory_projections
from .projections.memory_runtime import require_memory_projections_applied
from .store import EventStore


def _path_is_link_or_reparse(path: Path) -> bool:
    if path.is_symlink():
        return True
    try:
        return bool(int(getattr(path.stat(), "st_file_attributes", 0)) & 0x400)
    except FileNotFoundError:
        return False


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _git_head(repository_root: Path) -> str:
    try:
        return subprocess.check_output(
            ["git", "-C", str(repository_root), "rev-parse", "HEAD"],
            text=True,
            stderr=subprocess.DEVNULL,
        ).strip()
    except (OSError, subprocess.CalledProcessError) as exc:
        raise RuntimeError("restore_git_head_unavailable") from exc


def _relative_artifact_path(raw: str) -> Path:
    normalized = str(raw).replace("\\", "/")
    pure = PurePosixPath(normalized)
    if (
        not normalized
        or pure.is_absolute()
        or any(part in {"", ".", ".."} for part in pure.parts)
        or ":" in pure.parts[0]
    ):
        raise ValueError("restore_artifact_path_invalid")
    return Path(*pure.parts)


def _load_manifest(bundle: Path) -> dict[str, object]:
    try:
        value = json.loads((bundle / "manifest.json").read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise RuntimeError("restore_bundle_manifest_invalid") from exc
    if type(value) is not dict or value.get("schema_version") != 1 or value.get("bodyless") is not True:
        raise RuntimeError("restore_bundle_manifest_invalid")
    return value


def _bundle_artifact(bundle: Path, relative: Path) -> Path:
    current = bundle
    for part in relative.parts:
        current = current / part
        if _path_is_link_or_reparse(current):
            raise RuntimeError("restore_bundle_symlink_forbidden")
    try:
        current.resolve().relative_to(bundle.resolve())
    except ValueError as exc:
        raise RuntimeError("restore_bundle_symlink_forbidden") from exc
    return current


def create_memory_restore_bundle(
    *,
    source_db: str | Path,
    config_files: Mapping[str, str | Path],
    repository_root: str | Path,
    bundle_root: str | Path,
) -> dict[str, object]:
    """Create a new bodyless restore bundle without mutating the source database."""
    source = Path(source_db).resolve()
    repository = Path(repository_root).resolve()
    bundle = Path(bundle_root).resolve()
    if bundle.exists():
        raise FileExistsError(bundle)
    config_entries: list[tuple[str, Path, Path]] = []
    for raw_name, raw_source in config_files.items():
        relative = _relative_artifact_path(str(raw_name))
        config_source = Path(raw_source).resolve()
        if not config_source.is_file():
            raise FileNotFoundError(config_source)
        config_entries.append((relative.as_posix(), relative, config_source))
    staging = bundle.with_name(f".{bundle.name}.building")
    if staging.exists():
        raise FileExistsError(staging)
    try:
        (staging / "database").mkdir(parents=True)
        snapshot = staging / "database" / "pcltm.db"
        receipt = create_readonly_sqlite_snapshot(source, snapshot)
        config_hashes: dict[str, str] = {}
        for name, relative, config_source in sorted(config_entries):
            destination = staging / "configs" / relative
            destination.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(config_source, destination)
            config_hashes[name] = _sha256(destination)
        manifest: dict[str, object] = {
            "schema_version": 1,
            "bodyless": True,
            "git_head": _git_head(repository),
            "database": {
                "path": "database/pcltm.db",
                "sha256": receipt.snapshot_sha256,
                "quick_check": receipt.quick_check,
                "source_sha256_before": receipt.source_sha256_before,
                "source_sha256_after": receipt.source_sha256_after,
                "source_query_only": receipt.source_query_only,
            },
            "configs": dict(sorted(config_hashes.items())),
        }
        (staging / "manifest.json").write_text(
            json.dumps(manifest, ensure_ascii=False, sort_keys=True, separators=(",", ":")) + "\n",
            encoding="utf-8",
        )
        staging.replace(bundle)
        return manifest
    except BaseException:
        shutil.rmtree(staging, ignore_errors=True)
        raise


def _verify_bundle(bundle: Path, repository: Path) -> dict[str, object]:
    if _path_is_link_or_reparse(bundle):
        raise RuntimeError("restore_bundle_symlink_forbidden")
    manifest = _load_manifest(bundle)
    if str(manifest.get("git_head") or "") != _git_head(repository):
        raise RuntimeError("restore_git_head_mismatch")
    database = manifest.get("database")
    configs = manifest.get("configs")
    if type(database) is not dict or type(configs) is not dict:
        raise RuntimeError("restore_bundle_manifest_invalid")
    db_relative = _relative_artifact_path(str(database.get("path") or ""))
    db_artifact = _bundle_artifact(bundle, db_relative)
    expected_db_hash = database.get("sha256")
    if _path_is_link_or_reparse(db_artifact):
        raise RuntimeError("restore_bundle_symlink_forbidden")
    if not db_artifact.is_file() or _sha256(db_artifact) != expected_db_hash:
        raise RuntimeError("restore_bundle_hash_mismatch")
    connection = sqlite3.connect(f"file:{db_artifact.resolve().as_posix()}?mode=ro", uri=True)
    try:
        connection.execute("PRAGMA query_only=ON")
        if str(connection.execute("PRAGMA quick_check").fetchone()[0]) != "ok":
            raise RuntimeError("restore_bundle_integrity_failed")
    finally:
        connection.close()
    for raw_name, expected_hash in configs.items():
        relative = _relative_artifact_path(str(raw_name))
        artifact = _bundle_artifact(bundle, Path("configs") / relative)
        if _path_is_link_or_reparse(artifact):
            raise RuntimeError("restore_bundle_symlink_forbidden")
        if type(expected_hash) is not str or not artifact.is_file() or _sha256(artifact) != expected_hash:
            raise RuntimeError("restore_bundle_hash_mismatch")
    return manifest


def restore_memory_bundle_into_empty_directory(
    *,
    bundle_root: str | Path,
    restore_root: str | Path,
    repository_root: str | Path,
    fault_hook: Callable[[str], None] | None = None,
) -> dict[str, int | str]:
    """Restore DB/config bytes, rebuild derived projections, and remove partial output on fault."""
    raw_target = Path(restore_root).absolute()
    raw_bundle = Path(bundle_root).absolute()
    if _path_is_link_or_reparse(raw_target):
        raise RuntimeError("restore_destination_symlink_forbidden")
    if _path_is_link_or_reparse(raw_bundle):
        raise RuntimeError("restore_bundle_symlink_forbidden")
    bundle = raw_bundle.resolve()
    target = raw_target.resolve()
    repository = Path(repository_root).resolve()
    if target.exists() and (not target.is_dir() or any(target.iterdir())):
        raise RuntimeError("restore_destination_not_empty")
    manifest = _verify_bundle(bundle, repository)
    database = manifest["database"]
    configs = manifest["configs"]
    created_target = not target.exists()
    store: EventStore | None = None
    try:
        target.mkdir(parents=True, exist_ok=True)
        restored_db = target / "var" / "pcltm.db"
        restored_db.parent.mkdir(parents=True)
        shutil.copyfile(bundle / _relative_artifact_path(str(database["path"])), restored_db)
        if fault_hook is not None:
            fault_hook("after_database_copy")
        for raw_name in sorted(configs):
            relative = _relative_artifact_path(str(raw_name))
            destination = target / relative
            destination.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(bundle / "configs" / relative, destination)
        if fault_hook is not None:
            fault_hook("after_config_copy")
        store = EventStore(restored_db)
        rebuild = rebuild_all_memory_projections(store, memfs_root=target / "memfs")
        claim_ids = [int(row[0]) for row in store._conn.execute(
            "SELECT claim_id FROM memory_current WHERE lifecycle_state = 'active' ORDER BY claim_id"
        )]
        for claim_id in claim_ids:
            require_memory_projections_applied(
                store, memfs_root=target / "memfs", claim_id=claim_id,
            )
        quick_check = str(store._conn.execute("PRAGMA quick_check").fetchone()[0])
        if quick_check != "ok":
            raise RuntimeError("restore_integrity_failed")
        if fault_hook is not None:
            fault_hook("after_verify")
        return {
            "quick_check": quick_check,
            "git_head": str(manifest["git_head"]),
            "authority_claims": len(claim_ids),
            "memory_fts": int(rebuild["memory_fts"]),
            "memory_memfs": int(rebuild["memory_memfs"]),
        }
    except BaseException:
        if store is not None:
            store.close()
            store = None
        # Remove only artifacts created by this call; preserve a caller-owned empty directory.
        if target.exists():
            if created_target:
                shutil.rmtree(target)
            else:
                for child in target.iterdir():
                    if _path_is_link_or_reparse(child) or child.is_file():
                        child.unlink()
                    else:
                        shutil.rmtree(child)
        raise
    finally:
        if store is not None:
            store.close()


def _tree_manifest(root: Path) -> dict[str, str]:
    if not root.is_dir() or _path_is_link_or_reparse(root):
        raise RuntimeError("restore_switch_tree_invalid")
    manifest: dict[str, str] = {}
    for path in sorted(root.rglob("*"), key=lambda item: item.as_posix()):
        if _path_is_link_or_reparse(path):
            raise RuntimeError("restore_switch_tree_invalid")
        relative = path.relative_to(root).as_posix()
        if path.is_file():
            manifest[f"file:{relative}"] = _sha256(path)
        elif path.is_dir():
            manifest[f"dir:{relative}"] = hashlib.sha256(b"pcltm-directory-v1").hexdigest()
        else:
            raise RuntimeError("restore_switch_tree_invalid")
    return manifest


def _tree_sha256(root: Path) -> str:
    payload = json.dumps(
        _tree_manifest(root), sort_keys=True, separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def _path_or_ancestor_is_link_or_reparse(path: Path) -> bool:
    """Reject a raw path when it or any lexical ancestor is redirected."""
    raw = path.absolute()
    return any(_path_is_link_or_reparse(item) for item in (raw, *raw.parents))


def switch_memory_restore_with_rollback(
    *,
    restored_root: str | Path,
    live_root: str | Path,
    rollback_root: str | Path,
    fault_hook: Callable[[str], None] | None = None,
) -> dict[str, object]:
    """Atomically switch directory names and preserve a hash-bound rollback tree."""
    raw_restored = Path(restored_root).absolute()
    raw_live = Path(live_root).absolute()
    raw_rollback = Path(rollback_root).absolute()
    if any(
        _path_or_ancestor_is_link_or_reparse(path)
        for path in (raw_restored, raw_live, raw_rollback)
    ):
        raise RuntimeError("restore_switch_precondition_failed")
    restored = raw_restored.resolve()
    live = raw_live.resolve()
    rollback = raw_rollback.resolve()
    if (
        not restored.is_dir() or not live.is_dir() or rollback.exists()
        or restored.parent != live.parent or rollback.parent != live.parent
    ):
        raise RuntimeError("restore_switch_precondition_failed")
    if len({restored, live, rollback}) != 3:
        raise RuntimeError("restore_switch_precondition_failed")
    previous_hash = _tree_sha256(live)
    restored_hash = _tree_sha256(restored)
    moved_previous = False
    moved_restored = False
    try:
        live.replace(rollback)
        moved_previous = True
        if fault_hook is not None:
            fault_hook("after_live_backup")
        restored.replace(live)
        moved_restored = True
        if _tree_sha256(live) != restored_hash or _tree_sha256(rollback) != previous_hash:
            raise RuntimeError("restore_switch_hash_mismatch")
        if fault_hook is not None:
            fault_hook("after_switch")
        return {
            "schema_version": 1,
            "live_root": str(live),
            "rollback_root": str(rollback),
            "previous_tree_sha256": previous_hash,
            "restored_tree_sha256": restored_hash,
        }
    except BaseException:
        if moved_restored and live.exists():
            shutil.rmtree(live)
        if moved_previous and rollback.exists():
            rollback.replace(live)
        if not live.is_dir() or _tree_sha256(live) != previous_hash:
            raise RuntimeError("restore_switch_automatic_rollback_failed")
        raise


def rollback_memory_restore_switch(receipt: Mapping[str, object]) -> dict[str, object]:
    """Rollback one completed switch after verifying both receipt-bound trees."""
    required = {
        "schema_version", "live_root", "rollback_root",
        "previous_tree_sha256", "restored_tree_sha256",
    }
    if type(receipt) is not dict or set(receipt) != required or receipt.get("schema_version") != 1:
        raise RuntimeError("restore_switch_receipt_invalid")
    raw_live = Path(str(receipt["live_root"])).absolute()
    raw_rollback = Path(str(receipt["rollback_root"])).absolute()
    if (
        _path_or_ancestor_is_link_or_reparse(raw_live)
        or _path_or_ancestor_is_link_or_reparse(raw_rollback)
    ):
        raise RuntimeError("restore_switch_receipt_invalid")
    live = raw_live.resolve()
    rollback = raw_rollback.resolve()
    if not live.is_dir() or not rollback.is_dir() or live.parent != rollback.parent:
        raise RuntimeError("restore_switch_receipt_invalid")
    if _tree_sha256(live) != receipt["restored_tree_sha256"]:
        raise RuntimeError("restore_switch_live_hash_mismatch")
    if _tree_sha256(rollback) != receipt["previous_tree_sha256"]:
        raise RuntimeError("restore_switch_rollback_hash_mismatch")
    displaced = live.with_name(f".{live.name}.rollback-displaced")
    if displaced.exists() or _path_or_ancestor_is_link_or_reparse(displaced):
        raise RuntimeError("restore_switch_receipt_invalid")
    moved_live = False
    moved_rollback = False
    try:
        live.replace(displaced)
        moved_live = True
        rollback.replace(live)
        moved_rollback = True
        if _tree_sha256(live) != receipt["previous_tree_sha256"]:
            raise RuntimeError("restore_switch_rollback_verify_failed")
        shutil.rmtree(displaced)
    except BaseException:
        if moved_rollback and live.exists():
            live.replace(rollback)
        if moved_live and displaced.exists():
            displaced.replace(live)
        raise
    return {
        "rolled_back": True,
        "live_root": str(live),
        "tree_sha256": str(receipt["previous_tree_sha256"]),
    }


__all__ = [
    "create_memory_restore_bundle",
    "restore_memory_bundle_into_empty_directory",
    "switch_memory_restore_with_rollback",
    "rollback_memory_restore_switch",
]
