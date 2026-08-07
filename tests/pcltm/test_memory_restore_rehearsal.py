from __future__ import annotations

import hashlib
import json
import subprocess
from pathlib import Path

import pytest

from pcltm.memory_contracts import PersonaMode, Sensitivity
from pcltm.memory_restore_rehearsal import (
    create_memory_restore_bundle,
    restore_memory_bundle_into_empty_directory,
    rollback_memory_restore_switch,
    switch_memory_restore_with_rollback,
)
from pcltm.memory_write_service import MemoryWriteRequest, MemoryWriteService
from pcltm.projections.memory_runtime import drain_memory_projections
from pcltm.store import EventStore


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _git(root: Path, *args: str) -> str:
    return subprocess.check_output(
        ["git", "-C", str(root), *args], text=True,
    ).strip()


def _make_git_repo(root: Path) -> str:
    root.mkdir()
    _git(root, "init", "-q")
    _git(root, "config", "user.email", "restore-test@example.invalid")
    _git(root, "config", "user.name", "Restore Test")
    (root / "tracked.txt").write_text("tracked\n", encoding="utf-8")
    _git(root, "add", "tracked.txt")
    _git(root, "commit", "-qm", "fixture")
    return _git(root, "rev-parse", "HEAD")


def _write_claim(store: EventStore, content: str):
    return MemoryWriteService(store).write(MemoryWriteRequest(
        idempotency_key="restore-claim", content=content,
        canonical_key="profile:restore", target="profile",
        memory_type="preference", sensitivity=Sensitivity.NORMAL,
        mode_scope=(PersonaMode.DAILY,), injection_policy="allow",
    ))


def test_bundle_and_empty_directory_restore_rebuilds_derived_surfaces(tmp_path: Path) -> None:
    source_db = tmp_path / "source.db"
    source_memfs = tmp_path / "source-memfs"
    source = EventStore(source_db)
    try:
        receipt = _write_claim(source, "restore body sentinel")
        drain_memory_projections(source, memfs_root=source_memfs)
    finally:
        source.close()
    repository = tmp_path / "repo"
    head = _make_git_repo(repository)
    config = tmp_path / "runtime-config.json"
    config.write_text('{"authority":"pcltm"}\n', encoding="utf-8")
    bundle = tmp_path / "bundle"

    manifest = create_memory_restore_bundle(
        source_db=source_db,
        config_files={"config/runtime.json": config},
        repository_root=repository,
        bundle_root=bundle,
    )
    source_db_hash = _sha256(source_db)
    source_config_hash = _sha256(config)
    rendered = json.dumps(manifest, sort_keys=True)
    restore_root = tmp_path / "empty-restore"
    result = restore_memory_bundle_into_empty_directory(
        bundle_root=bundle,
        restore_root=restore_root,
        repository_root=repository,
    )

    assert manifest["bodyless"] is True
    assert manifest["git_head"] == head
    assert "restore body sentinel" not in rendered
    assert manifest["database"]["source_sha256_before"] == source_db_hash
    assert manifest["database"]["source_sha256_after"] == source_db_hash
    assert manifest["database"]["sha256"] == _sha256(bundle / "database" / "pcltm.db")
    assert manifest["configs"] == {"config/runtime.json": source_config_hash}
    assert result["quick_check"] == "ok"
    assert result["git_head"] == head
    assert result["authority_claims"] == 1
    assert result["memory_fts"] == 1
    assert result["memory_memfs"] == 1
    restored_db = restore_root / "var" / "pcltm.db"
    assert _sha256(restored_db) != source_db_hash  # rebuild changes only derived DB state
    assert _sha256(restore_root / "config" / "runtime.json") == source_config_hash
    assert (restore_root / "memfs" / "claims" / f"{receipt.claim_id:016d}.md").is_file()
    assert _sha256(source_db) == source_db_hash


def test_restore_requires_truly_empty_destination(tmp_path: Path) -> None:
    source_db = tmp_path / "source.db"
    EventStore(source_db).close()
    repository = tmp_path / "repo"
    _make_git_repo(repository)
    config = tmp_path / "config.json"
    config.write_text("{}", encoding="utf-8")
    bundle = tmp_path / "bundle"
    create_memory_restore_bundle(
        source_db=source_db, config_files={"config.json": config},
        repository_root=repository, bundle_root=bundle,
    )
    target = tmp_path / "target"
    target.mkdir()
    sentinel = target / "do-not-touch.txt"
    sentinel.write_text("sentinel", encoding="utf-8")

    with pytest.raises(RuntimeError, match="restore_destination_not_empty"):
        restore_memory_bundle_into_empty_directory(
            bundle_root=bundle, restore_root=target, repository_root=repository,
        )
    assert sentinel.read_text(encoding="utf-8") == "sentinel"


def test_restore_rejects_tampered_bundle_before_creating_destination(tmp_path: Path) -> None:
    source_db = tmp_path / "source.db"
    EventStore(source_db).close()
    repository = tmp_path / "repo"
    _make_git_repo(repository)
    config = tmp_path / "config.json"
    config.write_text("{}", encoding="utf-8")
    bundle = tmp_path / "bundle"
    create_memory_restore_bundle(
        source_db=source_db, config_files={"config.json": config},
        repository_root=repository, bundle_root=bundle,
    )
    (bundle / "database" / "pcltm.db").write_bytes(b"tampered")
    target = tmp_path / "target"

    with pytest.raises(RuntimeError, match="restore_bundle_hash_mismatch"):
        restore_memory_bundle_into_empty_directory(
            bundle_root=bundle, restore_root=target, repository_root=repository,
        )
    assert not target.exists()


def test_restore_rejects_git_head_drift(tmp_path: Path) -> None:
    source_db = tmp_path / "source.db"
    EventStore(source_db).close()
    repository = tmp_path / "repo"
    _make_git_repo(repository)
    config = tmp_path / "config.json"
    config.write_text("{}", encoding="utf-8")
    bundle = tmp_path / "bundle"
    create_memory_restore_bundle(
        source_db=source_db, config_files={"config.json": config},
        repository_root=repository, bundle_root=bundle,
    )
    (repository / "second.txt").write_text("second\n", encoding="utf-8")
    _git(repository, "add", "second.txt")
    _git(repository, "commit", "-qm", "drift")
    target = tmp_path / "target"

    with pytest.raises(RuntimeError, match="restore_git_head_mismatch"):
        restore_memory_bundle_into_empty_directory(
            bundle_root=bundle, restore_root=target, repository_root=repository,
        )
    assert not target.exists()


def test_restore_fault_removes_partial_destination_and_preserves_bundle(tmp_path: Path) -> None:
    source_db = tmp_path / "source.db"
    store = EventStore(source_db)
    try:
        _write_claim(store, "fault restore body")
    finally:
        store.close()
    repository = tmp_path / "repo"
    _make_git_repo(repository)
    config = tmp_path / "config.json"
    config.write_text("{}", encoding="utf-8")
    bundle = tmp_path / "bundle"
    create_memory_restore_bundle(
        source_db=source_db, config_files={"config.json": config},
        repository_root=repository, bundle_root=bundle,
    )
    manifest_before = (bundle / "manifest.json").read_bytes()
    target = tmp_path / "target"

    def fail(checkpoint: str) -> None:
        if checkpoint == "after_database_copy":
            raise RuntimeError("forced restore fault")

    with pytest.raises(RuntimeError, match="forced restore fault"):
        restore_memory_bundle_into_empty_directory(
            bundle_root=bundle, restore_root=target,
            repository_root=repository, fault_hook=fail,
        )
    assert not target.exists()
    assert (bundle / "manifest.json").read_bytes() == manifest_before


def test_restore_fault_preserves_caller_owned_empty_destination(tmp_path: Path) -> None:
    source_db = tmp_path / "source.db"
    EventStore(source_db).close()
    repository = tmp_path / "repo"
    _make_git_repo(repository)
    config = tmp_path / "config.json"
    config.write_text("{}", encoding="utf-8")
    bundle = tmp_path / "bundle"
    create_memory_restore_bundle(
        source_db=source_db, config_files={"config.json": config},
        repository_root=repository, bundle_root=bundle,
    )
    target = tmp_path / "caller-owned-empty"
    target.mkdir()

    with pytest.raises(RuntimeError, match="forced restore fault"):
        restore_memory_bundle_into_empty_directory(
            bundle_root=bundle, restore_root=target, repository_root=repository,
            fault_hook=lambda checkpoint: (_ for _ in ()).throw(
                RuntimeError("forced restore fault")
            ) if checkpoint == "after_database_copy" else None,
        )

    assert target.is_dir()
    assert list(target.iterdir()) == []


def test_switch_and_rollback_restore_original_tree_byte_for_byte(tmp_path: Path) -> None:
    live = tmp_path / "live"
    (live / "var").mkdir(parents=True)
    (live / "config").mkdir()
    (live / "var" / "pcltm.db").write_bytes(b"original-database-bytes")
    (live / "config" / "runtime.yaml").write_bytes(b"original: true\n")
    before = {
        path.relative_to(live).as_posix(): _sha256(path)
        for path in live.rglob("*") if path.is_file()
    }
    restored = tmp_path / "restored"
    (restored / "var").mkdir(parents=True)
    (restored / "config").mkdir()
    (restored / "var" / "pcltm.db").write_bytes(b"candidate-database-bytes")
    (restored / "config" / "runtime.yaml").write_bytes(b"candidate: true\n")
    rollback_root = tmp_path / "rollback"

    receipt = switch_memory_restore_with_rollback(
        restored_root=restored, live_root=live, rollback_root=rollback_root,
    )
    assert (live / "var" / "pcltm.db").read_bytes() == b"candidate-database-bytes"
    assert receipt["previous_tree_sha256"] != receipt["restored_tree_sha256"]

    result = rollback_memory_restore_switch(receipt)

    after = {
        path.relative_to(live).as_posix(): _sha256(path)
        for path in live.rglob("*") if path.is_file()
    }
    assert result["rolled_back"] is True
    assert after == before
    assert not rollback_root.exists()


def test_switch_failure_automatically_restores_original_bytes(tmp_path: Path) -> None:
    live = tmp_path / "live"
    live.mkdir()
    original = live / "sentinel.bin"
    original.write_bytes(b"original")
    restored = tmp_path / "restored"
    restored.mkdir()
    (restored / "sentinel.bin").write_bytes(b"candidate")
    rollback_root = tmp_path / "rollback"

    def fail(checkpoint: str) -> None:
        if checkpoint == "after_switch":
            raise RuntimeError("forced switch failure")

    with pytest.raises(RuntimeError, match="forced switch failure"):
        switch_memory_restore_with_rollback(
            restored_root=restored, live_root=live,
            rollback_root=rollback_root, fault_hook=fail,
        )

    assert original.read_bytes() == b"original"
    assert not rollback_root.exists()


def test_restore_rejects_symlink_destination_without_touching_target(tmp_path: Path) -> None:
    real = tmp_path / "real"
    real.mkdir()
    sentinel = real / "sentinel.bin"
    sentinel.write_bytes(b"preserve")
    link = tmp_path / "linked-restore"
    try:
        link.symlink_to(real, target_is_directory=True)
    except OSError:
        pytest.skip("directory symlinks unavailable")

    with pytest.raises(RuntimeError, match="restore_destination_symlink_forbidden"):
        restore_memory_bundle_into_empty_directory(
            bundle_root=tmp_path / "unused-bundle",
            restore_root=link,
            repository_root=tmp_path,
        )

    assert sentinel.read_bytes() == b"preserve"
    assert link.is_symlink()


def test_switch_requires_same_parent_sibling_directories(tmp_path: Path) -> None:
    live = tmp_path / "one" / "live"
    restored = tmp_path / "two" / "restored"
    rollback = tmp_path / "one" / "rollback"
    live.mkdir(parents=True)
    restored.mkdir(parents=True)

    with pytest.raises(RuntimeError, match="restore_switch_precondition_failed"):
        switch_memory_restore_with_rollback(
            restored_root=restored, live_root=live, rollback_root=rollback,
        )

    assert live.is_dir() and restored.is_dir() and not rollback.exists()


def test_switch_rejects_symlink_root_before_hash_or_rename(tmp_path: Path) -> None:
    real = tmp_path / "real"
    real.mkdir()
    (real / "sentinel.bin").write_bytes(b"preserve")
    linked = tmp_path / "linked"
    restored = tmp_path / "restored"
    restored.mkdir()
    try:
        linked.symlink_to(real, target_is_directory=True)
    except OSError:
        pytest.skip("directory symlinks unavailable")

    with pytest.raises(RuntimeError, match="restore_switch_precondition_failed"):
        switch_memory_restore_with_rollback(
            restored_root=restored, live_root=linked,
            rollback_root=tmp_path / "rollback",
        )

    assert (real / "sentinel.bin").read_bytes() == b"preserve"
    assert linked.is_symlink()


def test_restore_rejects_symlinked_bundle_artifact(tmp_path: Path) -> None:
    source = tmp_path / "source.db"
    EventStore(source).close()
    config = tmp_path / "runtime.json"
    config.write_bytes(b'{"authority":"pcltm"}\n')
    repository = tmp_path / "repo"
    _make_git_repo(repository)
    bundle = tmp_path / "bundle"
    create_memory_restore_bundle(
        source_db=source, config_files={"config/runtime.json": config},
        repository_root=repository, bundle_root=bundle,
    )
    artifact = bundle / "configs" / "config" / "runtime.json"
    external = tmp_path / "external.json"
    external.write_bytes(artifact.read_bytes())
    artifact.unlink()
    try:
        artifact.symlink_to(external)
    except OSError:
        pytest.skip("file symlinks unavailable")

    with pytest.raises(RuntimeError, match="restore_bundle_symlink_forbidden"):
        restore_memory_bundle_into_empty_directory(
            bundle_root=bundle, restore_root=tmp_path / "restore",
            repository_root=repository,
        )

    assert external.read_bytes() == config.read_bytes()
    assert not (tmp_path / "restore").exists()


def test_explicit_rollback_keeps_current_tree_if_old_tree_rename_fails(tmp_path: Path, monkeypatch) -> None:
    live = tmp_path / "live"
    rollback = tmp_path / "rollback"
    live.mkdir()
    rollback.mkdir()
    (live / "state.bin").write_bytes(b"candidate")
    (rollback / "state.bin").write_bytes(b"original")
    from pcltm import memory_restore_rehearsal as module
    receipt = {
        "schema_version": 1,
        "live_root": str(live),
        "rollback_root": str(rollback),
        "previous_tree_sha256": module._tree_sha256(rollback),
        "restored_tree_sha256": module._tree_sha256(live),
    }
    real_replace = Path.replace

    def fail_old_tree(source: Path, target: Path):
        if source.resolve() == rollback.resolve():
            raise OSError("forced old-tree rename failure")
        return real_replace(source, target)

    monkeypatch.setattr(Path, "replace", fail_old_tree)

    with pytest.raises(OSError, match="forced old-tree rename failure"):
        rollback_memory_restore_switch(receipt)

    assert (live / "state.bin").read_bytes() == b"candidate"
    assert (rollback / "state.bin").read_bytes() == b"original"


def test_restore_rejects_symlinked_bundle_parent_directory(tmp_path: Path) -> None:
    source = tmp_path / "source.db"
    EventStore(source).close()
    config = tmp_path / "runtime.json"
    config.write_bytes(b'{"authority":"pcltm"}\n')
    repository = tmp_path / "repo"
    _make_git_repo(repository)
    bundle = tmp_path / "bundle"
    create_memory_restore_bundle(
        source_db=source, config_files={"config/runtime.json": config},
        repository_root=repository, bundle_root=bundle,
    )
    real_configs = tmp_path / "real-configs"
    (bundle / "configs").replace(real_configs)
    try:
        (bundle / "configs").symlink_to(real_configs, target_is_directory=True)
    except OSError:
        pytest.skip("directory symlinks unavailable")

    with pytest.raises(RuntimeError, match="restore_bundle_symlink_forbidden"):
        restore_memory_bundle_into_empty_directory(
            bundle_root=bundle, restore_root=tmp_path / "restore",
            repository_root=repository,
        )

    assert not (tmp_path / "restore").exists()


def test_tree_commitment_includes_empty_directory_structure(tmp_path: Path) -> None:
    from pcltm import memory_restore_rehearsal as module
    first = tmp_path / "first"
    second = tmp_path / "second"
    (first / "empty-a").mkdir(parents=True)
    (second / "empty-b").mkdir(parents=True)

    assert module._tree_sha256(first) != module._tree_sha256(second)


def test_restore_rejects_windows_reparse_destination_before_bundle_read(tmp_path: Path, monkeypatch) -> None:
    from pcltm import memory_restore_rehearsal as module
    target = (tmp_path / "junction-like").absolute()
    real_check = module._path_is_link_or_reparse

    monkeypatch.setattr(
        module,
        "_path_is_link_or_reparse",
        lambda path: True if path.absolute() == target else real_check(path),
    )

    with pytest.raises(RuntimeError, match="restore_destination_symlink_forbidden"):
        restore_memory_bundle_into_empty_directory(
            bundle_root=tmp_path / "unused", restore_root=target,
            repository_root=tmp_path,
        )


def test_rollback_rejects_reparse_displaced_path_before_rename(tmp_path: Path, monkeypatch) -> None:
    from pcltm import memory_restore_rehearsal as module
    live = tmp_path / "live"
    rollback = tmp_path / "rollback"
    live.mkdir()
    rollback.mkdir()
    (live / "state.bin").write_bytes(b"candidate")
    (rollback / "state.bin").write_bytes(b"original")
    displaced = live.with_name(f".{live.name}.rollback-displaced").absolute()
    receipt = {
        "schema_version": 1,
        "live_root": str(live),
        "rollback_root": str(rollback),
        "previous_tree_sha256": module._tree_sha256(rollback),
        "restored_tree_sha256": module._tree_sha256(live),
    }
    real_check = module._path_is_link_or_reparse
    monkeypatch.setattr(
        module, "_path_is_link_or_reparse",
        lambda path: True if path.absolute() == displaced else real_check(path),
    )

    with pytest.raises(RuntimeError, match="restore_switch_receipt_invalid"):
        rollback_memory_restore_switch(receipt)
    assert (live / "state.bin").read_bytes() == b"candidate"
    assert (rollback / "state.bin").read_bytes() == b"original"


def test_switch_rejects_reparse_common_parent_before_hash_or_rename(tmp_path: Path, monkeypatch) -> None:
    from pcltm import memory_restore_rehearsal as module
    live = tmp_path / "live"
    restored = tmp_path / "restored"
    rollback = tmp_path / "rollback"
    live.mkdir()
    restored.mkdir()
    (live / "state.bin").write_bytes(b"original")
    (restored / "state.bin").write_bytes(b"candidate")
    common_parent = tmp_path.absolute()
    real_check = module._path_is_link_or_reparse

    monkeypatch.setattr(
        module, "_path_is_link_or_reparse",
        lambda path: True if path.absolute() == common_parent else real_check(path),
    )

    with pytest.raises(RuntimeError, match="restore_switch_precondition_failed"):
        switch_memory_restore_with_rollback(
            restored_root=restored, live_root=live, rollback_root=rollback,
        )
    assert (live / "state.bin").read_bytes() == b"original"
    assert (restored / "state.bin").read_bytes() == b"candidate"
    assert not rollback.exists()
