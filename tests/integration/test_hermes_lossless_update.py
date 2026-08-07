from __future__ import annotations

import contextlib
import json
import os
import sqlite3
import subprocess
from pathlib import Path

from soul_link.hermes_update import LosslessUpdateController, build_controller, main


def _sqlite(path: Path, value: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with contextlib.closing(sqlite3.connect(path)) as conn:
        conn.execute("create table if not exists state(value text)")
        conn.execute("delete from state")
        conn.execute("insert into state values (?)", (value,))
        conn.commit()


def _value(path: Path) -> str:
    with contextlib.closing(sqlite3.connect(path)) as conn:
        return str(conn.execute("select value from state").fetchone()[0])


def test_prepare_and_restore_recover_host_profile_and_sqlite_byte_state(tmp_path: Path) -> None:
    host = tmp_path / "hermes-agent"
    home = tmp_path / "hermes-home"
    soullink = tmp_path / "Soul-Llink"
    host.mkdir()
    home.mkdir()
    soullink.mkdir()
    (host / "agent.py").write_text("old host\n", encoding="utf-8")
    (home / "config.yaml").write_text("model: old\n", encoding="utf-8")
    (home / "SOUL.md").write_text("old soul\n", encoding="utf-8")
    database = soullink / "var/pcltm.sqlite3"
    _sqlite(database, "old memory")

    controller = LosslessUpdateController(
        soullink_root=soullink,
        host_root=host,
        hermes_home=home,
        sqlite_paths=(database,),
    )
    receipt_path = tmp_path / "recovery/receipt.json"
    receipt = controller.prepare(receipt_path)

    assert receipt.version == "2"
    assert receipt_path.is_file()
    assert json.loads(receipt_path.read_text(encoding="utf-8"))["verified"] is True

    (host / "agent.py").write_text("new host\n", encoding="utf-8")
    (host / "new-upstream.py").write_text("new\n", encoding="utf-8")
    (home / "config.yaml").write_text("model: new\n", encoding="utf-8")
    (home / "SOUL.md").unlink()
    _sqlite(database, "new memory")

    assert controller.restore(receipt_path) is True
    assert (host / "agent.py").read_text(encoding="utf-8") == "old host\n"
    assert not (host / "new-upstream.py").exists()
    assert (home / "config.yaml").read_text(encoding="utf-8") == "model: old\n"
    assert (home / "SOUL.md").read_text(encoding="utf-8") == "old soul\n"
    assert _value(database) == "old memory"


def test_execute_rolls_back_automatically_when_post_update_verification_fails(tmp_path: Path) -> None:
    host = tmp_path / "hermes-agent"
    home = tmp_path / "hermes-home"
    soullink = tmp_path / "Soul-Llink"
    host.mkdir()
    home.mkdir()
    soullink.mkdir()
    subprocess.run(["git", "init", "-q", str(host)], check=True)
    (host / "version.txt").write_text("old\n", encoding="utf-8")
    subprocess.run(["git", "-C", str(host), "add", "version.txt"], check=True)
    subprocess.run(["git", "-C", str(host), "-c", "user.name=test", "-c", "user.email=test@example.invalid", "commit", "-qm", "base"], check=True)
    (home / "config.yaml").write_text("memory: old\n", encoding="utf-8")
    database = soullink / "var/pcltm.sqlite3"
    _sqlite(database, "old memory")
    receipt_path = tmp_path / "recovery/receipt.json"
    controller = LosslessUpdateController(
        soullink_root=soullink,
        host_root=host,
        hermes_home=home,
        sqlite_paths=(database,),
    )

    def update() -> None:
        (host / "version.txt").write_text("new\n", encoding="utf-8")
        (home / "config.yaml").write_text("memory: broken\n", encoding="utf-8")
        _sqlite(database, "corrupted logical state")

    result = controller.execute(
        receipt_path,
        update=update,
        deploy=lambda: None,
        verify=lambda: False,
    )

    assert result["updated"] is False
    assert result["rolled_back"] is True
    assert result["activation_required"] is False
    assert (host / "version.txt").read_text(encoding="utf-8") == "old\n"
    assert (home / "config.yaml").read_text(encoding="utf-8") == "memory: old\n"
    assert _value(database) == "old memory"


def test_execute_keeps_verified_update_and_marks_restart_pending(tmp_path: Path) -> None:
    host = tmp_path / "hermes-agent"
    home = tmp_path / "hermes-home"
    soullink = tmp_path / "Soul-Llink"
    host.mkdir()
    home.mkdir()
    soullink.mkdir()
    subprocess.run(["git", "init", "-q", str(host)], check=True)
    (host / "version.txt").write_text("old\n", encoding="utf-8")
    subprocess.run(["git", "-C", str(host), "add", "version.txt"], check=True)
    subprocess.run(["git", "-C", str(host), "-c", "user.name=test", "-c", "user.email=test@example.invalid", "commit", "-qm", "base"], check=True)
    (home / "config.yaml").write_text("memory: old\n", encoding="utf-8")
    controller = LosslessUpdateController(
        soullink_root=soullink,
        host_root=host,
        hermes_home=home,
    )
    calls: list[str] = []

    result = controller.execute(
        tmp_path / "recovery/receipt.json",
        update=lambda: ((host / "version.txt").write_text("new\n", encoding="utf-8"), calls.append("update")),
        deploy=lambda: calls.append("deploy"),
        verify=lambda: calls.append("verify") or True,
    )

    assert calls == ["update", "deploy", "verify"]
    assert result == {
        "updated": True,
        "verified": True,
        "rolled_back": False,
        "activation_required": True,
        "receipt": str((tmp_path / "recovery/receipt.json").resolve()),
    }
    assert (host / "version.txt").read_text(encoding="utf-8") == "new\n"


def test_preflight_allows_declared_legacy_residue_but_rejects_unknown_host_delta(tmp_path: Path) -> None:
    host = tmp_path / "hermes-agent"
    home = tmp_path / "hermes-home"
    soullink = tmp_path / "Soul-Llink"
    host.mkdir()
    home.mkdir()
    soullink.mkdir()
    subprocess.run(["git", "init", "-q", str(host)], check=True)
    subprocess.run(["git", "-C", str(host), "config", "user.name", "test"], check=True)
    subprocess.run(["git", "-C", str(host), "config", "user.email", "test@example.invalid"], check=True)
    (host / "base.py").write_text("base\n", encoding="utf-8")
    subprocess.run(["git", "-C", str(host), "add", "base.py"], check=True)
    subprocess.run(["git", "-C", str(host), "commit", "-qm", "base"], check=True)
    legacy = host / "plugins/context_engine/pcltm-context/__init__.py"
    legacy.parent.mkdir(parents=True)
    legacy.write_text("legacy\n", encoding="utf-8")
    controller = LosslessUpdateController(
        soullink_root=soullink,
        host_root=host,
        hermes_home=home,
        allowed_host_deltas=("plugins/context_engine/pcltm-context/",),
    )

    allowed = controller.preflight()
    assert allowed["ready"] is True
    assert allowed["legacy_residue"] == ["plugins/context_engine/pcltm-context/"]

    (host / "unknown.txt").write_text("unowned\n", encoding="utf-8")
    blocked = controller.preflight()
    assert blocked["ready"] is False
    assert blocked["unknown_host_deltas"] == ["unknown.txt"]


def test_prepare_and_restore_preserve_soullink_source_and_memfs(tmp_path: Path) -> None:
    host = tmp_path / "hermes-agent"
    home = tmp_path / "hermes-home"
    soullink = tmp_path / "Soul-Llink"
    host.mkdir()
    home.mkdir()
    soullink.mkdir()
    (host / "host.py").write_text("host\n", encoding="utf-8")
    (soullink / ".git").mkdir()
    (soullink / ".git/HEAD").write_text("ref: refs/heads/main\n", encoding="utf-8")
    (soullink / "soul_link").mkdir()
    (soullink / "soul_link/runtime.py").write_text("production\n", encoding="utf-8")
    memfs = soullink / "var/memfs/pinned/user.md"
    memfs.parent.mkdir(parents=True)
    memfs.write_text("durable memory\n", encoding="utf-8")
    controller = LosslessUpdateController(
        soullink_root=soullink,
        host_root=host,
        hermes_home=home,
    )
    receipt_path = tmp_path / "recovery/receipt.json"

    controller.prepare(receipt_path)
    (soullink / "soul_link/runtime.py").write_text("broken\n", encoding="utf-8")
    (soullink / ".git/HEAD").write_text("broken\n", encoding="utf-8")
    memfs.write_text("lost\n", encoding="utf-8")

    assert controller.restore(receipt_path) is True
    assert (soullink / "soul_link/runtime.py").read_text(encoding="utf-8") == "production\n"
    assert (soullink / ".git/HEAD").read_text(encoding="utf-8") == "ref: refs/heads/main\n"
    assert memfs.read_text(encoding="utf-8") == "durable memory\n"


def test_restore_is_safe_when_host_and_soullink_are_nested_in_hermes_home(tmp_path: Path) -> None:
    home = tmp_path / "hermes-home"
    host = home / "hermes-agent"
    soullink = home / "plugins/Soul-Llink"
    host.mkdir(parents=True)
    soullink.mkdir(parents=True)
    (host / "host.py").write_text("old host\n", encoding="utf-8")
    (soullink / "runtime.py").write_text("old runtime\n", encoding="utf-8")
    other_plugin = home / "plugins/other/plugin.py"
    other_plugin.parent.mkdir(parents=True)
    other_plugin.write_text("old plugin\n", encoding="utf-8")
    database = soullink / "var/pcltm-prod.db"
    _sqlite(database, "old memory")
    controller = LosslessUpdateController(
        soullink_root=soullink,
        host_root=host,
        hermes_home=home,
        sqlite_paths=(database,),
    )
    receipt_path = tmp_path / "recovery/receipt.json"
    controller.prepare(receipt_path)

    (host / "host.py").write_text("new host\n", encoding="utf-8")
    (soullink / "runtime.py").write_text("new runtime\n", encoding="utf-8")
    other_plugin.write_text("new plugin\n", encoding="utf-8")
    _sqlite(database, "new memory")

    assert controller.restore(receipt_path) is True
    assert (host / "host.py").read_text(encoding="utf-8") == "old host\n"
    assert (soullink / "runtime.py").read_text(encoding="utf-8") == "old runtime\n"
    assert other_plugin.read_text(encoding="utf-8") == "old plugin\n"
    assert _value(database) == "old memory"


def test_prepare_rejects_recovery_location_inside_any_managed_root(tmp_path: Path) -> None:
    home = tmp_path / "home"
    host = home / "hermes-agent"
    soullink = home / "plugins/Soul-Llink"
    host.mkdir(parents=True)
    soullink.mkdir(parents=True)
    controller = LosslessUpdateController(soullink_root=soullink, host_root=host, hermes_home=home)

    import pytest

    with pytest.raises(RuntimeError, match="outside Hermes home"):
        controller.prepare(home / "recovery/receipt.json")


def test_execute_refuses_unknown_host_delta_before_creating_recovery_point(tmp_path: Path) -> None:
    host = tmp_path / "host"
    home = tmp_path / "home"
    soullink = tmp_path / "soullink"
    host.mkdir()
    home.mkdir()
    soullink.mkdir()
    subprocess.run(["git", "init", "-q", str(host)], check=True)
    (host / "unknown.txt").write_text("unknown\n", encoding="utf-8")
    receipt = tmp_path / "recovery/receipt.json"
    controller = LosslessUpdateController(soullink_root=soullink, host_root=host, hermes_home=home)

    import pytest

    with pytest.raises(RuntimeError, match="preflight blocked"):
        controller.execute(receipt, update=lambda: None, deploy=lambda: None, verify=lambda: True)
    assert not receipt.exists()


def test_build_controller_discovers_manifest_paths_database_and_orphan_receipt(tmp_path: Path) -> None:
    home = tmp_path / "home"
    host = home / "hermes-agent"
    soullink = home / "plugins/Soul-Llink"
    (soullink / "adapters/hermes").mkdir(parents=True)
    host.mkdir(parents=True)
    subprocess.run(["git", "init", "-q", str(host)], check=True)
    (soullink / "adapters/hermes/compatibility-soullink-runtime.yaml").write_text(
        "required_paths:\n  - agent/a.py\ncreated_paths:\n  - agent/b.py\n",
        encoding="utf-8",
    )
    database = soullink / "var/pcltm-prod.db"
    _sqlite(database, "memory")
    state_db = home / "state.db"
    _sqlite(state_db, "session")
    (home / "soullink-deployment-receipt.json").write_text(
        json.dumps({"adapter_version": "2", "backup_path": str(home / "missing-backup")}),
        encoding="utf-8",
    )

    controller = build_controller(soullink, host, home)
    report = controller.preflight()

    assert set(controller.allowed_host_deltas) == {
        "agent/a.py",
        "agent/b.py",
        "plugins/context_engine/pcltm-context/",
    }
    assert controller.sqlite_paths == (database.resolve(), state_db.resolve())
    assert report["legacy_deployment_receipt"] == "orphaned"


def test_sqlite_wal_sidecars_are_not_archived_and_restore_uses_consistent_backup(tmp_path: Path) -> None:
    host = tmp_path / "host"
    home = tmp_path / "home"
    soullink = tmp_path / "soullink"
    host.mkdir()
    home.mkdir()
    soullink.mkdir()
    database = soullink / "var/pcltm-prod.db"
    _sqlite(database, "old")
    (database.parent / "pcltm-prod.db-wal").write_bytes(b"stale-wal")
    (database.parent / "pcltm-prod.db-shm").write_bytes(b"stale-shm")
    controller = LosslessUpdateController(
        soullink_root=soullink, host_root=host, hermes_home=home, sqlite_paths=(database,)
    )
    receipt_path = tmp_path / "recovery/receipt.json"

    receipt = controller.prepare(receipt_path)

    import zipfile

    with zipfile.ZipFile(receipt.soullink_archive) as archive:
        names = set(archive.namelist())
    assert "root/var/pcltm-prod.db" not in names
    assert "root/var/pcltm-prod.db-wal" not in names
    assert "root/var/pcltm-prod.db-shm" not in names


def test_soullink_rebuildable_and_historical_trees_are_excluded_but_preserved_on_restore(tmp_path: Path) -> None:
    host = tmp_path / "host"
    home = tmp_path / "home"
    soullink = tmp_path / "soullink"
    host.mkdir()
    home.mkdir()
    soullink.mkdir()
    (host / "agent.py").write_text("old\n", encoding="utf-8")
    (host / "venv/cache.bin").parent.mkdir()
    (host / "venv/cache.bin").write_bytes(b"rebuildable")
    (soullink / "runtime.py").write_text("old\n", encoding="utf-8")
    (soullink / ".venv/cache.bin").parent.mkdir()
    (soullink / ".venv/cache.bin").write_bytes(b"rebuildable")
    historical = soullink / "var/backups/history.db"
    historical.parent.mkdir(parents=True)
    historical.write_bytes(b"historical")
    controller = LosslessUpdateController(soullink_root=soullink, host_root=host, hermes_home=home)
    receipt_path = tmp_path / "recovery/receipt.json"

    receipt = controller.prepare(receipt_path)

    import zipfile

    with zipfile.ZipFile(receipt.host_archive) as archive:
        assert "root/venv/cache.bin" in set(archive.namelist())
    with zipfile.ZipFile(receipt.soullink_archive) as archive:
        names = set(archive.namelist())
        assert "root/.venv/cache.bin" not in names
        assert "root/var/backups/history.db" not in names
    (host / "agent.py").write_text("new\n", encoding="utf-8")
    (soullink / "runtime.py").write_text("new\n", encoding="utf-8")
    assert controller.restore(receipt_path) is True
    assert (host / "venv/cache.bin").read_bytes() == b"rebuildable"
    assert (soullink / ".venv/cache.bin").read_bytes() == b"rebuildable"
    assert historical.read_bytes() == b"historical"


def test_cli_preflight_returns_machine_readable_json(tmp_path: Path, capsys) -> None:
    home = tmp_path / "home"
    host = home / "hermes-agent"
    soullink = home / "plugins/Soul-Llink"
    (soullink / "adapters/hermes").mkdir(parents=True)
    host.mkdir(parents=True)
    subprocess.run(["git", "init", "-q", str(host)], check=True)
    (soullink / "adapters/hermes/compatibility-soullink-runtime.yaml").write_text(
        "required_paths: []\ncreated_paths: []\n", encoding="utf-8"
    )

    code = main([
        "preflight", "--soullink-root", str(soullink), "--host-root", str(host),
        "--hermes-home", str(home),
    ])
    payload = json.loads(capsys.readouterr().out)

    assert code == 0
    assert payload["ready"] is True


def test_restore_rejects_archive_path_outside_bound_recovery_root(tmp_path: Path) -> None:
    host = tmp_path / "host"
    home = tmp_path / "home"
    soullink = tmp_path / "soullink"
    host.mkdir()
    home.mkdir()
    soullink.mkdir()
    (host / "host.py").write_text("safe\n", encoding="utf-8")
    controller = LosslessUpdateController(soullink_root=soullink, host_root=host, hermes_home=home)
    receipt_path = tmp_path / "recovery/receipt.json"
    receipt = controller.prepare(receipt_path)
    external = tmp_path / "external.zip"

    import hashlib
    import zipfile

    with zipfile.ZipFile(external, "w") as archive:
        archive.writestr("root/pwned.txt", "pwned\n")
    data = json.loads(receipt_path.read_text(encoding="utf-8"))
    data["host_archive"] = str(external.resolve())
    data["hashes"]["host.zip"] = hashlib.sha256(external.read_bytes()).hexdigest()
    receipt_path.write_text(json.dumps(data), encoding="utf-8")
    (receipt.recovery_root / "manifest.json").write_text(json.dumps(data), encoding="utf-8")

    import pytest

    with pytest.raises(RuntimeError, match="authentication"):
        controller.restore(receipt_path)
    assert not (host / "pwned.txt").exists()


def test_authentication_key_is_written_as_exact_binary_bytes(tmp_path: Path, monkeypatch) -> None:
    host = tmp_path / "host"
    home = tmp_path / "home"
    soullink = tmp_path / "soullink"
    host.mkdir()
    home.mkdir()
    soullink.mkdir()
    controller = LosslessUpdateController(soullink_root=soullink, host_root=host, hermes_home=home)
    key = b"A" * 15 + b"\n" + b"B" * 16
    monkeypatch.setattr("soul_link.hermes_update.secrets.token_bytes", lambda size: key)

    controller.prepare(tmp_path / "recovery/receipt.json")

    assert controller._auth_key_path.read_bytes() == key
    assert controller._auth_key_path.stat().st_size == 32


def test_restore_rejects_self_consistent_archive_and_metadata_tampering(tmp_path: Path) -> None:
    host = tmp_path / "host"
    home = tmp_path / "home"
    soullink = tmp_path / "soullink"
    host.mkdir()
    home.mkdir()
    soullink.mkdir()
    (host / "host.py").write_text("safe\n", encoding="utf-8")
    controller = LosslessUpdateController(soullink_root=soullink, host_root=host, hermes_home=home)
    receipt_path = tmp_path / "recovery/receipt.json"
    receipt = controller.prepare(receipt_path)

    import hashlib
    import pytest
    import zipfile

    with zipfile.ZipFile(receipt.host_archive, "w") as archive:
        archive.writestr("root/pwned.txt", "attacker-controlled\n")
    data = json.loads(receipt_path.read_text(encoding="utf-8"))
    data["hashes"]["host.zip"] = hashlib.sha256(receipt.host_archive.read_bytes()).hexdigest()
    receipt_path.write_text(json.dumps(data), encoding="utf-8")
    (receipt.recovery_root / "manifest.json").write_text(json.dumps(data), encoding="utf-8")

    with pytest.raises(RuntimeError, match="authentication"):
        controller.restore(receipt_path)
    assert not (host / "pwned.txt").exists()
    assert (host / "host.py").read_text(encoding="utf-8") == "safe\n"


def test_restore_compensates_all_surfaces_when_mid_restore_step_fails(tmp_path: Path, monkeypatch) -> None:
    host = tmp_path / "host"
    home = tmp_path / "home"
    soullink = tmp_path / "soullink"
    host.mkdir()
    home.mkdir()
    soullink.mkdir()
    (host / "host.py").write_text("old host\n", encoding="utf-8")
    (home / "config.yaml").write_text("old profile\n", encoding="utf-8")
    (soullink / "runtime.py").write_text("old soul\n", encoding="utf-8")
    database = soullink / "var/pcltm-prod.db"
    _sqlite(database, "old db")
    controller = LosslessUpdateController(
        soullink_root=soullink, host_root=host, hermes_home=home, sqlite_paths=(database,)
    )
    receipt = tmp_path / "recovery/receipt.json"
    controller.prepare(receipt)
    (host / "host.py").write_text("current host\n", encoding="utf-8")
    (home / "config.yaml").write_text("current profile\n", encoding="utf-8")
    (soullink / "runtime.py").write_text("current soul\n", encoding="utf-8")
    _sqlite(database, "current db")
    original = controller._restore_soullink
    calls = 0

    def fail_once(staged: Path) -> None:
        nonlocal calls
        calls += 1
        if calls == 1:
            raise RuntimeError("injected restore failure")
        original(staged)

    monkeypatch.setattr(controller, "_restore_soullink", fail_once)

    import pytest

    with pytest.raises(RuntimeError, match="injected restore failure"):
        controller.restore(receipt)
    assert (host / "host.py").read_text(encoding="utf-8") == "current host\n"
    assert (home / "config.yaml").read_text(encoding="utf-8") == "current profile\n"
    assert (soullink / "runtime.py").read_text(encoding="utf-8") == "current soul\n"
    assert _value(database) == "current db"


def test_update_lock_is_exclusive_across_controllers(tmp_path: Path) -> None:
    host = tmp_path / "host"
    home = tmp_path / "home"
    soullink = tmp_path / "soullink"
    host.mkdir()
    home.mkdir()
    soullink.mkdir()
    first = LosslessUpdateController(soullink_root=soullink, host_root=host, hermes_home=home)
    second = LosslessUpdateController(soullink_root=soullink, host_root=host, hermes_home=home)

    import pytest

    with first.update_lock():
        with pytest.raises(RuntimeError, match="already active"):
            with second.update_lock():
                pass
    with second.update_lock():
        pass


def test_restore_removes_sqlite_sidecars_before_recovery(tmp_path: Path) -> None:
    host = tmp_path / "host"
    home = tmp_path / "home"
    soullink = tmp_path / "soullink"
    host.mkdir()
    home.mkdir()
    soullink.mkdir()
    database = soullink / "var/pcltm-prod.db"
    _sqlite(database, "old")
    controller = LosslessUpdateController(
        soullink_root=soullink, host_root=host, hermes_home=home, sqlite_paths=(database,)
    )
    receipt = tmp_path / "recovery/receipt.json"
    controller.prepare(receipt)
    _sqlite(database, "new")
    for suffix in ("-wal", "-shm", "-journal"):
        Path(str(database) + suffix).write_bytes(b"stale")

    assert controller.restore(receipt) is True
    assert _value(database) == "old"
    assert all(not Path(str(database) + suffix).exists() for suffix in ("-wal", "-shm", "-journal"))


def test_prepare_refuses_to_overwrite_existing_receipt(tmp_path: Path) -> None:
    host = tmp_path / "host"
    home = tmp_path / "home"
    soullink = tmp_path / "soullink"
    host.mkdir()
    home.mkdir()
    soullink.mkdir()
    receipt = tmp_path / "recovery/receipt.json"
    receipt.parent.mkdir()
    receipt.write_text("operator evidence\n", encoding="utf-8")
    controller = LosslessUpdateController(soullink_root=soullink, host_root=host, hermes_home=home)

    import pytest

    with pytest.raises(RuntimeError, match="already exists"):
        controller.prepare(receipt)
    assert receipt.read_text(encoding="utf-8") == "operator evidence\n"


def test_managed_host_update_normalizes_divergent_checkout_after_backup(tmp_path: Path, monkeypatch) -> None:
    host = tmp_path / "host"
    host.mkdir()
    executable = host / "venv/Scripts/hermes.exe"
    executable.parent.mkdir(parents=True)
    executable.write_text("launcher\n", encoding="utf-8")
    python = host / "venv/Scripts/python.exe"
    python.write_text("python\n", encoding="utf-8")
    calls: list[tuple[list[str], Path | None]] = []

    class Result:
        returncode = 0
        stdout = ""
        stderr = ""

    def fake_run(command, **kwargs):
        calls.append(([str(value) for value in command], kwargs.get("cwd")))
        return Result()

    monkeypatch.setattr("soul_link.hermes_update.subprocess.run", fake_run)
    monkeypatch.setattr("soul_link.hermes_update.shutil.which", lambda name: "uv" if name == "uv" else None)
    from soul_link.hermes_update import _update_managed_host

    _update_managed_host(
        host,
        executable,
        {"HERMES_HOME": str(tmp_path / "home")},
        ("agent/managed.py", "plugins/context_engine/pcltm-context/"),
    )

    assert calls[0][0] == ["git", "fetch", "--depth", "2", "origin", "main"]
    assert calls[1][0] == ["git", "rev-parse", "--verify", "origin/main^"]
    assert calls[2][0] == ["git", "reset", "--hard", "origin/main^"]
    assert calls[3][0] == [
        "git", "clean", "-fd", "--", "agent/managed.py", "plugins/context_engine/pcltm-context/"
    ]
    assert calls[4][0] == [str(executable), "update", "--yes", "--no-backup"]
    assert all(cwd == host for _, cwd in calls)


def test_restore_recovers_when_current_sqlite_is_missing_or_corrupt(tmp_path: Path) -> None:
    for condition in ("missing", "corrupt"):
        root = tmp_path / condition
        host = root / "host"
        home = root / "home"
        soullink = root / "soullink"
        host.mkdir(parents=True)
        home.mkdir()
        soullink.mkdir()
        database = soullink / "var/pcltm-prod.db"
        _sqlite(database, "frozen")
        controller = LosslessUpdateController(
            soullink_root=soullink, host_root=host, hermes_home=home, sqlite_paths=(database,)
        )
        receipt = root / "recovery/receipt.json"
        controller.prepare(receipt)
        if condition == "missing":
            database.unlink()
        else:
            database.write_bytes(b"not a database")

        assert controller.restore(receipt) is True
        assert _value(database) == "frozen"


def test_restore_rejects_profile_symlink_without_touching_external_target(tmp_path: Path) -> None:
    host = tmp_path / "host"
    home = tmp_path / "home"
    soullink = tmp_path / "soullink"
    host.mkdir()
    home.mkdir()
    soullink.mkdir()
    (home / "config.yaml").write_text("frozen\n", encoding="utf-8")
    controller = LosslessUpdateController(soullink_root=soullink, host_root=host, hermes_home=home)
    receipt = tmp_path / "recovery/receipt.json"
    controller.prepare(receipt)
    external = tmp_path / "external.yaml"
    external.write_text("external\n", encoding="utf-8")
    (home / "config.yaml").unlink()
    (home / "config.yaml").symlink_to(external)

    import pytest

    with pytest.raises(RuntimeError, match="symlink or reparse"):
        controller.restore(receipt)
    assert external.read_text(encoding="utf-8") == "external\n"


def test_preflight_matches_allowed_paths_on_component_boundary(tmp_path: Path) -> None:
    host = tmp_path / "host"
    host.mkdir()
    subprocess.run(["git", "init"], cwd=host, check=True, capture_output=True)
    (host / "agent").mkdir()
    (host / "agent/memory_provider.py.evil").write_text("evil\n", encoding="utf-8")
    controller = LosslessUpdateController(
        soullink_root=tmp_path / "soul",
        host_root=host,
        hermes_home=tmp_path / "home",
        allowed_host_deltas=("agent/memory_provider.py",),
    )

    report = controller.preflight()
    assert report["ready"] is False
    assert report["unknown_host_deltas"] == ["agent/memory_provider.py.evil"]


def test_archive_materializes_internal_directory_symlink_but_rejects_external(tmp_path: Path) -> None:
    source = tmp_path / "source"
    internal = source / "packages/internal"
    internal.mkdir(parents=True)
    (internal / "payload.txt").write_text("inside\n", encoding="utf-8")
    (source / "node_modules").mkdir()
    (source / "node_modules/internal").symlink_to(internal, target_is_directory=True)
    archive = tmp_path / "internal.zip"

    LosslessUpdateController._archive_tree(source, archive)

    import pytest
    import zipfile

    with zipfile.ZipFile(archive) as bundle:
        assert bundle.read("root/node_modules/internal/payload.txt").replace(b"\r\n", b"\n") == b"inside\n"

    external = tmp_path / "external"
    external.mkdir()
    (external / "secret.txt").write_text("outside\n", encoding="utf-8")
    (source / "node_modules/external").symlink_to(external, target_is_directory=True)
    with pytest.raises(RuntimeError, match="escapes through a reparse point"):
        LosslessUpdateController._archive_tree(source, tmp_path / "external.zip")


def test_archive_excludes_all_sqlite_sidecars(tmp_path: Path) -> None:
    source = tmp_path / "source"
    source.mkdir()
    database = source / "db.sqlite"
    database.write_bytes(b"db")
    for suffix in ("-wal", "-shm", "-journal"):
        Path(str(database) + suffix).write_bytes(b"sidecar")
    archive = tmp_path / "tree.zip"

    LosslessUpdateController._archive_tree(source, archive, excluded=(database,))

    import zipfile

    with zipfile.ZipFile(archive) as bundle:
        assert set(bundle.namelist()) == {"root/"}


def test_update_lock_recovers_stale_pid_file(tmp_path: Path, monkeypatch) -> None:
    host = tmp_path / "host"
    home = tmp_path / "home"
    soullink = tmp_path / "soullink"
    host.mkdir()
    home.mkdir()
    soullink.mkdir()
    controller = LosslessUpdateController(soullink_root=soullink, host_root=host, hermes_home=home)
    controller._lock_path.write_text("99999999", encoding="ascii")
    monkeypatch.setattr(controller, "_pid_is_alive", lambda pid: False)

    with controller.update_lock():
        assert controller._lock_path.read_text(encoding="ascii") == str(os.getpid())
    assert not controller._lock_path.exists()


def test_restore_uses_short_same_volume_host_staging_path(tmp_path: Path, monkeypatch) -> None:
    host = tmp_path / "host"
    home = tmp_path / "home"
    soullink = tmp_path / "soullink"
    host.mkdir()
    home.mkdir()
    soullink.mkdir()
    deep = host.joinpath(*(["deep"] * 8), "payload.txt")
    deep.parent.mkdir(parents=True)
    deep.write_text("frozen\n", encoding="utf-8")
    controller = LosslessUpdateController(soullink_root=soullink, host_root=host, hermes_home=home)
    receipt_path = tmp_path / "recovery/receipt.json"
    receipt = controller.prepare(receipt_path)
    seen: list[Path] = []
    original_extract = controller._extract

    def recording_extract(archive: Path, destination: Path) -> None:
        if archive == receipt.host_archive:
            seen.append(destination)
        original_extract(archive, destination)

    monkeypatch.setattr(controller, "_extract", recording_extract)
    deep.write_text("damaged\n", encoding="utf-8")

    assert controller.restore(receipt_path) is True
    assert deep.read_text(encoding="utf-8") == "frozen\n"
    assert seen and seen[0].parent == host.parent
    assert seen[0].name.startswith(".slh-")
    assert not seen[0].exists()


def test_prepare_rejects_profile_file_symlink_without_reading_external_target(tmp_path: Path) -> None:
    host = tmp_path / "host"
    home = tmp_path / "home"
    soullink = tmp_path / "soullink"
    host.mkdir()
    home.mkdir()
    soullink.mkdir()
    external = tmp_path / "external-secret.yaml"
    external.write_text("must-not-enter-archive\n", encoding="utf-8")
    (home / "config.yaml").symlink_to(external)
    controller = LosslessUpdateController(soullink_root=soullink, host_root=host, hermes_home=home)

    import pytest

    with pytest.raises(RuntimeError, match="symlink or reparse"):
        controller.prepare(tmp_path / "recovery/receipt.json")
    assert external.read_text(encoding="utf-8") == "must-not-enter-archive\n"


def test_prepare_rejects_symlinked_profile_root(tmp_path: Path) -> None:
    host = tmp_path / "host"
    real_home = tmp_path / "real-home"
    linked_home = tmp_path / "linked-home"
    soullink = tmp_path / "soullink"
    host.mkdir()
    real_home.mkdir()
    soullink.mkdir()
    linked_home.symlink_to(real_home, target_is_directory=True)
    import pytest

    with pytest.raises(RuntimeError, match="symlink or reparse"):
        LosslessUpdateController(
            soullink_root=soullink, host_root=host, hermes_home=linked_home
        )


def test_update_lock_treats_uninitialized_lock_as_active(tmp_path: Path) -> None:
    host = tmp_path / "host"
    home = tmp_path / "home"
    soullink = tmp_path / "soullink"
    host.mkdir()
    home.mkdir()
    soullink.mkdir()
    controller = LosslessUpdateController(soullink_root=soullink, host_root=host, hermes_home=home)
    controller._lock_path.write_bytes(b"")

    import pytest

    with pytest.raises(RuntimeError, match="another SoulLink update is already active"):
        with controller.update_lock():
            pass
    assert controller._lock_path.exists()
