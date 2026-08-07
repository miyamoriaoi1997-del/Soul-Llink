from __future__ import annotations

import argparse
import contextlib
import hashlib
import hmac
import json
import os
import secrets
import shutil
import sqlite3
import subprocess
import sys
import zipfile
from dataclasses import dataclass
from pathlib import Path
from uuid import uuid4

import yaml


@dataclass(frozen=True, slots=True)
class LosslessUpdateReceipt:
    version: str
    soullink_root: Path
    host_root: Path
    hermes_home: Path
    recovery_root: Path
    host_archive: Path
    profile_archive: Path
    soullink_archive: Path
    sqlite_backups: tuple[tuple[Path, Path], ...]
    verified: bool


class LosslessUpdateController:
    """Create and exercise a recovery point before a Hermes host update.

    The recovery point contains a complete host checkout, selected profile state,
    and online-consistent SQLite copies. ``prepare`` does not mutate production.
    ``restore`` fails closed until every recorded artifact passes its hash and
    integrity checks.
    """

    version = "2"
    profile_entries = (
        "config.yaml",
        ".env",
        "SOUL.md",
        "auth.json",
        "state.db",
        "skills",
        "cron",
    )
    profile_plugin_entries = ("soullink", "pcltm-context")
    soullink_preserved_entries = (".venv", "var/backups")

    def __init__(
        self,
        *,
        soullink_root: Path,
        host_root: Path,
        hermes_home: Path,
        sqlite_paths: tuple[Path, ...] = (),
        allowed_host_deltas: tuple[str, ...] = (),
    ) -> None:
        self._configured_hermes_home = Path(os.path.abspath(os.fspath(hermes_home)))
        self.soullink_root = Path(soullink_root).resolve()
        self.host_root = Path(host_root).resolve()
        self.hermes_home = self._configured_hermes_home.resolve()
        self.sqlite_paths = tuple(Path(path).resolve() for path in sqlite_paths)
        self.allowed_host_deltas = tuple(str(path).replace(chr(92), "/") for path in allowed_host_deltas)
        self._lock_path = self.hermes_home.parent / f".{self.hermes_home.name}.soullink-update.lock"
        self._auth_key_path = self.hermes_home.parent / f".{self.hermes_home.name}.soullink-recovery.key"

    @contextlib.contextmanager
    def update_lock(self):
        self._lock_path.parent.mkdir(parents=True, exist_ok=True)
        acquired = False
        temp_lock = self._lock_path.with_name(f".{self._lock_path.name}.{os.getpid()}.{uuid4().hex}.tmp")
        temp_lock.write_text(str(os.getpid()), encoding="ascii")
        try:
            for _ in range(2):
                try:
                    os.link(temp_lock, self._lock_path)
                    acquired = True
                    break
                except FileExistsError as exc:
                    try:
                        owner = int(self._lock_path.read_text(encoding="ascii").strip())
                    except (OSError, ValueError):
                        raise RuntimeError(
                            f"another SoulLink update is already active: {self._lock_path}"
                        ) from exc
                    if owner <= 0 or self._pid_is_alive(owner):
                        raise RuntimeError(f"another SoulLink update is already active: {self._lock_path}") from exc
                    try:
                        self._lock_path.unlink()
                    except OSError as unlink_error:
                        raise RuntimeError(
                            f"stale SoulLink update lock cannot be removed: {self._lock_path}"
                        ) from unlink_error
        finally:
            temp_lock.unlink(missing_ok=True)
        if not acquired:
            raise RuntimeError(f"could not acquire SoulLink update lock: {self._lock_path}")
        try:
            yield
        finally:
            temp_lock.unlink(missing_ok=True)
            try:
                if self._lock_path.read_text(encoding="ascii").strip() == str(os.getpid()):
                    self._lock_path.unlink(missing_ok=True)
            except OSError:
                pass

    @staticmethod
    def _pid_is_alive(pid: int) -> bool:
        if pid == os.getpid():
            return True
        if sys.platform == "win32":
            result = subprocess.run(
                ["tasklist", "/FI", f"PID eq {pid}", "/NH"],
                text=True,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                check=False,
            )
            return result.returncode == 0 and str(pid) in result.stdout
        try:
            os.kill(pid, 0)
        except (ProcessLookupError, ValueError):
            return False
        except PermissionError:
            return True
        return True

    def preflight(self) -> dict[str, object]:
        """Fail closed when the host contains deltas outside the declared update boundary."""
        completed = subprocess.run(
            ["git", "-C", str(self.host_root), "status", "--porcelain=v1", "--untracked-files=all", "-z"],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            check=False,
        )
        if completed.returncode != 0:
            raise RuntimeError("Hermes host is not a readable Git checkout")
        paths: list[str] = []
        for raw in completed.stdout.split(b"\0"):
            if not raw:
                continue
            text = raw.decode("utf-8", "replace")
            path = text[3:]
            if " -> " in path:
                path = path.split(" -> ", 1)[1]
            paths.append(path.replace(chr(92), "/"))
        legacy: list[str] = []
        unknown: list[str] = []
        for path in paths:
            matched = next(
                (
                    allowed
                    for allowed in self.allowed_host_deltas
                    if path == allowed.rstrip("/")
                    or (allowed.endswith("/") and path.startswith(allowed))
                ),
                None,
            )
            if matched:
                if matched not in legacy:
                    legacy.append(matched)
            else:
                unknown.append(path)
        report: dict[str, object] = {
            "ready": not unknown,
            "legacy_residue": sorted(legacy),
            "unknown_host_deltas": sorted(unknown),
        }
        legacy_receipt = self.hermes_home / "soullink-deployment-receipt.json"
        if legacy_receipt.is_file():
            try:
                receipt_data = json.loads(legacy_receipt.read_text(encoding="utf-8"))
                backup = Path(str(receipt_data.get("backup_path", ""))).resolve()
                report["legacy_deployment_receipt"] = "complete" if backup.is_dir() else "orphaned"
            except (OSError, ValueError, json.JSONDecodeError):
                report["legacy_deployment_receipt"] = "invalid"
        else:
            report["legacy_deployment_receipt"] = "absent"
        return report

    def prepare(self, receipt_path: Path) -> LosslessUpdateReceipt:
        self._reject_profile_reparse_points()
        receipt_path = Path(receipt_path).resolve()
        if receipt_path.exists() or receipt_path.is_symlink():
            raise RuntimeError(f"recovery receipt already exists: {receipt_path}")
        recovery_root = receipt_path.parent / f"lossless-update-{uuid4().hex}"
        self._require_outside(recovery_root, self.hermes_home, "Hermes home")
        self._require_outside(recovery_root, self.host_root, "host")
        self._require_outside(recovery_root, self.soullink_root, "SoulLink")
        recovery_root.mkdir(parents=True)
        host_archive = recovery_root / "host.zip"
        profile_archive = recovery_root / "profile.zip"
        soullink_archive = recovery_root / "soullink.zip"
        sqlite_backups: list[tuple[Path, Path]] = []
        try:
            print("[1/7] Backing up Hermes host...", file=sys.stderr, flush=True)
            self._archive_tree(self.host_root, host_archive)
            print("[2/7] Backing up Hermes profile...", file=sys.stderr, flush=True)
            self._archive_profile(profile_archive)
            print("[3/7] Backing up SoulLink runtime...", file=sys.stderr, flush=True)
            self._archive_tree(
                self.soullink_root,
                soullink_archive,
                excluded=self.sqlite_paths,
                excluded_trees=tuple(self.soullink_root / path for path in self.soullink_preserved_entries),
            )
            sqlite_root = recovery_root / "sqlite"
            sqlite_root.mkdir()
            print("[4/7] Backing up SQLite databases...", file=sys.stderr, flush=True)
            for index, source in enumerate(self.sqlite_paths):
                if not source.is_file():
                    raise RuntimeError(f"SQLite source missing: {source}")
                backup = sqlite_root / f"{index:03d}-{source.name}"
                self._backup_sqlite(source, backup)
                sqlite_backups.append((source, backup))

            artifacts = (host_archive, profile_archive, soullink_archive, *(backup for _, backup in sqlite_backups))
            hashes = {str(path.relative_to(recovery_root)): self._sha256(path) for path in artifacts}
            self._verify_archives(host_archive, profile_archive, soullink_archive)
            for _, backup in sqlite_backups:
                self._check_sqlite(backup)
            marker = {
                "version": self.version,
                "soullink_root": str(self.soullink_root),
                "host_root": str(self.host_root),
                "hermes_home": str(self.hermes_home),
                "recovery_root": str(recovery_root),
                "host_archive": str(host_archive),
                "profile_archive": str(profile_archive),
                "soullink_archive": str(soullink_archive),
                "sqlite_backups": [[str(source), str(backup)] for source, backup in sqlite_backups],
                "hashes": hashes,
                "verified": True,
            }
            marker["authentication"] = self._authenticate(marker, create_key=True)
            self._atomic_json(recovery_root / "manifest.json", marker)
            self._atomic_json(receipt_path, marker)
            print("[4/7] Recovery point verified.", file=sys.stderr, flush=True)
            return self._receipt(marker)
        except BaseException:
            shutil.rmtree(recovery_root, ignore_errors=True)
            receipt_path.unlink(missing_ok=True)
            raise

    def execute(self, receipt_path: Path, *, update, deploy, verify) -> dict[str, object]:
        """Run an update transaction; restore the frozen state on any failure."""
        receipt_path = Path(receipt_path).resolve()
        preflight = self.preflight()
        if not preflight["ready"]:
            raise RuntimeError(f"lossless update preflight blocked: {preflight}")
        self.prepare(receipt_path)
        try:
            print("[5/7] Running official Hermes update...", file=sys.stderr, flush=True)
            update()
            print("[6/7] Deploying SoulLink adapter...", file=sys.stderr, flush=True)
            deploy()
            print("[7/7] Verifying SoulLink runtime...", file=sys.stderr, flush=True)
            if not verify():
                raise RuntimeError("post-update verification failed")
        except BaseException as exc:
            rolled_back = self.restore(receipt_path)
            return {
                "updated": False,
                "verified": False,
                "rolled_back": rolled_back,
                "activation_required": False,
                "receipt": str(receipt_path),
                "error": str(exc),
            }
        return {
            "updated": True,
            "verified": True,
            "rolled_back": False,
            "activation_required": True,
            "receipt": str(receipt_path),
        }

    def restore(self, receipt_path: Path) -> bool:
        receipt_path = Path(receipt_path).resolve()
        marker, receipt = self._validate_restore_receipt(receipt_path)
        self._reject_profile_reparse_points()
        compensation_path = receipt_path.parent / f".restore-compensation-{uuid4().hex}.json"
        compensation_controller = LosslessUpdateController(
            soullink_root=self.soullink_root,
            host_root=self.host_root,
            hermes_home=self.hermes_home,
            sqlite_paths=(),
            allowed_host_deltas=self.allowed_host_deltas,
        )
        compensation = compensation_controller.prepare(compensation_path)
        raw_sqlite = self._snapshot_raw_sqlite(compensation.recovery_root / "raw-sqlite")
        try:
            self._apply_restore(receipt)
        except BaseException as original:
            try:
                _, rollback_receipt = compensation_controller._validate_restore_receipt(compensation_path)
                compensation_controller._apply_restore(rollback_receipt)
                self._restore_raw_sqlite(raw_sqlite)
            except BaseException as compensation_error:
                raise RuntimeError(
                    f"restore failed and compensation failed; repair required: {compensation_error}"
                ) from original
            raise
        finally:
            if compensation_path.exists():
                compensation_path.unlink()
            shutil.rmtree(compensation.recovery_root, ignore_errors=True)
        return True

    def _snapshot_raw_sqlite(self, root: Path) -> tuple[tuple[Path, tuple[tuple[str, Path | None], ...]], ...]:
        root.mkdir()
        snapshots: list[tuple[Path, tuple[tuple[str, Path | None], ...]]] = []
        for index, source in enumerate(self.sqlite_paths):
            parts: list[tuple[str, Path | None]] = []
            for suffix in ("", "-wal", "-shm", "-journal"):
                current = Path(str(source) + suffix)
                saved: Path | None = None
                if current.is_file():
                    saved = root / f"{index:03d}{suffix or '-db'}"
                    shutil.copy2(current, saved)
                parts.append((suffix, saved))
            snapshots.append((source, tuple(parts)))
        return tuple(snapshots)

    @staticmethod
    def _restore_raw_sqlite(
        snapshots: tuple[tuple[Path, tuple[tuple[str, Path | None], ...]], ...]
    ) -> None:
        for destination, parts in snapshots:
            destination.parent.mkdir(parents=True, exist_ok=True)
            for suffix, saved in parts:
                current = Path(str(destination) + suffix)
                current.unlink(missing_ok=True)
                if saved is not None:
                    shutil.copy2(saved, current)

    def _reject_profile_reparse_points(self) -> None:
        configured = self._configured_hermes_home
        for path in (*reversed(configured.parents), configured):
            if self._is_reparse(path):
                raise RuntimeError(f"profile root is a symlink or reparse point: {path}")
        roots = [self.hermes_home / relative for relative in self.profile_entries]
        roots.append(self.hermes_home / "plugins")
        for root in roots:
            if self._is_reparse(root):
                raise RuntimeError(f"profile target is a symlink or reparse point: {root}")
            if not root.is_dir():
                continue
            for current, directories, files in os.walk(root, followlinks=False):
                for name in (*directories, *files):
                    path = Path(current) / name
                    if self._is_reparse(path):
                        raise RuntimeError(f"profile target is a symlink or reparse point: {path}")

    @staticmethod
    def _is_reparse(path: Path) -> bool:
        if path.is_symlink():
            return True
        try:
            return bool(path.lstat().st_file_attributes & 0x400)
        except (AttributeError, FileNotFoundError, OSError):
            return False

    def _validate_restore_receipt(self, receipt_path: Path) -> tuple[dict, LosslessUpdateReceipt]:
        marker = json.loads(receipt_path.read_text(encoding="utf-8"))
        supplied_authentication = marker.get("authentication")
        if not isinstance(supplied_authentication, str) or not hmac.compare_digest(
            supplied_authentication, self._authenticate(marker, create_key=False)
        ):
            raise RuntimeError("lossless update receipt authentication failed")
        receipt = self._receipt(marker)
        if receipt.version != self.version or not receipt.verified:
            raise RuntimeError("lossless update receipt is incompatible or unverified")
        if (
            receipt.host_root != self.host_root
            or receipt.hermes_home != self.hermes_home
            or receipt.soullink_root != self.soullink_root
        ):
            raise RuntimeError("lossless update receipt target mismatch")
        if receipt.recovery_root.parent != receipt_path.parent or not receipt.recovery_root.name.startswith(
            "lossless-update-"
        ):
            raise RuntimeError("lossless update recovery root mismatch")
        expected_archives = {
            "host_archive": receipt.recovery_root / "host.zip",
            "profile_archive": receipt.recovery_root / "profile.zip",
            "soullink_archive": receipt.recovery_root / "soullink.zip",
        }
        for field, expected in expected_archives.items():
            if Path(marker[field]).resolve() != expected.resolve():
                raise RuntimeError(f"recovery artifact path mismatch: {field}")
        if len(receipt.sqlite_backups) != len(self.sqlite_paths):
            raise RuntimeError("SQLite recovery binding mismatch")
        for index, ((source, backup), expected_source) in enumerate(zip(receipt.sqlite_backups, self.sqlite_paths)):
            expected_backup = receipt.recovery_root / "sqlite" / f"{index:03d}-{expected_source.name}"
            if source != expected_source or backup != expected_backup.resolve():
                raise RuntimeError("SQLite recovery binding mismatch")
        manifest = receipt.recovery_root / "manifest.json"
        if not manifest.is_file():
            raise RuntimeError("lossless update recovery manifest missing")
        canonical = json.loads(manifest.read_text(encoding="utf-8"))
        if canonical != marker:
            raise RuntimeError("lossless update receipt does not match recovery manifest")
        expected_hash_keys = {"host.zip", "profile.zip", "soullink.zip"} | {
            f"sqlite/{index:03d}-{path.name}" for index, path in enumerate(self.sqlite_paths)
        }
        hashes = {str(key).replace(chr(92), "/"): value for key, value in marker.get("hashes", {}).items()}
        if set(hashes) != expected_hash_keys:
            raise RuntimeError("recovery artifact hash manifest mismatch")
        for relative, expected in hashes.items():
            artifact = (receipt.recovery_root / relative).resolve()
            try:
                artifact.relative_to(receipt.recovery_root)
            except ValueError as exc:
                raise RuntimeError("recovery artifact escapes recovery root") from exc
            if not artifact.is_file() or self._sha256(artifact) != expected:
                raise RuntimeError(f"recovery artifact hash mismatch: {relative}")
        self._verify_archives(receipt.host_archive, receipt.profile_archive, receipt.soullink_archive)
        for _, backup in receipt.sqlite_backups:
            self._check_sqlite(backup)
        return marker, receipt

    def _apply_restore(self, receipt: LosslessUpdateReceipt) -> None:
        stage_token = uuid4().hex[:10]
        staged_host = receipt.host_root.parent / f".slh-{stage_token}"
        staged_profile = receipt.recovery_root / ".restore-profile"
        staged_soullink = receipt.recovery_root / ".restore-soullink"
        shutil.rmtree(staged_host, ignore_errors=True)
        shutil.rmtree(staged_profile, ignore_errors=True)
        shutil.rmtree(staged_soullink, ignore_errors=True)
        staged_host.mkdir()
        staged_profile.mkdir()
        staged_soullink.mkdir()
        try:
            self._extract(receipt.host_archive, staged_host)
            self._extract(receipt.profile_archive, staged_profile)
            self._extract(receipt.soullink_archive, staged_soullink)

            old_host = receipt.host_root.with_name(receipt.host_root.name + f".before-restore-{uuid4().hex}")
            os.replace(receipt.host_root, old_host)
            try:
                os.replace(staged_host / "root", receipt.host_root)
                self._restore_profile(staged_profile / "root")
                self._restore_soullink(staged_soullink / "root")
                for destination, backup in receipt.sqlite_backups:
                    destination.parent.mkdir(parents=True, exist_ok=True)
                    self._restore_sqlite(backup, destination)
                    self._check_sqlite(destination)
            except BaseException:
                if receipt.host_root.exists():
                    shutil.rmtree(receipt.host_root, ignore_errors=True)
                os.replace(old_host, receipt.host_root)
                raise
            shutil.rmtree(old_host, ignore_errors=True)
        finally:
            shutil.rmtree(staged_host, ignore_errors=True)
            shutil.rmtree(staged_profile, ignore_errors=True)
            shutil.rmtree(staged_soullink, ignore_errors=True)

    def _restore_soullink(self, staged_root: Path) -> None:
        protected = {path.resolve() for path in self.sqlite_paths}
        protected_sidecars = {
            Path(str(path) + suffix).resolve()
            for path in self.sqlite_paths
            for suffix in ("-wal", "-shm", "-journal")
        }
        preserved_roots = tuple((self.soullink_root / path).resolve() for path in self.soullink_preserved_entries)
        for path in sorted(self.soullink_root.rglob("*"), reverse=True):
            resolved = path.resolve()
            if (
                resolved in protected
                or resolved in protected_sidecars
                or any(resolved == root or root in resolved.parents for root in preserved_roots)
            ):
                continue
            if path.is_file() or path.is_symlink():
                path.unlink()
            elif path.is_dir():
                try:
                    path.rmdir()
                except OSError:
                    pass
        for child in staged_root.iterdir():
            target = self.soullink_root / child.name
            if child.is_dir():
                shutil.copytree(child, target, dirs_exist_ok=True)
            else:
                shutil.copy2(child, target)

    def _archive_profile(self, destination: Path) -> None:
        with zipfile.ZipFile(destination, "w", compression=zipfile.ZIP_DEFLATED) as archive:
            archive.writestr("root/", b"")
            for relative in self.profile_entries:
                source = self.hermes_home / relative
                if source.resolve() in self.sqlite_paths:
                    continue
                if source.is_file():
                    archive.write(source, Path("root") / relative)
                elif source.is_dir():
                    self._write_tree(archive, source, Path("root") / relative)
            plugins_root = self.hermes_home / "plugins"
            if plugins_root.is_dir():
                for source in sorted(plugins_root.iterdir()):
                    if source.resolve() == self.soullink_root:
                        continue
                    if source.is_dir():
                        self._write_tree(archive, source, Path("root/plugins") / source.name)
                    elif source.is_file():
                        archive.write(source, Path("root/plugins") / source.name)

    def _restore_profile(self, staged_root: Path) -> None:
        for relative in self.profile_entries:
            target = (self.hermes_home / relative).resolve()
            if target in self.sqlite_paths:
                continue
            self._restore_entry(staged_root / relative, target)
        plugins_root = self.hermes_home / "plugins"
        staged_plugins = staged_root / "plugins"
        plugins_root.mkdir(parents=True, exist_ok=True)
        for target in list(plugins_root.iterdir()):
            if target.resolve() == self.soullink_root:
                continue
            if target.is_dir():
                shutil.rmtree(target)
            else:
                target.unlink()
        if staged_plugins.is_dir():
            for staged in staged_plugins.iterdir():
                target = plugins_root / staged.name
                if staged.is_dir():
                    shutil.copytree(staged, target)
                else:
                    shutil.copy2(staged, target)

    @staticmethod
    def _restore_entry(staged: Path, target: Path) -> None:
        if target.is_dir():
            shutil.rmtree(target)
        elif target.exists() or target.is_symlink():
            target.unlink()
        if staged.is_dir():
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copytree(staged, target)
        elif staged.is_file():
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(staged, target)

    @staticmethod
    def _archive_tree(
        source: Path,
        destination: Path,
        *,
        excluded: tuple[Path, ...] = (),
        excluded_trees: tuple[Path, ...] = (),
    ) -> None:
        excluded_set: set[Path] = set()
        for path in excluded:
            resolved = path.resolve()
            excluded_set.update(
                {resolved, *(Path(str(resolved) + suffix) for suffix in ("-wal", "-shm", "-journal"))}
            )
        excluded_roots = tuple(path.resolve() for path in excluded_trees)
        with zipfile.ZipFile(destination, "w", compression=zipfile.ZIP_DEFLATED) as archive:
            LosslessUpdateController._write_tree(
                archive, source, Path("root"), excluded=excluded_set, excluded_roots=excluded_roots
            )

    @staticmethod
    def _write_tree(
        archive: zipfile.ZipFile,
        source: Path,
        prefix: Path,
        *,
        excluded: set[Path] | None = None,
        excluded_roots: tuple[Path, ...] = (),
    ) -> None:
        excluded = excluded or set()
        archive.writestr(prefix.as_posix().rstrip("/") + "/", b"")
        for path in sorted(source.rglob("*")):
            if LosslessUpdateController._is_reparse(path):
                resolved = path.resolve()
                try:
                    resolved.relative_to(source.resolve())
                except ValueError as exc:
                    raise RuntimeError(f"archive source escapes through a reparse point: {path}") from exc
                materialized_prefix = prefix / path.relative_to(source)
                if resolved.is_dir():
                    LosslessUpdateController._write_tree(
                        archive,
                        resolved,
                        materialized_prefix,
                        excluded=excluded,
                        excluded_roots=excluded_roots,
                    )
                elif resolved.is_file() and resolved not in excluded:
                    archive.write(resolved, materialized_prefix)
                continue
            resolved = path.resolve()
            if path.is_file() and resolved not in excluded and not any(
                resolved == root or root in resolved.parents for root in excluded_roots
            ):
                archive.write(path, prefix / path.relative_to(source))

    @staticmethod
    def _extract(archive_path: Path, destination: Path) -> None:
        with zipfile.ZipFile(archive_path) as archive:
            base = destination.resolve()
            for member in archive.infolist():
                target = (base / member.filename).resolve()
                try:
                    target.relative_to(base)
                except ValueError as exc:
                    raise RuntimeError("unsafe recovery archive member") from exc
            archive.extractall(destination)

    @staticmethod
    def _verify_archives(*paths: Path) -> None:
        for path in paths:
            with zipfile.ZipFile(path) as archive:
                if archive.testzip() is not None:
                    raise RuntimeError(f"recovery archive failed CRC verification: {path}")

    @staticmethod
    def _backup_sqlite(source: Path, destination: Path) -> None:
        with contextlib.closing(sqlite3.connect(source)) as src, contextlib.closing(
            sqlite3.connect(destination)
        ) as dst:
            src.backup(dst)

    @staticmethod
    def _restore_sqlite(backup: Path, destination: Path) -> None:
        for suffix in ("", "-wal", "-shm", "-journal"):
            Path(str(destination) + suffix).unlink(missing_ok=True)
        with contextlib.closing(sqlite3.connect(backup)) as src, contextlib.closing(
            sqlite3.connect(destination)
        ) as dst:
            src.backup(dst)
        for suffix in ("-wal", "-shm", "-journal"):
            Path(str(destination) + suffix).unlink(missing_ok=True)

    @staticmethod
    def _check_sqlite(path: Path) -> None:
        with contextlib.closing(sqlite3.connect(path)) as conn:
            result = conn.execute("pragma integrity_check").fetchone()
        if not result or result[0] != "ok":
            raise RuntimeError(f"SQLite integrity check failed: {path}")

    @staticmethod
    def _sha256(path: Path) -> str:
        digest = hashlib.sha256()
        with path.open("rb") as handle:
            for chunk in iter(lambda: handle.read(1024 * 1024), b""):
                digest.update(chunk)
        return digest.hexdigest()

    def _authenticate(self, marker: dict, *, create_key: bool) -> str:
        key_path = self._auth_key_path
        if self._is_reparse(key_path):
            raise RuntimeError("lossless update authentication key is unsafe")
        try:
            key = key_path.read_bytes()
        except FileNotFoundError:
            if not create_key:
                raise RuntimeError("lossless update authentication key is missing")
            key_path.parent.mkdir(parents=True, exist_ok=True)
            key = secrets.token_bytes(32)
            try:
                flags = os.O_CREAT | os.O_EXCL | os.O_WRONLY | getattr(os, "O_BINARY", 0)
                fd = os.open(key_path, flags, 0o600)
            except FileExistsError:
                key = key_path.read_bytes()
            else:
                try:
                    os.write(fd, key)
                finally:
                    os.close(fd)
        if len(key) != 32:
            raise RuntimeError("lossless update authentication key is invalid")
        payload = {key: value for key, value in marker.items() if key != "authentication"}
        canonical = json.dumps(
            payload, ensure_ascii=False, sort_keys=True, separators=(",", ":")
        ).encode("utf-8")
        return hmac.new(key, canonical, hashlib.sha256).hexdigest()

    @staticmethod
    def _atomic_json(path: Path, data: dict) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        temp = path.with_name(path.name + ".tmp")
        temp.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")
        os.replace(temp, path)

    @staticmethod
    def _receipt(data: dict) -> LosslessUpdateReceipt:
        return LosslessUpdateReceipt(
            version=str(data["version"]),
            soullink_root=Path(data["soullink_root"]).resolve(),
            host_root=Path(data["host_root"]).resolve(),
            hermes_home=Path(data["hermes_home"]).resolve(),
            recovery_root=Path(data["recovery_root"]).resolve(),
            host_archive=Path(data["host_archive"]).resolve(),
            profile_archive=Path(data["profile_archive"]).resolve(),
            soullink_archive=Path(data["soullink_archive"]).resolve(),
            sqlite_backups=tuple((Path(a).resolve(), Path(b).resolve()) for a, b in data["sqlite_backups"]),
            verified=bool(data["verified"]),
        )

    @staticmethod
    def _require_outside(path: Path, root: Path, label: str) -> None:
        try:
            path.resolve().relative_to(root.resolve())
        except ValueError:
            return
        raise RuntimeError(f"recovery path must be outside {label} root")


def build_controller(soullink_root: Path, host_root: Path, hermes_home: Path) -> LosslessUpdateController:
    root = Path(soullink_root).resolve()
    manifest_path = root / "adapters/hermes/compatibility-soullink-runtime.yaml"
    data = yaml.safe_load(manifest_path.read_text(encoding="utf-8")) or {}
    allowed = [str(path).replace(chr(92), "/") for key in ("required_paths", "created_paths") for path in data.get(key, [])]
    allowed.append("plugins/context_engine/pcltm-context/")
    database = root / "var/pcltm-prod.db"
    state_db = Path(hermes_home).resolve() / "state.db"
    sqlite_paths = (database.resolve(), state_db.resolve())
    return LosslessUpdateController(
        soullink_root=root,
        host_root=host_root,
        hermes_home=hermes_home,
        sqlite_paths=sqlite_paths,
        allowed_host_deltas=tuple(allowed),
    )


def _run_checked(command: list[str], *, cwd: Path | None = None, env: dict[str, str] | None = None) -> None:
    result = subprocess.run(
        command,
        cwd=cwd,
        env=env,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        check=False,
    )
    if result.stdout:
        print(result.stdout, file=sys.stderr, end="")
    if result.stderr:
        print(result.stderr, file=sys.stderr, end="")
    if result.returncode != 0:
        raise RuntimeError(f"command failed with exit code {result.returncode}: {command}")


def _update_managed_host(
    host_root: Path,
    executable: Path,
    env: dict[str, str],
    managed_paths: tuple[str, ...],
) -> None:
    """Normalize a shallow/divergent managed checkout, then let Hermes repair its venv.

    This is called only after ``execute`` has durably verified a complete recovery
    point.  The owner manifest is the authority for the dirty paths removed here.
    """
    root = Path(host_root).resolve()
    _run_checked(["git", "fetch", "--depth", "2", "origin", "main"], cwd=root, env=env)
    _run_checked(["git", "rev-parse", "--verify", "origin/main^"], cwd=root, env=env)
    _run_checked(["git", "reset", "--hard", "origin/main^"], cwd=root, env=env)
    if managed_paths:
        _run_checked(["git", "clean", "-fd", "--", *managed_paths], cwd=root, env=env)
    refreshed = next(
        (
            candidate
            for candidate in (
                root / "venv" / "Scripts" / "hermes.exe",
                root / "venv" / "Scripts" / "hermes",
                root / "venv" / "bin" / "hermes",
            )
            if candidate.is_file()
        ),
        None,
    )
    launcher = refreshed if refreshed is not None else Path(executable)
    _run_checked([str(launcher), "update", "--yes", "--no-backup"], cwd=root, env=env)


def _assert_hermes_stopped(action: str) -> None:
    if sys.platform != "win32":
        return
    running = subprocess.run(
        ["tasklist", "/FI", "IMAGENAME eq Hermes.exe", "/NH"],
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        check=False,
    )
    if "Hermes.exe" in running.stdout:
        raise RuntimeError(f"Hermes Desktop is running; close it before lossless {action}")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="soullink-hermes-update")
    parser.add_argument("action", choices=("preflight", "prepare", "run", "restore"))
    parser.add_argument("--soullink-root", type=Path, required=True)
    parser.add_argument("--host-root", type=Path, required=True)
    parser.add_argument("--hermes-home", type=Path, required=True)
    parser.add_argument("--receipt", type=Path)
    args = parser.parse_args(argv)
    controller = build_controller(args.soullink_root, args.host_root, args.hermes_home)
    try:
        if args.action == "preflight":
            payload = controller.preflight()
        elif args.action == "prepare":
            if args.receipt is None:
                raise RuntimeError("--receipt is required")
            _assert_hermes_stopped("recovery-point preparation")
            with controller.update_lock():
                receipt = controller.prepare(args.receipt)
            payload = {"prepared": receipt.verified, "receipt": str(args.receipt.resolve())}
        elif args.action == "restore":
            if args.receipt is None:
                raise RuntimeError("--receipt is required")
            _assert_hermes_stopped("restore")
            with controller.update_lock():
                payload = {"restored": controller.restore(args.receipt)}
        else:
            if args.receipt is None:
                raise RuntimeError("--receipt is required")
            _assert_hermes_stopped("update activation")
            from soul_link.hermes_deploy import HermesDeployment

            deployment = HermesDeployment(args.soullink_root)
            executable = next(
                (
                    candidate
                    for candidate in (
                        args.host_root.resolve() / "venv" / "Scripts" / "hermes.exe",
                        args.host_root.resolve() / "venv" / "Scripts" / "hermes",
                        args.host_root.resolve() / "venv" / "bin" / "hermes",
                    )
                    if candidate.is_file()
                ),
                None,
            )
            if executable is None:
                raise RuntimeError("target Hermes executable missing from host virtual environment")
            update_env = os.environ.copy()
            update_env["HERMES_HOME"] = str(args.hermes_home.resolve())
            with controller.update_lock():
                payload = controller.execute(
                    args.receipt,
                    update=lambda: _update_managed_host(
                        args.host_root.resolve(), executable, update_env, controller.allowed_host_deltas
                    ),
                    deploy=lambda: deployment.apply(args.host_root, args.hermes_home),
                    verify=lambda: deployment.verify(args.host_root, args.hermes_home),
                )
        print(json.dumps(payload, ensure_ascii=False))
        if args.action == "preflight":
            return 0 if payload.get("ready") is True else 2
        if args.action == "prepare":
            return 0 if payload.get("prepared") is True else 3
        if args.action == "restore":
            return 0 if payload.get("restored") is True else 4
        return 0 if payload.get("updated") is True and payload.get("verified") is True else 6
    except Exception as exc:
        print(json.dumps({"success": False, "error": str(exc)}, ensure_ascii=False))
        return 5


__all__ = ["LosslessUpdateController", "LosslessUpdateReceipt", "build_controller", "main"]


if __name__ == "__main__":
    raise SystemExit(main())
