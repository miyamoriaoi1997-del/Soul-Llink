"""Versioned, atomic SoulLink runtime-config persistence with rollback receipts."""

from __future__ import annotations

import hashlib
import os
import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml

CURRENT_RUNTIME_CONFIG_SCHEMA = 1


def _sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


@dataclass(frozen=True)
class RuntimeConfigReceipt:
    config_path: Path
    backup_path: Path
    existed_before: bool
    schema_from: int
    schema_to: int
    before_sha256: str
    after_sha256: str


class RuntimeConfigStore:
    def __init__(self, path: str | Path):
        self.path = Path(path).resolve()

    def load(self) -> dict[str, Any]:
        if not self.path.is_file():
            return {}
        try:
            loaded = yaml.safe_load(self.path.read_text(encoding="utf-8-sig"))
        except (OSError, UnicodeError, yaml.YAMLError) as exc:
            raise RuntimeError(f"invalid runtime config: {self.path}") from exc
        config = {} if loaded is None else loaded
        if not isinstance(config, dict):
            raise RuntimeError(f"runtime config must be a mapping: {self.path}")
        return config

    @staticmethod
    def _soullink_entry(config: dict[str, Any]) -> dict[str, Any]:
        plugins = config.setdefault("plugins", {})
        if not isinstance(plugins, dict):
            raise RuntimeError("runtime config plugins must be a mapping")
        entries = plugins.setdefault("entries", {})
        if not isinstance(entries, dict):
            raise RuntimeError("runtime config plugin entries must be a mapping")
        soullink = entries.setdefault("soullink", {})
        if not isinstance(soullink, dict):
            raise RuntimeError("SoulLink runtime config must be a mapping")
        return soullink

    def migrate(self, config: dict[str, Any]) -> tuple[dict[str, Any], int]:
        migrated = dict(config)
        soullink = self._soullink_entry(migrated)
        raw_version = soullink.get("schema_version", 0)
        if isinstance(raw_version, bool) or not isinstance(raw_version, int):
            raise RuntimeError("SoulLink runtime config schema_version must be an integer")
        if raw_version < 0 or raw_version > CURRENT_RUNTIME_CONFIG_SCHEMA:
            raise RuntimeError(f"unsupported SoulLink runtime config schema_version: {raw_version}")
        schema_from = raw_version
        if raw_version == 0:
            soullink["schema_version"] = CURRENT_RUNTIME_CONFIG_SCHEMA
        return migrated, schema_from

    def save(self, config: dict[str, Any]) -> RuntimeConfigReceipt:
        migrated, schema_from = self.migrate(config)
        existed_before = self.path.is_file()
        before = self.path.read_bytes() if existed_before else b""
        rendered = yaml.safe_dump(migrated, sort_keys=False, allow_unicode=True).encode("utf-8")
        self.path.parent.mkdir(parents=True, exist_ok=True)
        backup = self.path.with_name(f".{self.path.name}.soullink-backup-{uuid.uuid4().hex}")
        temp = self.path.with_name(f".{self.path.name}.soullink-tmp-{uuid.uuid4().hex}")
        try:
            backup.write_bytes(before)
            temp.write_bytes(rendered)
            os.replace(temp, self.path)
        except BaseException:
            temp.unlink(missing_ok=True)
            backup.unlink(missing_ok=True)
            raise
        return RuntimeConfigReceipt(
            config_path=self.path,
            backup_path=backup,
            existed_before=existed_before,
            schema_from=schema_from,
            schema_to=CURRENT_RUNTIME_CONFIG_SCHEMA,
            before_sha256=_sha256(before),
            after_sha256=_sha256(rendered),
        )

    def rollback(self, receipt: RuntimeConfigReceipt) -> None:
        if receipt.config_path.resolve() != self.path:
            raise RuntimeError("rollback receipt config path mismatch")
        if not self.path.is_file() or _sha256(self.path.read_bytes()) != receipt.after_sha256:
            raise RuntimeError("current config hash mismatch")
        if not receipt.backup_path.is_file():
            raise RuntimeError("rollback backup is missing")
        backup = receipt.backup_path.read_bytes()
        if _sha256(backup) != receipt.before_sha256:
            raise RuntimeError("rollback backup hash mismatch")
        if receipt.existed_before:
            temp = self.path.with_name(f".{self.path.name}.soullink-rollback-{uuid.uuid4().hex}")
            temp.write_bytes(backup)
            os.replace(temp, self.path)
        else:
            self.path.unlink()
        receipt.backup_path.unlink()


__all__ = [
    "CURRENT_RUNTIME_CONFIG_SCHEMA",
    "RuntimeConfigReceipt",
    "RuntimeConfigStore",
]
