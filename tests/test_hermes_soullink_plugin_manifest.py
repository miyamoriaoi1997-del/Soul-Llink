from __future__ import annotations

import logging
import os
import sys
from pathlib import Path

import pytest
import yaml

HERMES_AGENT_ROOT_VALUE = os.environ.get("SOULLINK_HERMES_AGENT_ROOT")
if not HERMES_AGENT_ROOT_VALUE:
    pytest.skip(
        "set SOULLINK_HERMES_AGENT_ROOT to run installed-host manifest tests",
        allow_module_level=True,
    )
HERMES_AGENT_ROOT = Path(HERMES_AGENT_ROOT_VALUE)
if str(HERMES_AGENT_ROOT) not in sys.path:
    sys.path.insert(0, str(HERMES_AGENT_ROOT))

from hermes_cli.plugins import PluginManager  # noqa: E402


def test_soullink_manifest_uses_host_recognized_exclusive_kind(caplog):
    repo_root = Path(__file__).resolve().parents[1]
    manifest_paths = (
        repo_root / "adapters/hermes/soullink-plugin.yaml",
        repo_root / "adapters/hermes/memory_provider/plugin.yaml",
    )
    for manifest_path in manifest_paths:
        data = yaml.safe_load(manifest_path.read_text(encoding="utf-8"))
        assert data["name"] == "soullink"
        assert data["kind"] == "exclusive"

    manager = PluginManager()
    with caplog.at_level(logging.WARNING):
        manifests = [
            manager._parse_manifest(path, path.parent, "user", "")
            for path in manifest_paths
        ]

    assert all(manifest is not None for manifest in manifests)
    assert all(manifest.kind == "exclusive" for manifest in manifests)
    assert not any("unknown kind" in record.getMessage() for record in caplog.records)
