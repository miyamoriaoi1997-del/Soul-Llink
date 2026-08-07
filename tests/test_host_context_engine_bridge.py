from __future__ import annotations

import importlib
from pathlib import Path

import pytest


@pytest.fixture
def isolated_external_context_plugin(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> tuple[Path, Path]:
    """Install the external bridge into an isolated Hermes home."""
    soullink_root = Path(__file__).resolve().parents[1]
    plugin_dir = tmp_path / "plugins" / "pcltm-context"
    plugin_dir.mkdir(parents=True)
    plugin_dir.joinpath("__init__.py").write_text(
        (
            "from pathlib import Path\n"
            "import os\n"
            "import sys\n\n"
            "_ROOT_FILE = Path(__file__).with_name('soullink-root.txt')\n"
            "if not _ROOT_FILE.is_file():\n"
            "    raise RuntimeError('PCLTM context plugin is incomplete: soullink-root.txt is missing')\n"
            "_SOULLINK_ROOT = Path(_ROOT_FILE.read_text(encoding='utf-8').strip()).expanduser().resolve()\n"
            "if not (_SOULLINK_ROOT / 'packages' / 'pcltm').is_dir():\n"
            "    raise RuntimeError(f'SoulLink root is invalid: {_SOULLINK_ROOT}')\n"
            "os.environ['SOULLINK_ROOT'] = str(_SOULLINK_ROOT)\n"
            "for _path in (_SOULLINK_ROOT, _SOULLINK_ROOT / 'packages'):\n"
            "    if str(_path) not in sys.path:\n"
            "        sys.path.insert(0, str(_path))\n\n"
            "from soul_link.hermes_plugin.context_engine import PCLTMContextCompressionEngine\n\n"
            "def register(ctx) -> None:\n"
            "    ctx.register_context_engine(PCLTMContextCompressionEngine())\n"
        ),
        encoding="utf-8",
    )
    plugin_dir.joinpath("plugin.yaml").write_text(
        "manifest_version: 1\n"
        "name: pcltm-context\n"
        "version: 2.0.0\n"
        "description: SoulLink/PCLTM governed context engine for Hermes Agent.\n"
        "kind: standalone\n",
        encoding="utf-8",
    )
    plugin_dir.joinpath("soullink-root.txt").write_text(
        str(soullink_root), encoding="utf-8"
    )
    config = tmp_path / "config.yaml"
    config.write_text("plugins:\n  enabled:\n    - pcltm-context\n", encoding="utf-8")
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    monkeypatch.setenv("HERMES_CONFIG_FILE", str(config))
    monkeypatch.delenv("LOCALAPPDATA", raising=False)
    return tmp_path, plugin_dir


def test_external_context_plugin_register_resolves_pcltm_engine(
    isolated_external_context_plugin: tuple[Path, Path],
) -> None:
    pytest.importorskip("hermes_cli", reason="set HERMES_HOST_ROOT for host integration tests")
    _, plugin_dir = isolated_external_context_plugin
    assert (plugin_dir / "__init__.py").is_file()
    assert (plugin_dir / "plugin.yaml").is_file()

    import hermes_cli.config as config_host
    import hermes_cli.plugins as plugin_host

    config_host = importlib.reload(config_host)
    plugin_host = importlib.reload(plugin_host)
    manager = plugin_host.PluginManager()
    manager.discover_and_load()
    engine = manager._context_engine

    assert engine is not None
    assert type(engine).__name__ == "PCLTMContextCompressionEngine"
    assert engine.name == "pcltm-context"


def test_external_context_plugin_marker_points_to_valid_soullink_root(
    isolated_external_context_plugin: tuple[Path, Path],
) -> None:
    _, plugin_dir = isolated_external_context_plugin
    marker = plugin_dir / "soullink-root.txt"
    assert marker.is_file()
    root = Path(marker.read_text(encoding="utf-8").strip())
    assert (root / "packages" / "pcltm").is_dir()


def test_repository_local_context_bridge_is_absent(
    isolated_external_context_plugin: tuple[Path, Path],
) -> None:
    hermes_home, _ = isolated_external_context_plugin
    bridge = hermes_home / "hermes-agent" / "plugins" / "context_engine" / "pcltm-context"
    assert not bridge.exists(), "repository-local context bridge must remain externalized"
