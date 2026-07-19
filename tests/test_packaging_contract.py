from __future__ import annotations

import tomllib
from pathlib import Path


def test_portable_runtime_subpackages_are_in_wheel_configuration() -> None:
    root = Path(__file__).resolve().parents[1]
    config = tomllib.loads((root / "pyproject.toml").read_text(encoding="utf-8"))
    setuptools = config["tool"]["setuptools"]

    assert "soul_link.integration" in setuptools["packages"]
    assert setuptools["package-dir"]["soul_link.integration"] == "soul_link/integration"
    assert "pcltm.projections" in setuptools["packages"]
    assert setuptools["package-dir"]["pcltm.projections"] == "packages/pcltm/projections"
