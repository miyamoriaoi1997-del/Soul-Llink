from __future__ import annotations

import tomllib
from pathlib import Path


def test_portable_runtime_subpackages_are_in_wheel_configuration() -> None:
    root = Path(__file__).resolve().parents[1]
    config = tomllib.loads((root / "pyproject.toml").read_text(encoding="utf-8"))
    setuptools = config["tool"]["setuptools"]

    assert "soul_link.integration" in setuptools["packages"]
    assert setuptools["package-dir"]["soul_link.integration"] == "soul_link/integration"


def test_all_regular_runtime_packages_are_in_wheel_configuration() -> None:
    root = Path(__file__).resolve().parents[1]
    config = tomllib.loads((root / "pyproject.toml").read_text(encoding="utf-8"))
    declared = set(config["tool"]["setuptools"]["packages"])
    package_roots = {
        "model_router": root / "packages" / "model_router",
        "pcltm": root / "packages" / "pcltm",
        "persona_engine": root / "packages" / "persona_engine",
        "soul_link": root / "soul_link",
    }

    regular_packages = set()
    for package_name, package_root in package_roots.items():
        for init_file in package_root.rglob("__init__.py"):
            relative = init_file.parent.relative_to(package_root)
            suffix = ".".join(relative.parts)
            regular_packages.add(package_name + (f".{suffix}" if suffix else ""))

    assert regular_packages <= declared


def test_obsolete_root_deployment_manifest_is_retired() -> None:
    root = Path(__file__).resolve().parents[1]

    assert not (root / "compute_manifest.py").exists()
    assert not (root / "DEPLOYMENT_MANIFEST.json").exists()
