"""Path safety tests for _inside() method.

Tests that _inside() rejects unsafe paths before resolve/relative_to runtime checks.
"""
from pathlib import Path

import pytest

from soul_link.hermes_deploy import HermesDeployment
from soul_link.host_adaptation import CompatibilityManifest


def test_inside_rejects_empty_path(tmp_path: Path) -> None:
    """_inside must reject empty path string."""
    with pytest.raises(RuntimeError, match="unsafe managed path|empty"):
        HermesDeployment._inside(tmp_path, "")


def test_inside_rejects_posix_absolute_path(tmp_path: Path) -> None:
    """_inside must reject POSIX absolute paths."""
    with pytest.raises(RuntimeError, match="unsafe managed path|absolute"):
        HermesDeployment._inside(tmp_path, "/etc/passwd")


def test_inside_rejects_windows_drive_path(tmp_path: Path) -> None:
    """_inside must reject Windows drive-qualified paths."""
    with pytest.raises(RuntimeError, match="unsafe managed path|drive|absolute"):
        HermesDeployment._inside(tmp_path, "C:\\Windows\\System32")


def test_inside_rejects_windows_unc_path(tmp_path: Path) -> None:
    """_inside must reject Windows UNC paths."""
    with pytest.raises(RuntimeError, match="unsafe managed path|absolute|anchor"):
        HermesDeployment._inside(tmp_path, "\\\\server\\share\\file")


def test_inside_rejects_parent_traversal(tmp_path: Path) -> None:
    """_inside must reject paths with .. components."""
    with pytest.raises(RuntimeError, match="unsafe managed path|traversal"):
        HermesDeployment._inside(tmp_path, "plugins/../../etc/shadow")


def test_inside_rejects_path_with_anchor(tmp_path: Path) -> None:
    """_inside must reject paths with drive/anchor."""
    path = Path("C:/")  # Has anchor
    if path.anchor:
        with pytest.raises(RuntimeError, match="unsafe managed path"):
            HermesDeployment._inside(tmp_path, str(path))


def test_inside_accepts_safe_relative_path(tmp_path: Path) -> None:
    """_inside must accept safe relative paths."""
    result = HermesDeployment._inside(tmp_path, "plugins/soullink")
    assert result.is_relative_to(tmp_path)
    assert result == (tmp_path / "plugins/soullink").resolve()


@pytest.mark.parametrize(
    "unsafe",
    ("C:\\Windows\\System32", "\\\\server\\share\\file", "plugins\\..\\outside.py"),
)
def test_manifest_rejects_windows_paths_cross_platform(tmp_path: Path, unsafe: str) -> None:
    """Manifest paths are checked with Windows semantics on every build platform."""
    with pytest.raises(ValueError, match="unsafe host path"):
        CompatibilityManifest(
            host="hermes",
            adapter_version="test",
            required_paths=(unsafe,),
            patch_path=(tmp_path / "adapter.patch").resolve(),
        )
