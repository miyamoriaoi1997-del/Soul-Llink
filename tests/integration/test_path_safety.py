"""Path safety tests for _inside() method.

Tests that _inside() rejects unsafe paths before resolve/relative_to runtime checks.
"""
from pathlib import Path

import pytest

from soul_link.hermes_deploy import DeploymentReceipt, HermesDeployment
from soul_link.host_adaptation import AdaptationReceipt, CompatibilityManifest, HostAdapterController
from soul_link.hermes_update import LosslessUpdateController, build_controller


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


def test_host_adapter_rejects_symlinked_host_root(tmp_path: Path) -> None:
    actual = tmp_path / "actual-host"
    actual.mkdir()
    (actual / "required.py").write_text("safe", encoding="utf-8")
    linked = tmp_path / "linked-host"
    linked.symlink_to(actual, target_is_directory=True)
    patch = tmp_path / "adapter.patch"
    patch.write_text("", encoding="utf-8")
    controller = HostAdapterController(
        CompatibilityManifest("hermes", "test", ("required.py",), patch),
        command_runner=lambda *_: 0,
    )

    with pytest.raises(RuntimeError, match="symlink or reparse"):
        controller.detect(linked)


def test_host_adapter_detect_reports_missing_root_as_incompatible(tmp_path: Path) -> None:
    missing = tmp_path / "missing-host"
    patch = tmp_path / "adapter.patch"
    patch.write_text("", encoding="utf-8")
    controller = HostAdapterController(
        CompatibilityManifest("hermes", "test", ("required.py",), patch),
        command_runner=lambda *_: 1,
    )

    result = controller.detect(missing)

    assert result.classification == "incompatible"
    assert result.patch_state == "not_checked"
    assert result.missing_paths == controller.manifest.required_paths


def test_host_adapter_rejects_symlinked_managed_file(tmp_path: Path) -> None:
    host = tmp_path / "host"
    outside = tmp_path / "outside.py"
    host.mkdir()
    outside.write_text("external", encoding="utf-8")
    (host / "required.py").symlink_to(outside)
    patch = tmp_path / "adapter.patch"
    patch.write_text("", encoding="utf-8")
    controller = HostAdapterController(
        CompatibilityManifest("hermes", "test", ("required.py",), patch),
        command_runner=lambda *_: 0,
    )

    with pytest.raises(RuntimeError, match="symlink or reparse"):
        controller.detect(host)


def test_hermes_deployment_rejects_symlinked_home(tmp_path: Path) -> None:
    actual = tmp_path / "actual-home"
    actual.mkdir()
    linked = tmp_path / "linked-home"
    linked.symlink_to(actual, target_is_directory=True)

    with pytest.raises(RuntimeError, match="symlink or reparse"):
        HermesDeployment._inside(linked, "config.yaml")


def test_hermes_deployment_detect_reports_missing_host_as_incompatible(tmp_path: Path) -> None:
    deployment = HermesDeployment(Path(__file__).resolve().parents[2])

    state = deployment.detect(tmp_path / "missing-host", tmp_path / "home")

    assert state["classification"] == "incompatible"
    assert state["host_adaptation"]["patch_state"] == "not_checked"
    assert state["missing_host_paths"] == list(deployment.host_contract)


@pytest.mark.parametrize("root_name", ("host", "soullink"))
def test_lossless_update_rejects_symlinked_roots(tmp_path: Path, root_name: str) -> None:
    actual_host = tmp_path / "actual-host"
    actual_soullink = tmp_path / "actual-soullink"
    home = tmp_path / "home"
    actual_host.mkdir()
    actual_soullink.mkdir()
    home.mkdir()
    linked = tmp_path / f"linked-{root_name}"
    linked.symlink_to(
        actual_host if root_name == "host" else actual_soullink,
        target_is_directory=True,
    )

    with pytest.raises(RuntimeError, match="symlink or reparse"):
        LosslessUpdateController(
            soullink_root=linked if root_name == "soullink" else actual_soullink,
            host_root=linked if root_name == "host" else actual_host,
            hermes_home=home,
        )


def test_lossless_update_factory_rejects_symlinked_soullink_root(tmp_path: Path) -> None:
    actual = Path(__file__).resolve().parents[2]
    linked = tmp_path / "linked-soullink"
    linked.symlink_to(actual, target_is_directory=True)
    host = tmp_path / "host"
    home = tmp_path / "home"
    host.mkdir()
    home.mkdir()

    with pytest.raises(RuntimeError, match="symlink or reparse"):
        build_controller(linked, host, home)


def test_lossless_update_rejects_symlinked_hermes_home_at_construction(tmp_path: Path) -> None:
    soullink = tmp_path / "soullink"
    host = tmp_path / "host"
    actual_home = tmp_path / "actual-home"
    linked_home = tmp_path / "linked-home"
    soullink.mkdir()
    host.mkdir()
    actual_home.mkdir()
    linked_home.symlink_to(actual_home, target_is_directory=True)

    with pytest.raises(RuntimeError, match="symlink or reparse"):
        LosslessUpdateController(
            soullink_root=soullink,
            host_root=host,
            hermes_home=linked_home,
        )


def test_lossless_update_factory_rejects_symlinked_hermes_home(tmp_path: Path) -> None:
    root = Path(__file__).resolve().parents[2]
    host = tmp_path / "host"
    actual_home = tmp_path / "actual-home"
    linked_home = tmp_path / "linked-home"
    host.mkdir()
    actual_home.mkdir()
    linked_home.symlink_to(actual_home, target_is_directory=True)

    with pytest.raises(RuntimeError, match="symlink or reparse"):
        build_controller(root, host, linked_home)


def test_deployment_receipt_load_preserves_reparse_evidence(tmp_path: Path) -> None:
    actual_home = tmp_path / "actual-home"
    linked_home = tmp_path / "linked-home"
    actual_home.mkdir()
    linked_home.symlink_to(actual_home, target_is_directory=True)
    receipt_path = tmp_path / "deployment.json"
    receipt_path.write_text(
        __import__("json").dumps(
            {
                "host_root": str(tmp_path / "host"),
                "hermes_home": str(linked_home),
                "soullink_root": str(tmp_path / "soullink"),
                "backup_path": str(linked_home / ".soullink-deploy-backup-test"),
                "host_adaptation_receipt": "",
                "adapter_version": "3",
                "fingerprints": {},
                "entries": {},
            }
        ),
        encoding="utf-8",
    )

    receipt = DeploymentReceipt.load(receipt_path)
    assert receipt.hermes_home == linked_home.absolute()
    with pytest.raises(RuntimeError, match="symlink or reparse"):
        HermesDeployment(Path(__file__).resolve().parents[2]).rollback(receipt)


def test_adaptation_receipt_load_preserves_reparse_evidence(tmp_path: Path) -> None:
    actual_host = tmp_path / "actual-host"
    linked_host = tmp_path / "linked-host"
    actual_host.mkdir()
    linked_host.symlink_to(actual_host, target_is_directory=True)
    receipt_path = tmp_path / "adaptation.json"
    receipt_path.write_text(
        __import__("json").dumps(
            {
                "host_root": str(linked_host),
                "backup_path": str(linked_host / ".soullink-adapter-backup-test"),
                "adapter_version": "test",
                "fingerprints": {},
            }
        ),
        encoding="utf-8",
    )
    patch = tmp_path / "adapter.patch"
    patch.write_text("", encoding="utf-8")
    controller = HostAdapterController(
        CompatibilityManifest("hermes", "test", (), patch),
        command_runner=lambda *_: 0,
    )

    receipt = AdaptationReceipt.load(receipt_path)
    assert receipt.host_root == linked_host.absolute()
    with pytest.raises(RuntimeError, match="symlink or reparse"):
        controller.rollback(receipt)


def test_host_adapter_rollback_rejects_symlinked_marker(tmp_path: Path) -> None:
    host = tmp_path / "host"
    backup = host / ".soullink-adapter-backup-test"
    backup.mkdir(parents=True)
    external = tmp_path / "external-adaptation-marker.json"
    external.write_text("{}", encoding="utf-8")
    (backup / ".soullink-backup.json").symlink_to(external)
    patch = tmp_path / "adapter.patch"
    patch.write_text("", encoding="utf-8")
    controller = HostAdapterController(
        CompatibilityManifest("hermes", "test", (), patch),
        command_runner=lambda *_: 0,
    )
    receipt = AdaptationReceipt(host, backup, "test", {})

    with pytest.raises(RuntimeError, match="invalid or missing adaptation backup"):
        controller.rollback(receipt)

    assert backup.is_dir()
    assert external.is_file()


def test_deployment_rollback_rejects_symlinked_marker(tmp_path: Path) -> None:
    root = Path(__file__).resolve().parents[2]
    home = tmp_path / "home"
    backup = home / ".soullink-deploy-backup-test"
    host = tmp_path / "host"
    home.mkdir()
    host.mkdir()
    backup.mkdir()
    external = tmp_path / "external-deployment-marker.json"
    external.write_text("{}", encoding="utf-8")
    (backup / ".soullink-deploy.json").symlink_to(external)
    receipt = DeploymentReceipt(
        host,
        home,
        root,
        backup,
        None,
        "3",
        {},
        {},
    )

    with pytest.raises(RuntimeError, match="symlink or reparse"):
        HermesDeployment(root).rollback(receipt)

    assert backup.is_dir()
    assert external.is_file()
