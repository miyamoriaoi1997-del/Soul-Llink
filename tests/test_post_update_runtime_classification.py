from __future__ import annotations

import hashlib
from pathlib import Path

from soul_link.hermes_deploy import HermesDeployment
from soul_link.host_adaptation import CompatibilityResult


ROOT = Path(__file__).resolve().parents[1]


class _DegradedAuthorityController:
    def detect(self, host: Path) -> CompatibilityResult:
        return CompatibilityResult("incompatible", "mismatch", ())


def _host(tmp_path: Path) -> Path:
    host = tmp_path / "hermes"
    files = {
        "agent/context_engine.py": "class ContextEngine: pass\n",
        "agent/memory_provider.py": "class MemoryProvider: pass\n",
        "plugins/memory/__init__.py": "def load_memory_provider(): pass\n",
        "hermes_cli/plugins.py": "def register_context_engine(): pass\n",
    }
    for relative, text in files.items():
        path = host / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(text, encoding="utf-8")
    return host


def _installed_home(deployment: HermesDeployment, tmp_path: Path) -> Path:
    home = tmp_path / "home"
    deployment._install_plugin(home)
    deployment._install_config(home)
    deployment._install_soul(home)
    return home


def _tree_hash(root: Path) -> str:
    digest = hashlib.sha256()
    for path in sorted(path for path in root.rglob("*") if path.is_file()):
        digest.update(path.relative_to(root).as_posix().encode("utf-8"))
        digest.update(path.read_bytes())
    return digest.hexdigest()


def test_detect_reports_runtime_healthy_but_authority_degraded_after_host_drift(
    tmp_path: Path, monkeypatch,
) -> None:
    deployment = HermesDeployment(ROOT)
    host = _host(tmp_path)
    home = _installed_home(deployment, tmp_path)
    monkeypatch.setattr(
        deployment, "_host_controller", lambda: _DegradedAuthorityController()
    )

    state = deployment.detect(host, home)

    assert state["runtime_health"] == "healthy"
    assert state["authority_health"] == "degraded"
    assert state["classification"] == "degraded"
    assert state["host_adaptation"]["classification"] == "incompatible"
    assert state["host_adaptation"]["patch_state"] == "mismatch"
    assert state["installed"] is True


def test_detect_does_not_mutate_host_or_profile(tmp_path: Path, monkeypatch) -> None:
    deployment = HermesDeployment(ROOT)
    host = _host(tmp_path)
    home = _installed_home(deployment, tmp_path)
    monkeypatch.setattr(
        deployment, "_host_controller", lambda: _DegradedAuthorityController()
    )
    before = (_tree_hash(host), _tree_hash(home))

    deployment.detect(host, home)

    assert (_tree_hash(host), _tree_hash(home)) == before
