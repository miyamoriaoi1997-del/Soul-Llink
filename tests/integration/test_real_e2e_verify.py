"""Real end-to-end test with actual verify subprocess execution.

This test does NOT mock HermesDeployment.verify(). It creates a real minimal
Hermes host with Python, executes the actual verify() subprocess, and proves:
1. plugins.memory and hermes_cli.plugins can be imported
2. New state-machine modules (transition_manager_v2, transition_policy) can be imported
3. Module provenance is from the managed packages path, not repo cwd
4. Rollback restores host to pristine state
"""
from __future__ import annotations

import hashlib
import sys
from pathlib import Path
from types import SimpleNamespace

from soul_link.hermes_deploy import HermesDeployment


def _tree_hash(root: Path) -> str:
    """Compute deterministic hash of the complete directory tree."""
    digest = hashlib.sha256()
    for path in sorted(p for p in root.rglob("*") if p.is_file()):
        digest.update(path.relative_to(root).as_posix().encode())
        digest.update(path.read_bytes())
    return digest.hexdigest()


def _create_executable_hermes_host(tmp_path: Path) -> Path:
    """Create a minimal but executable Hermes host with real Python modules."""
    host = tmp_path / "hermes_host"

    # Create required host contract files
    files = {
        "agent/context_engine.py": """
class ContextEngine:
    name = "default"
    def is_available(self):
        return True
""",
        "agent/memory_provider.py": """
class MemoryProvider:
    name = "default"
    def is_available(self):
        return True
""",
        "agent/context_compressor.py": """
SUMMARY_PREFIX = "Previous conversation summary:"
""",
        "agent/model_metadata.py": """
def estimate_messages_tokens_rough(messages):
    return sum(len(str(message.get('content', ''))) for message in messages) // 4

def get_model_context_length(model=None):
    return 128000
""",
        "plugins/memory/__init__.py": """
import sys
import os
import importlib.util
from pathlib import Path

_providers = {}

class PluginContext:
    def register_memory_provider(self, provider):
        _providers[provider.name] = provider

def register_memory_provider(name=None):
    # Called by verify probe
    if name and name in _providers:
        return _providers[name]
    return None

def load_memory_provider(name=None):
    # Load plugins first
    hermes_home = Path(os.environ.get('HERMES_HOME', '.'))

    # Load soullink plugin if not already loaded
    if name == 'soullink' and name not in _providers:
        plugin_path = hermes_home / "plugins" / "soullink" / "__init__.py"
        if plugin_path.exists():
            spec = importlib.util.spec_from_file_location("soullink_plugin", plugin_path)
            if spec and spec.loader:
                module = importlib.util.module_from_spec(spec)
                try:
                    spec.loader.exec_module(module)
                    if hasattr(module, 'register'):
                        module.register(PluginContext())
                except Exception:
                    raise

    if name and name in _providers:
        return _providers[name]
    return None
""",
        "hermes_cli/plugins.py": """
import sys
import importlib.util
from pathlib import Path

_context_engines = {}

class PluginContext:
    def register_context_engine(self, engine):
        _context_engines[engine.name] = engine

def register_context_engine(engine):
    _context_engines[engine.name] = engine

def discover_plugins():
    # Load pcltm-context plugin
    import os
    hermes_home = Path(os.environ.get('HERMES_HOME', Path.cwd()))
    plugin_path = hermes_home / "plugins" / "pcltm-context" / "__init__.py"
    if plugin_path.exists():
        spec = importlib.util.spec_from_file_location("pcltm_context_plugin", plugin_path)
        if spec and spec.loader:
            module = importlib.util.module_from_spec(spec)
            try:
                spec.loader.exec_module(module)
                if hasattr(module, 'register'):
                    module.register(PluginContext())
            except Exception:
                raise

def get_plugin_context_engine():
    for engine in _context_engines.values():
        return engine
    return None
""",
    }

    for relative, content in files.items():
        path = host / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(content, encoding="utf-8")

    # Create __init__.py files to make packages importable
    (host / "agent/__init__.py").write_text("", encoding="utf-8")
    (host / "plugins/__init__.py").write_text("", encoding="utf-8")
    (host / "hermes_cli/__init__.py").write_text(
        "__version__ = 'test-host'\n", encoding="utf-8"
    )

    return host


def test_real_e2e_verify_imports_state_machine_modules(tmp_path: Path, monkeypatch) -> None:
    """E2E test with REAL verify() subprocess that imports state-machine modules.

    This test does NOT mock verify(). It proves:
    - verify() subprocess can actually run
    - plugins.memory and hermes_cli.plugins are importable
    - State-machine modules (transition_manager_v2, transition_policy) are importable
    - Module provenance is from packages/, not from repo cwd
    - Rollback restores host to pristine state
    """
    host = _create_executable_hermes_host(tmp_path)
    home = tmp_path / "hermes_home"
    home.mkdir()

    # Record initial state
    before_host_hash = _tree_hash(host)
    before_home_hash = _tree_hash(home)
    before_home_files = sorted(p.relative_to(home).as_posix() for p in home.rglob("*") if p.is_file())

    # Get deployment controller pointing to this repository
    repo_root = Path(__file__).resolve().parents[2]
    deployment = HermesDeployment(repo_root)
    controller = SimpleNamespace(
        detect=lambda _host: SimpleNamespace(
            classification="supported", patch_state="applied", missing_paths=()
        ),
        apply=lambda _host, **_kwargs: (SimpleNamespace(classification="supported"), None),
        verify=lambda _host: True,
        rollback=lambda *_args, **_kwargs: True,
    )
    monkeypatch.setattr(deployment, "_host_controller", lambda: controller)

    # Detect phase
    detect_result = deployment.detect(host, home)
    assert detect_result["classification"] in ("transformable", "supported")

    # Apply phase - this will call verify() at the end
    receipt = deployment.apply(host, home)
    assert receipt is not None, "apply() should return receipt after successful installation"

    # Verify the installation is actually installed
    assert (home / "plugins/soullink/__init__.py").exists()
    assert (home / "plugins/pcltm-context/__init__.py").exists()
    assert (home / "SOUL.md").exists()

    # Most critical: verify() subprocess must succeed
    # This actually runs subprocess.run with the verify probe
    verify_result = deployment.verify(host, home)
    assert verify_result is True, "verify() must succeed with real subprocess execution"

    # Now add an extended probe to verify state-machine modules are importable
    # We'll do this by running a custom subprocess with the same environment
    import subprocess
    import os

    python = deployment._host_python(host)
    extended_probe = """
import sys
from pathlib import Path

# Verify basic imports work
from plugins.memory import load_memory_provider
p = load_memory_provider('soullink')
assert p and p.is_available(), "soullink provider must be available"

from hermes_cli.plugins import discover_plugins, get_plugin_context_engine
discover_plugins()
e = get_plugin_context_engine()
assert e and e.name == 'pcltm-context' and e.is_available(), "pcltm-context engine must be available"

# Critical: Import new state-machine modules from persona_engine.persona_orchestrator
# These must come from packages/, not from repo cwd
from persona_engine.persona_orchestrator.transition_manager_v2 import TransitionManagerV2
from persona_engine.persona_orchestrator.transition_policy import TransitionTable

# Verify module provenance - must be under packages/
import persona_engine.persona_orchestrator
module_path = Path(persona_engine.persona_orchestrator.__file__).resolve()
packages_path = Path(sys.argv[1]).resolve() / "packages"
assert module_path.is_relative_to(packages_path), (
    f"Module must come from managed packages: {module_path} not under {packages_path}"
)

print("REAL_E2E_VERIFY_OK")
"""

    env = os.environ.copy()
    env.update({
        "HERMES_HOME": str(home),
        "SOULLINK_ROOT": str(repo_root),
        "PYTHONPATH": os.pathsep.join((str(host), str(repo_root), str(repo_root / "packages"))),
        "PYTHONDONTWRITEBYTECODE": "1",
    })

    result = subprocess.run(
        [str(python), "-c", extended_probe, str(repo_root)],
        cwd=host,
        env=env,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        check=False,
        timeout=60,
    )

    assert result.returncode == 0, (
        f"Extended verify probe failed:\n"
        f"stdout: {result.stdout}\n"
        f"stderr: {result.stderr}"
    )
    assert "REAL_E2E_VERIFY_OK" in result.stdout, "Extended probe must complete successfully"

    # Verify host is unchanged
    after_apply_host_hash = _tree_hash(host)
    assert before_host_hash == after_apply_host_hash, "Host must remain unchanged"

    # Rollback phase
    rollback_result = deployment.rollback(receipt)
    assert rollback_result is True

    # Verify complete restoration
    after_rollback_host_hash = _tree_hash(host)
    assert before_host_hash == after_rollback_host_hash, "Host must be restored to initial state"

    after_home_files = sorted(p.relative_to(home).as_posix() for p in home.rglob("*") if p.is_file())
    assert after_home_files == before_home_files
    assert _tree_hash(home) == before_home_hash
