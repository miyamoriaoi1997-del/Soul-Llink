"""Import-boundary tests for Soul-Link/PCLTM host independence.

These tests intentionally scan source text instead of importing modules. The
contract under test is architectural: core packages must not grow direct Hermes
runtime dependencies. Hermes-specific integration belongs under adapters/hermes.
"""

from __future__ import annotations

import ast
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
CORE_PACKAGE_DIRS = (
    ROOT / "packages" / "pcltm",
    ROOT / "packages" / "persona_engine",
    ROOT / "packages" / "soul_link",
)
HERMES_ADAPTER_DIR = ROOT / "adapters" / "hermes"
FORBIDDEN_HERMES_IMPORT_ROOTS = {
    "agent",
    "batch_runner",
    "cli",
    "gateway",
    "hermes_cli",
    "hermes_constants",
    "hermes_logging",
    "hermes_state",
    "model_tools",
    "run_agent",
    "toolsets",
    "tools",
}
ALLOWED_IMPORTS = {
    # Soul-Link's public contract module intentionally exposes host path
    # constants so doctor/smoke tools can report integration status. It must
    # remain side-effect-light and must not import Hermes runtime modules.
    ("packages/soul_link/contracts.py", "path_literals"),
    # PCLTM memory_adapter keeps environment-variable compatibility names for
    # the current Hermes-hosted deployment. This is configuration compatibility,
    # not a Hermes runtime import. Future work should rename these behind
    # host-neutral aliases, but Phase 0 only forbids runtime lock-in.
    ("packages/pcltm/memory_adapter.py", "compat_env_names"),
}


def _python_files(paths: tuple[Path, ...]) -> list[Path]:
    files: list[Path] = []
    for root in paths:
        files.extend(
            path
            for path in root.rglob("*.py")
            if "__pycache__" not in path.parts
        )
    return sorted(files)


def _rel(path: Path) -> str:
    return path.relative_to(ROOT).as_posix()


def _import_roots(path: Path) -> set[str]:
    tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    roots: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                roots.add(alias.name.split(".", 1)[0])
        elif isinstance(node, ast.ImportFrom):
            if node.level:
                continue
            if node.module:
                roots.add(node.module.split(".", 1)[0])
    return roots


def test_core_packages_do_not_import_hermes_runtime_modules():
    offenders: list[str] = []
    for path in _python_files(CORE_PACKAGE_DIRS):
        forbidden = sorted(_import_roots(path) & FORBIDDEN_HERMES_IMPORT_ROOTS)
        if forbidden:
            offenders.append(f"{_rel(path)} imports {', '.join(forbidden)}")

    assert offenders == []


def test_hermes_adapter_is_the_only_tree_allowed_to_import_hermes_runtime_modules():
    adapter_files = _python_files((HERMES_ADAPTER_DIR,))
    assert adapter_files, "expected Hermes adapter files to exist"

    adapter_imports = {
        root
        for path in adapter_files
        for root in _import_roots(path)
        if root in FORBIDDEN_HERMES_IMPORT_ROOTS
    }
    assert adapter_imports, "Hermes adapter should be the integration boundary for Hermes runtime imports"


def test_core_packages_do_not_depend_on_hermes_source_paths_or_home_paths():
    offenders: list[str] = []
    forbidden_literals = (
        str(Path.home() / ".hermes" / "hermes-agent"),
        "~/.hermes/hermes-agent",
        "from run_agent",
        "import run_agent",
        "from gateway",
        "import gateway",
        "from hermes_state",
        "import hermes_state",
        "from tools.memory_tool",
        "import tools.memory_tool",
        "from agent.system_prompt",
        "import agent.system_prompt",
    )
    for path in _python_files(CORE_PACKAGE_DIRS):
        text = path.read_text(encoding="utf-8")
        rel = _rel(path)
        if (rel, "path_literals") in ALLOWED_IMPORTS:
            text = text.replace(str(Path.home() / ".hermes" / "hermes-agent"), '')
        hits = [literal for literal in forbidden_literals if literal in text]
        if hits:
            offenders.append(f"{rel} contains {', '.join(hits)}")

    assert offenders == []


def test_core_public_facade_does_not_expose_internal_orchestrator_passthrough():
    import soul_link

    assert "SoulLink" in soul_link.__all__
    assert "StateOrchestrator" not in soul_link.__all__
    assert "EmotionStateManager" not in soul_link.__all__
    assert "orchestrator" not in dir(soul_link)
