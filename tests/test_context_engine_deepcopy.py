from __future__ import annotations

import copy
import importlib.util
import threading
from pathlib import Path


def _load_engine_module():
    path = Path(__file__).resolve().parents[1] / "soul_link/hermes_plugin/context_engine.py"
    name = "pcltm_context_engine_deepcopy_test"
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    import sys
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def test_deepcopy_produces_independent_engine() -> None:
    module = _load_engine_module()
    engine = module.PCLTMContextCompressionEngine(model="test-model")
    engine.compression_count = 7
    engine._evidence_capsules_written = {"a", "b"}

    cloned = copy.deepcopy(engine)

    # Independent instance, same identity/name.
    assert cloned is not engine
    assert type(cloned) is type(engine)
    assert cloned.name == "pcltm-context"
    # State copied, not shared.
    assert cloned.compression_count == 7
    assert cloned._evidence_capsules_written == {"a", "b"}
    assert cloned._evidence_capsules_written is not engine._evidence_capsules_written
    # Mutating the clone must not leak into the original.
    cloned.compression_count = 99
    cloned._evidence_capsules_written.add("c")
    assert engine.compression_count == 7
    assert engine._evidence_capsules_written == {"a", "b"}


def test_deepcopy_survives_uncopyable_runtime_state() -> None:
    module = _load_engine_module()
    engine = module.PCLTMContextCompressionEngine(model="test-model")
    lock = threading.Lock()
    engine._runtime_lock = lock  # simulate a future uncopyable resource

    cloned = copy.deepcopy(engine)

    # Copy succeeds; the uncopyable resource is shared by reference instead
    # of failing the whole copy (which would silently fall back to the
    # built-in compressor in the host).
    assert cloned is not engine
    assert cloned._runtime_lock is lock
    assert cloned.name == "pcltm-context"


def test_deepcopy_preserves_configured_budget_state() -> None:
    module = _load_engine_module()
    engine = module.PCLTMContextCompressionEngine(model="test-model")
    engine.configure({"budget_tokens": 256_000, "message_budget_tokens": 120_000})

    cloned = copy.deepcopy(engine)

    assert cloned._configured_budget_tokens == engine._configured_budget_tokens
    assert cloned._configured_budget_tokens == 256_000
    assert cloned._request_budget_safety_margin_tokens == engine._request_budget_safety_margin_tokens
