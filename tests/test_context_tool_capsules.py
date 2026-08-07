from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import pytest


def _load_engine_module():
    path = Path(__file__).resolve().parents[1] / "soul_link/hermes_plugin/context_engine.py"
    name = "pcltm_context_engine_tool_capsule_test"
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    import sys
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def test_terminal_capsule_redacts_secrets_and_preserves_final_result() -> None:
    module = _load_engine_module()
    output = (
        "BEGIN\n"
        + "authorization: Bearer sk-test-secret-value-123456789\n"
        + ("progress line\n" * 300)
        + "RESULT: 37 passed\n"
    )
    raw = {
        "output": output,
        "exit_code": 0,
        "error": None,
        "authorization": "Bearer sk-test-secret-value-123456789",
    }
    text = json.dumps(raw)

    capsule = module._render_tool_capsule(
        tool_name="terminal",
        tool_call_id="call-test",
        text=text,
        raw_content=raw,
        kind="terminal",
        preserve_edges=True,
    )

    assert "sk-test-secret-value-123456789" not in capsule
    assert "[REDACTED" in capsule or "authorization: ***" in capsule
    assert "RESULT: 37 passed" in capsule
    assert '"exit_code": 0' in capsule


@pytest.mark.parametrize(
    "key",
    [
        "token", "access_token", "api_key", "client_secret",
        "private_key", "password", "passwd", "authorization",
    ],
)
def test_fallback_redactor_fail_closed_for_json_secret_values(monkeypatch, key: str) -> None:
    module = _load_engine_module()
    monkeypatch.setattr(
        module,
        "import_pcltm_module",
        lambda _name: (_ for _ in ()).throw(ModuleNotFoundError("injected")),
    )
    secret = 'opaque-secret\\"-value\nsecond-line-123456789'
    text = json.dumps({"nested": {key: secret}, "status": "ok"})

    redacted = module._redact_tool_evidence(text)
    parsed = json.loads(redacted)

    assert secret not in redacted
    assert "[REDACTED]" in redacted
    assert parsed["nested"][key] == "[REDACTED]"
    assert parsed["status"] == "ok"


def test_fallback_redactor_masks_value_patterns_inside_json(monkeypatch) -> None:
    module = _load_engine_module()
    monkeypatch.setattr(
        module,
        "import_pcltm_module",
        lambda _name: (_ for _ in ()).throw(ModuleNotFoundError("injected")),
    )
    bearer = "Bearer abcdefghijklmnopqrstuvwxyz123456"
    text = json.dumps({"message": bearer, "token": "opaque-secret-value-123456789"})

    redacted = module._redact_tool_evidence(text)

    assert bearer not in redacted
    assert "opaque-secret-value-123456789" not in redacted


@pytest.mark.parametrize(
    "secret_value",
    [
        "api_key=opaque-secret-value-123456789",
        "postgres://alice:opaque-password-123456789@db.example.test/app",
        "xox" + "b-123456789012-abcdefghijklmnop",
        "github_pat_ABCDEFGHIJKLMNOPQRSTUVWXYZ1234567890",
        "-----BEGIN PRIVATE KEY-----\nopaque-private-material\n-----END PRIVATE KEY-----",
    ],
)
def test_fallback_redactor_masks_canonical_value_patterns_inside_json(
    monkeypatch, secret_value: str,
) -> None:
    module = _load_engine_module()
    monkeypatch.setattr(
        module,
        "import_pcltm_module",
        lambda _name: (_ for _ in ()).throw(ModuleNotFoundError("injected")),
    )
    text = json.dumps({"message": secret_value, "status": "ok"})

    redacted = module._redact_tool_evidence(text)
    parsed = json.loads(redacted)

    assert secret_value not in redacted
    assert parsed["status"] == "ok"


@pytest.mark.parametrize(
    "secret_value",
    [
        "api_key=opaque-secret-value-123456789",
        "postgres://alice:opaque-password-123456789@db.example.test/app",
        "xox" + "b-123456789012-abcdefghijklmnop",
        "github_pat_ABCDEFGHIJKLMNOPQRSTUVWXYZ1234567890",
        "-----BEGIN PRIVATE KEY-----\nopaque-private-material\n-----END PRIVATE KEY-----",
        "Bearer abcdefghijklmnopqrstuvwxyz123456",
    ],
)
def test_fallback_redactor_uses_same_patterns_for_non_json_text(
    monkeypatch, secret_value: str,
) -> None:
    module = _load_engine_module()
    monkeypatch.setattr(
        module,
        "import_pcltm_module",
        lambda _name: (_ for _ in ()).throw(ModuleNotFoundError("injected")),
    )

    redacted = module._redact_tool_evidence(f"prefix {secret_value} suffix")

    assert secret_value not in redacted