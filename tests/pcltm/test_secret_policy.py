from __future__ import annotations

import json

import pytest

from pcltm.secret_policy import (
    contains_secret_value,
    evaluate_memory_write,
    extract_safe_connection_metadata,
    redact_secrets,
)


def test_secret_policy_allows_env_var_reference() -> None:
    decision = evaluate_memory_write(
        "OpenAI access is provided at runtime via OPENAI_API_KEY; do not store the key value.",
        target_file="MEMORY.md",
    )

    assert decision.allowed is True
    assert decision.action == "allow"
    assert decision.sensitivity == "normal"
    assert decision.metadata["credential_reference_only"] is True
    assert "OPENAI_API_KEY" in (decision.sanitized_content or "")


def test_secret_policy_rejects_raw_api_key_without_metadata() -> None:
    fake_key = "sk-test_secret_fake_key_1234567890"
    decision = evaluate_memory_write(f"remember {fake_key}", target_file="USER.md")

    assert decision.allowed is False
    assert decision.action == "reject"
    assert decision.sensitivity == "secret"
    assert fake_key not in (decision.sanitized_content or "")
    assert decision.metadata["rejected_raw_secret"] is True


def test_secret_policy_sanitizes_connection_metadata_with_password() -> None:
    content = "SSH server host=203.0.113.10 user=ubuntu password=hunter2 path=/srv/soul-link port=22"
    decision = evaluate_memory_write(content, target_file="MEMORY.md")

    assert decision.allowed is True
    assert decision.action == "sanitize"
    sanitized = decision.sanitized_content or ""
    assert "203.0.113.10" in sanitized
    assert "ubuntu" in sanitized
    assert "/srv/soul-link" in sanitized
    assert "22" in sanitized
    assert "hunter2" not in sanitized
    assert "password=hunter2" not in sanitized
    assert decision.metadata["sanitized_from_secret"] is True


def test_secret_policy_redacts_password_bearing_database_url() -> None:
    url = "postgres://alice:secretpw@db.example.com:5432/app"
    redacted = redact_secrets(f"DATABASE_URL={url}")

    assert "secretpw" not in redacted
    assert "[REDACTED_SECRET]" in redacted


def test_secret_policy_does_not_flag_plain_host_user_path() -> None:
    content = "Server metadata: host=203.0.113.10 user=ubuntu path=/srv/app port=22"

    assert contains_secret_value(content) is False
    assert extract_safe_connection_metadata(content) is not None
    decision = evaluate_memory_write(content, target_file="MEMORY.md")
    assert decision.allowed is True
    assert decision.action == "allow"
    assert decision.sanitized_content == content


def test_redact_secrets_masks_known_patterns() -> None:
    fake_github = "ghp_abcdefghijklmnopqrstuvwxyz123456"
    fake_bearer = "Bearer abcdefghijklmnopqrstuvwxyz123456"
    text = f"token={fake_github} auth={fake_bearer}"

    redacted = redact_secrets(text)

    assert fake_github not in redacted
    assert fake_bearer not in redacted
    assert redacted.count("[REDACTED_SECRET]") >= 2


@pytest.mark.parametrize(
    "key",
    [
        "token", "access_token", "api_key", "client_secret",
        "private_key", "password", "passwd", "authorization",
    ],
)
def test_redact_secrets_masks_structured_json_secret_fields(key: str) -> None:
    secret = 'opaque-secret\\"-value\nsecond-line-123456789'
    text = json.dumps({"nested": {key: secret}, "status": "ok"})

    redacted = redact_secrets(text)
    parsed = json.loads(redacted)

    assert secret not in redacted
    assert parsed["nested"][key] == "[REDACTED_SECRET]"
    assert parsed["status"] == "ok"


def test_secret_policy_detects_structured_json_secret_field() -> None:
    secret = "opaque-secret-value-123456789"
    text = json.dumps({"access_token": secret})

    assert contains_secret_value(text) is True
    decision = evaluate_memory_write(text, target_file="MEMORY.md")
    assert decision.action == "reject"
    assert secret not in (decision.sanitized_content or "")


def test_redact_secrets_still_masks_value_patterns_inside_json() -> None:
    bearer = "Bearer abcdefghijklmnopqrstuvwxyz123456"
    text = json.dumps({"message": bearer, "token": "opaque-secret-value-123456789"})

    redacted = redact_secrets(text)

    assert bearer not in redacted
    assert "opaque-secret-value-123456789" not in redacted
    assert json.loads(redacted)["message"] == "[REDACTED_SECRET]"
