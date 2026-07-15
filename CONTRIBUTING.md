# Contributing

SoulLink accepts changes that preserve explicit identity, memory, context, and host-adapter boundaries.

## Development

```bash
uv sync --group dev
uv run pytest -q
PYTHONPATH="packages/persona_engine:packages:adapters" uv run pytest -q packages/persona_engine/tests
uv run python scripts/public_release_audit.py --root .
```

On Windows Git Bash, use `;` rather than `:` inside `PYTHONPATH`.

## Public-data rules

- Use synthetic personas, users, scopes, paths, messages, and secrets.
- Do not commit runtime databases, MemFS state, logs, backups, `.env` files, credentials, private overlays, or real conversations.
- Core packages remain host-neutral. Host integration belongs in an explicit adapter or versioned patchset with capability detection, verification, and rollback.
- Add or update tests for behavior-bearing changes.

## Pull requests

Explain the trust boundary affected, tests run, rollback impact, and any compatibility constraint. A passing unit suite does not replace clean-wheel installation and public-release audit evidence.
