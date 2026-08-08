# SoulLink Public 2.2.1

SoulLink Public 2.2.1 is a patch release for the public persona-runtime contract.

## Fixed

- The dynamic `emotion_modifier` now emits its compact trajectory and control-boundary anchor even when the emotion state is neutral.
- The public scope constraint is present independently of optional daily-mood output: it does not mutate persistent state, trigger the adult route on its own, or override work/crisis boundaries.
- This resolves the public `packages/persona_engine/tests` CI failure introduced by the prior runtime-forwarding update.

## Verification

The release candidate was verified with the same GitHub Actions sequence:

- `uv run pytest -q` — 1255 passed, 3 skipped
- `PYTHONPATH=".:packages/persona_engine:packages:adapters" uv run pytest -q packages/persona_engine/tests` — 619 passed, 10 skipped
- `python scripts/public_release_audit.py --root .` — PASS
- `uv build` — wheel and source distribution built

## Install

Download these assets from the `v2.2.1` release:

- `soullink_public-2.2.1-py3-none-any.whl`
- `soullink_public-2.2.1.tar.gz`
- `SHA256SUMS.txt`

Verify the checksums before installing, then install the wheel in a clean Python 3.11+ environment:

```bash
python -m pip install soullink_public-2.2.1-py3-none-any.whl
soullink init
soullink doctor
```

## Public boundary

This patch contains public runtime code, tests, documentation, and release artifacts only. It contains no private persona data, user memories, conversations, credentials, runtime databases, logs, telemetry, or deployment state.
