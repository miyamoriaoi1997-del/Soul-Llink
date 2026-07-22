# SoulLink Public 2.0

SoulLink Public 2.0 is the first packaged public release of the long-running persona runtime.

## Highlights

- Layered persona composition with stable identity and mode-specific behavior.
- Continuous four-axis emotional state with intensity, momentum, and aftereffects.
- PCLTM governed memory: archive → candidates → policy judgment → final influence.
- Read-only SoulLink Observatory that distinguishes exact host capture from reconstruction previews.
- Explicit Hermes and Codex adapter lifecycles with detection, verification, receipts, and rollback.
- OpenAI-compatible model router with auditable request metadata boundaries.

## Install

1. Download the wheel and `SHA256SUMS.txt` from this release.
2. Verify the wheel checksum.
3. Install into a clean Python 3.11+ environment:

```bash
python -m pip install soullink_public_2_0-2.0.0-py3-none-any.whl
soullink init
soullink doctor
soullink webui
```

The WebUI binds to `127.0.0.1` and is read-only by default.

## Public boundary

This release contains the reusable runtime, neutral public persona templates, tests, and explicit host adapters. It does not contain private personas, user memories, conversations, credentials, runtime databases, logs, or deployment-specific state.

## Verification

Release assets are built only after the public test suites, release audit, package build, clean-wheel installation, CLI smoke checks, and archive-member inspection complete successfully. See `RELEASE_CHECKLIST.md` for the gate.
