# Security Policy

## Supported release

Security fixes are applied to the latest public release candidate and the latest published release.

## Reporting a vulnerability

Do not open a public issue for credentials, private memories, host takeover paths, or data exposure. Use the repository's private security-advisory channel. Include affected version, reproduction steps, impact, and the smallest safe evidence needed to verify the report.

Never include real API keys, private persona overlays, production databases, memory exports, logs, or user conversations in a report. Replace them with synthetic fixtures.

## Trust boundary

The reusable `soul_link`, `pcltm`, `persona_engine`, and `model_router` packages are host-neutral. Optional adapters under `adapters/` and versioned host patchsets are explicit integration surfaces. Installation into a host must be version-checked, backed up, verifiable, and reversible.

Run `python scripts/public_release_audit.py --root .` before publishing an archive.
