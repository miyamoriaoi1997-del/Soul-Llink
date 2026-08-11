# SoulLink Public 2.2.3

SoulLink Public 2.2.3 is a governed-memory projection and error-visibility patch release.

## Fixed

- Multi-source governed claims now project every retained authority reference into the SQLite FTS projection.
- Multi-source governed claims now preserve the complete authority-reference set in the MemFS projection.
- Projection verification checks all retained `authority_refs` instead of accepting only the first source.
- MemFS system-layer projection failures are surfaced instead of being silently ignored.
- Retrieval statistics and citation lookups expose SQLite failures while still cleaning up their connections.

## Verification

The release includes focused regression coverage for FTS projection, MemFS projection, authority-reference preservation, and adapter error visibility. The final release tree is also checked with the complete public test suite, strict public release/privacy audit, distribution-member inspection, checksum verification, and clean-environment wheel installation and CLI smoke tests.

## Safety and privacy

This patch changes only portable governed-memory projection and error-handling behavior plus its regression tests and release metadata. It contains no private persona data, user memories, conversations, credentials, runtime databases, logs, telemetry, or deployment state.

## Install

Download these assets from the `v2.2.3` release:

- `soullink_public-2.2.3-py3-none-any.whl`
- `soullink_public-2.2.3.tar.gz`
- `SHA256SUMS.txt`

Verify the checksums before installing, then install the wheel in a clean Python 3.11+ environment:

```bash
python -m pip install soullink_public-2.2.3-py3-none-any.whl
soullink init
soullink doctor
```
