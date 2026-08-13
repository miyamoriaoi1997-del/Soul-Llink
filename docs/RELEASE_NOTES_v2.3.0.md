# SoulLink Public 2.3.0

SoulLink Public 2.3.0 is a feature release for governed continuity, tool-context capsules, and runtime configuration ownership.

## Added

- Added bounded continuity capsules for carrying recent, typed conversation context without exposing implementation details to the persona layer.
- Added tool capsules that render tool results into a bounded, redacted context representation while preserving completion and failure signals.
- Added a runtime configuration store with explicit defaults, environment-aware loading, and atomic persistence behavior.
- Added module-wiring coverage for the Hermes-facing context engine and its capsule boundaries.

## Changed

- Updated the context engine to use the capsule boundaries consistently across context construction and tool-result handling.
- Strengthened persona humanization contracts across core, daily, work, and adult-boundary layers without changing the stable identity anchor.
- Published the new runtime and continuity tests as part of the public regression suite.

## Compatibility and safety

This release preserves the existing public package entry points and Python 3.11+ requirement. The capsule and runtime-configuration APIs are additive; existing callers that do not use them retain their previous behavior.

The release contains no private persona data, user memories, conversations, credentials, runtime databases, logs, telemetry, or deployment state. Hermes host changes are not committed to the Hermes repository; the public package contains only the SoulLink-owned portable runtime and tests.

## Verification

The release candidate was verified with the isolated public environment using the complete pytest suite, the focused capsule/runtime/persona contract suite, the strict public release/privacy audit, Python compilation checks, staged diff checks, clean-environment package checks, and distribution archive inspection.

## Install

Download these assets from the `v2.3.0` release:

- `soullink_public-2.3.0-py3-none-any.whl`
- `soullink_public-2.3.0.tar.gz`
- `SHA256SUMS.txt`

Verify the checksums before installing, then install the wheel in a clean Python 3.11+ environment:

```bash
python -m pip install soullink_public-2.3.0-py3-none-any.whl
soullink init
soullink doctor
```
