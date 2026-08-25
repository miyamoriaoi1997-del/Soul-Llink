# SoulLink Public 2.3.3

SoulLink Public 2.3.3 is a patch release that completes the public dynamic emotion-injection contract and restores the remote verification gate for the 2.3 line. It preserves the v2.3.2 runtime behavior while making the generated modifier fully observable and compatible with the public persona-engine test contract.

## What changed

- Added a bounded, continuous `可见度=<n>%` expression-pressure field to the dynamic emotion modifier. The value tracks emotional deviation monotonically within a tier and is an observable expression metric, not a second emotion score.
- Restored explicit public control anchors in every generated modifier: `不改真实STATE`, `不单独触发sex`, and `不覆盖work或crisis边界`.
- Preserved the separation between dynamic expression and persistent state: a modifier can describe tone and event sensitivity without mutating `STATE.md`, independently activating a sexual route, or overriding work/crisis boundaries.
- Kept mood-calendar noise subordinate to expression: it does not write the real state, independently trigger sex, or override task and crisis boundaries.
- Updated package metadata, README installation commands, release assets, and lockfile to `2.3.3`.

## Dynamic emotion injection

The runtime continues to provide the full public emotion path introduced in 2.3.2:

1. Detect conversational signals such as recognition, care, sharing, apology, intimacy, teasing, and interruption through the public rules/lexicon path with optional sentiment analysis.
2. Update the bounded four-dimensional state: `affection`, `trust`, `possessiveness`, and `patience`.
3. Preserve continuity through smoothing, inertia, trust-to-patience coupling, and non-linear time decay toward baseline.
4. Derive an expression modifier carrying direction, intensity, secondary emotion, aftereffect, expression budget, and continuous visibility pressure.
5. Keep strong-emotion continuation explicit through `emotion-state.json` and record prompt-safe evidence through the governed PCLTM evidence-capsule path.
6. Fail open to neutral expression when the optional emotion component is unavailable.

Dynamic emotion changes expression, distance, initiative, and boundary firmness. It does not replace persona identity or override facts, safety, permissions, tool rules, work/crisis boundaries, or persistent state ownership.

## Compatibility and installation

This release supports Python 3.11 and newer. Core runtime use does not require Hermes or Codex; those integrations remain optional adapters.

Install the wheel from the GitHub Release after verifying its checksum:

```bash
python -m pip install soullink_public-2.3.3-py3-none-any.whl
soullink init
soullink doctor
soullink webui
```

## Release assets

- `soullink_public-2.3.3-py3-none-any.whl`
- `soullink_public-2.3.3.tar.gz`
- `SHA256SUMS.txt`

## Verification performed

- Public root suite: `1400 passed, 2 skipped, 2 warnings`.
- Persona-engine CI-equivalent suite after the contract fix: `622 passed, 10 skipped`.
- Python compilation check: passed.
- Wheel and source distribution build: passed.
- Fresh-process runtime smoke test: passed.
- Staged `git diff --check` with Windows CRLF-aware validation: passed.
- Package-asset archive inspection: passed; the dynamic injection lexicon is included.

The two warnings are existing SWIG deprecation warnings emitted by the memory-provider test path; they are not test failures.

## Upgrade note

If upgrading from 2.3.2, reinstall the package so the final modifier contract and `injection_lexicon.yaml` asset are present in the installed distribution. Do not copy production runtime state into the public checkout.
