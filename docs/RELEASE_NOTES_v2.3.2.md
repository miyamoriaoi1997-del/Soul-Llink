# SoulLink Public 2.3.2

SoulLink Public 2.3.2 introduces the public dynamic emotion-injection runtime. It turns conversational signals into a persistent, bounded emotional state and carries that state into persona expression without replacing the persona's identity, safety boundaries, factual discipline, permissions, or tool rules.

## What changed

### Dynamic emotion injection

The public persona engine now provides a complete emotion path for each conversation turn:

1. **Detects conversational signals** from the user message, including recognition, care, sharing, apology, intimacy, teasing, interruption, and other supported interaction patterns. The detector combines the public rules/lexicon path with the optional sentiment analyzer; when the optional neural component is unavailable or uncertain, the deterministic path remains usable.
2. **Updates a four-dimensional relationship state** consisting of `affection`, `trust`, `possessiveness`, and `patience`. Event intensity produces bounded deltas instead of replacing the previous state.
3. **Preserves continuity between turns** through smoothing and inertia. Trust also affects how patience changes, so the same event does not produce an identical reaction at every state.
4. **Applies time decay** toward the configured baselines when the persona is idle. Larger deviations and ordinary residual emotion are handled by the calculator's non-linear recovery model rather than resetting the state at the next turn.
5. **Derives an expression modifier** from the resulting state. The modifier describes the current emotional direction, intensity, secondary feeling, aftereffect, and expression budget for the persona layer. It changes tone, distance, initiative, and boundary firmness; it does not rewrite the core identity or override task, safety, or permission constraints.
6. **Carries strong emotional continuation explicitly** through `emotion-state.json`. The Stop hook consumes the persisted continuation decision rather than recomputing a different emotional result at the end of the turn.
7. **Records prompt-safe emotion evidence** through the governed PCLTM evidence-capsule path, keeping the current state and tone traceable without treating emotion as an independent memory authority.

The default state is a neutral four-dimensional baseline. A neutral state still produces a bounded stable modifier, so the injection surface has a predictable contract instead of disappearing unpredictably between sessions.

### Packaging and public boundary

- Added the dynamic injection lexicon as a package asset so installed builds and source-tree runs use the same public resources.
- Updated the public runtime packaging metadata and lockfile to version `2.3.2`.
- Kept the public host-neutral runtime boundary intact. Private production adapters, persona layers, memories, databases, logs, deployment backups, credentials, and host-local deployment material are excluded.
- Emotion-layer failures remain fail-open: an unavailable emotion component degrades to a neutral tone and does not block or interrupt the host session.

## Compatibility and installation

This release supports Python 3.11 and newer. Core runtime use does not require Hermes or Codex; those integrations remain optional adapters.

Install the wheel from the GitHub Release after verifying its checksum:

```bash
python -m pip install soullink_public-2.3.2-py3-none-any.whl
soullink init
soullink doctor
soullink webui
```

## Release assets

- `soullink_public-2.3.2-py3-none-any.whl`
- `soullink_public-2.3.2.tar.gz`
- `SHA256SUMS.txt`

## Verification performed

The release candidate was verified with real commands:

- Full public test suite: `1400 passed, 2 skipped, 2 warnings`.
- Python compilation check: passed.
- Source distribution and wheel build: passed.
- Wheel installation into a fresh Python 3.11 virtual environment: passed.
- Fresh-process runtime smoke test (`EmotionBridge` and stable emotion modifier): passed.
- `git diff --check`: passed after release whitespace normalization.
- Public-path scan: no production checkout paths, private production repository identifiers, credentials, databases, or runtime logs were included in the tracked release change.

The two warnings are existing SWIG deprecation warnings emitted by the memory-provider test path; they are not test failures.

## Upgrade note

If upgrading from an earlier 2.3 release, rebuild or reinstall the package so the `injection_lexicon.yaml` package asset is present in the installed distribution. Do not copy production runtime state into the public checkout.
