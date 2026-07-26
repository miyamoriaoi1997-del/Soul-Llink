# SoulLink Public 2.1

SoulLink Public 2.1 strengthens PCLTM recall and context continuity while retiring the active DAC execution path.

## Highlights

- Retires active DAC assembly, recall, storage, doctor, and CLI paths while preserving read-only compatibility for legacy database evidence.
- Adds governed tiered retrieval with explicit provider and policy boundaries.
- Improves recall precision for memory-system diagnostics and prevents unrelated records from entering diagnostic recall.
- Carries typed continuity evidence and session identity through the Hermes memory-provider boundary.
- Tightens active-frame and recall-intent observability without changing the read-only WebUI boundary.
- Improves WebUI timing presentation so captured timing and unavailable evidence remain distinguishable.

## Install

1. Download the wheel and `SHA256SUMS.txt` from this release.
2. Verify the checksum.
3. Install into a clean Python 3.11+ environment:

```bash
python -m pip install soullink_public-2.1.0-py3-none-any.whl
soullink init
soullink doctor
soullink webui
```

The WebUI binds to `127.0.0.1` and remains read-only by default.

## Migration notes

- The distribution name is now `soullink-public`; console commands remain unchanged.
- Active imports from `pcltm.dac` must be removed. Existing DAC tables may remain in older databases for read-only migration evidence, but SoulLink no longer creates or writes them.
- Retrieval-provider APIs are additive. Existing PCLTM databases are upgraded through the normal schema bootstrap path.

## Public boundary

The release contains reusable runtime code, neutral public templates, tests, and explicit host adapters. It contains no private personas, user memories, conversations, credentials, runtime databases, logs, or deployment-specific state.
