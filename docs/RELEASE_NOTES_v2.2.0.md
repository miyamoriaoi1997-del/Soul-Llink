# SoulLink Public 2.2

SoulLink Public 2.2 synchronizes the reusable public runtime with the latest verified production code while preserving the public privacy and host-integration boundaries.

## Highlights

- Consolidates governed memory authority around typed claims, policy decisions, evidence-led retrieval, and controlled injection.
- Adds automatic conversation-to-candidate processing with governance-aware promotion and projection updates.
- Adds governed SQLite, FTS, MemFS, and semantic retrieval surfaces with explicit authority and provenance boundaries.
- Improves exact recall, context capsules, continuity handling, projection recovery, and legacy-memory migration safety.
- Adds adaptive semantic retrieval and semantic-fusion observability without allowing optional neural signals to authorize facts or injection.
- Adds transactional Codex integration through STDIO MCP and lifecycle hooks with receipts and byte-exact rollback.
- Adds lossless Hermes update transactions with authenticated recovery points, compatibility checks, and rollback evidence.
- Refreshes the read-only observability WebUI and its synthetic public contract tests.

## Install

1. Download the wheel and `SHA256SUMS.txt` from this release.
2. Verify the checksum.
3. Install into a clean Python 3.11+ environment:

```bash
python -m pip install soullink_public-2.2.0-py3-none-any.whl
soullink init
soullink doctor
soullink webui
```

The WebUI binds to `127.0.0.1` and remains read-only by default.

## Migration notes

- Existing databases are upgraded through the normal schema bootstrap path. Back up deployment-owned runtime state before upgrading.
- Legacy memory tables and MemFS content are migration/evidence surfaces; governed claims and projections are the active authority.
- New `soullink-hermes-update` and Codex commands are optional host-integration surfaces. Run detect/verify in an isolated fixture before any production mutation.
- Optional lexical, vector, or neural retrieval remains candidate-only and cannot independently authorize durable memory or prompt injection.

## Public boundary

The release contains reusable runtime code, neutral public templates, tests, and explicit host adapters. It contains no private personas, user memories, conversations, credentials, runtime databases, logs, private production audit reports, or deployment-specific state.
