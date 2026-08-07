# SoulLink Public 2.2.0

SoulLink Public 2.2.0 synchronizes the reusable public runtime with the verified production architecture while preserving the public privacy, memory-authority, evidence, and reversible-host-integration boundaries.

![SoulLink Observatory 2.2 public demo](assets/soullink-observatory-demo.png)

*The image uses the production-served 2.2 frontend with explicit synthetic browser-side demo values. It contains no production turn, timestamp, emotion, relationship, memory, token, conversation, or host data.*

## Release highlights

### Governed memory authority

- Durable memory authority is carried by governed claims and their provenance/evidence.
- Legacy tables, MemFS files, FTS rows, and optional semantic indexes are compatibility, retrieval, migration, or projection surfaces; they do not become independent sources of truth.
- Memory writes pass through typed candidates, validation, policy judgment, conflict handling, promotion, and projection updates.
- Rejected or ambiguous candidates may remain auditable without becoming injectable durable memory.

### Retrieval and final influence

- Adds governed SQLite/FTS, MemFS, episodic, and optional semantic retrieval surfaces.
- Adaptive retrieval and local semantic fusion improve ranking while staying subordinate to configured authority and deterministic fallback.
- Selection, governance, and final-forward evidence are distinct. Similarity or answer content is never treated as proof that a record reached the model.
- Hosts without an exact final-model-input boundary report `unavailable_host_boundary` rather than promoting a sidecar preview into false exact evidence.

### Candidate processing, migration, and recovery

- Adds conversation-to-candidate processing and governance-aware promotion.
- Adds projection rebuild, restore rehearsal, lineage recovery, and legacy-shadow migration paths.
- Recovery preserves provenance and fails closed when source evidence, ownership, or path safety is ambiguous.
- Optional retrieval projections can be rebuilt without minting new authoritative facts.

### WebUI and observability

- Refreshes the loopback-only, read-only Neural Observatory.
- Separates `exact_host_capture`, `sidecar_reconstruction_preview`, stale observations, and unavailable evidence.
- Connects posture, decision authority, semantic fusion, memory causality, affective state, causal trace, governance selection, context architecture, and fact-base provenance.
- Keeps all collector repair/rebuild operations disabled on the monitoring path.

### Host integration

- Adds transactional Codex integration through STDIO MCP and lifecycle hooks, with receipts and byte-exact rollback.
- Adds `soullink-hermes-update`, a lossless Hermes update transaction with authenticated recovery points, compatibility checks, verification evidence, and rollback.
- Hardens Windows deployment/update/rollback paths against reparse-point and path-redirection escapes outside the selected host root.

## Install and verify

Download these three assets from the release:

- `soullink_public-2.2.0-py3-none-any.whl`
- `soullink_public-2.2.0.tar.gz`
- `SHA256SUMS.txt`

Verify the artifacts before installation:

```bash
sha256sum -c SHA256SUMS.txt
```

Install the wheel in a clean Python 3.11+ environment:

```bash
python -m venv .venv
# Linux/macOS: . .venv/bin/activate
# Windows PowerShell: .venv\Scripts\Activate.ps1
python -m pip install soullink_public-2.2.0-py3-none-any.whl
soullink init
soullink doctor
soullink webui
```

The WebUI binds to `127.0.0.1:8765` and is read-only by default.

## Upgrade and migration notes

1. Back up deployment-owned SQLite, MemFS, configuration, receipts, and host integration state before upgrading.
2. Do not infer production hygiene from `git status`. Inspect ignored caches, disabled-component runtime files, temporary deployment backups, generated reports, and stale logs separately.
3. Existing databases are upgraded through normal schema bootstrap. Legacy memory tables and files remain migration/evidence surfaces; governed claims and projections are the active architecture.
4. Run projection rebuild, restore rehearsal, or migration in an isolated copy/dry-run path when available, retain evidence, and verify the authoritative store plus each derived projection afterward.
5. Treat `soullink-hermes-update`, `soullink-hermes-deploy`, and Codex deployment commands as explicit host mutations. Detect and verify compatibility first, preserve receipts, and rehearse rollback in an isolated host fixture.
6. Optional lexical, vector, or neural retrieval is candidate-only and cannot independently authorize durable memory or prompt injection.
7. A sidecar reconstruction is diagnostic preview evidence, not a substitute for exact host capture.

## Public boundary

The release contains reusable runtime code, neutral templates, tests, and explicit host adapters. It contains no private personas, user memories, conversations, credentials, runtime databases, logs, production telemetry, private audit reports, or deployment-specific state.

## Verification scope

The release workflow validates the public source audit, focused deployment/update/adapter/rollback security tests, WebUI and packaging contracts, wheel/sdist archive contents, clean-environment installation, console entry points, temporary-directory initialization/doctor smoke, checksums, and GitHub Actions. Full command evidence belongs to the release commit and CI logs; the presence of this note alone is not proof that a local checkout passed.
