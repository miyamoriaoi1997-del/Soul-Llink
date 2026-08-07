# SoulLink/PCLTM Memory Authority Contract

## Architecture status registry

| Status | Surfaces | Runtime authority |
| --- | --- | --- |
| **canonical** | `events`, `memory_claims`, `memory_current`, `memory_governance_events`, projection outbox, governed search/open/exact/injection | Yes. These are the only durable-memory and model-facing recall authorities. |
| **legacy** | `memory_records`, legacy MemFS materializations, read-only classification/shadow-migration tools | Evidence and migration input only. They may filter stale legacy materializations but may not add prompt content or authorize promotion. |
| **retired** | direct `memory_records → prompt`, layered MemFS prompt selection, legacy materialization/live entries, noncanonical MemFS archival search/open, DB/MemFS fallback, LIKE supersede, transaction-internal file writes, quota-ignore fallback | Disabled or removed. Compatibility seams fail closed and package-root exports are withdrawn. |

The executable registry is `pcltm.memory_architecture_status.ARCHITECTURE_SURFACES`.

SoulLink/PCLTM is the only authority for durable memory, user preferences,
cross-session recall, derived memory, and continuity evidence. Hermes built-in
memory and `session_search` remain host assets for compatibility/visibility, but
are not a provider fallback and must not answer SoulLink-owned recall questions.

## Runtime boundary

The SoulLink memory provider is the exclusive model-facing seam:

- prompt injection: `memory_current → search_governed_memories → build_governed_memory_context`;
- recall tools: `soullink_memory_search`, `soullink_memory_recall_exact`, then
  `soullink_memory_open` when a full body is needed;
- persistence: `soullink_memory_remember → MemoryWriteService → projection outbox`;
- lifecycle: versioned replace/retire/expire transitions with CAS receipts;
- unavailable backend: explicit `status: unavailable`, with fallback forbidden.

`load_prompt_context`, `load_view`, `load_layered_prompt_context`,
`select_context_snapshot`, `load_entries`, legacy MemFS materialization,
`search_archival_memories`, `open_archival_memory`, and
`sync_memory_tool_write` are retired compatibility seams. They return empty or a
typed retired result and never disclose legacy DB/MemFS bodies; the mutation seam
raises `legacy_memory_tool_sync_retired`. They are not durable-memory authorities
or package-root exports. Governed claim search/open are the only full-body recall
surfaces.

Read-only shadow migration is deliberately separate from runtime recall:
`run_readonly_shadow_replay` executes a frozen, versioned legacy lexical replay
and the canonical governed search against the same query-only SQLite snapshot.
It emits only query/content commitments, statuses, and reason codes, then feeds
`compare_shadow_recall`; it never injects, promotes, writes, or falls back. Result
commitments preserve rank order and duplicates. The replay runner computes
`query_sha256` from the query it executes. The comparator requires an exact
`query_bindings` mapping, recomputes every query hash, and rejects missing,
extra, non-string, or mismatched bindings before comparison. A production result
of governed `unavailable` is a blocking migration/readiness signal, not permission
to substitute legacy output.

The `pcltm-context` plugin remains the context-engine seam and the `soullink`
plugin remains the exclusive memory-provider seam. The dedicated
`compatibility-memory-authority.yaml` manifest and generated host patch add the
required schema and execution gates without adding a second provider.

## Backup, restore, and projection boundary

A governed restore bundle contains:

- SQLite online backup with source-before/source-after and snapshot hashes;
- byte-for-byte copies of explicitly selected, non-secret configuration files;
- Git HEAD commitment;
- a bodyless manifest containing hashes, paths, counts, and integrity receipts.

Empty-directory restore verifies bundle hashes and Git HEAD, rejects symlinked
roots/artifacts, restores SQLite and configuration, does **not** restore stale
MemFS, rebuilds FTS/MemFS projections from canonical SQLite authority, runs
`PRAGMA quick_check`, and verifies current claim/projection convergence. Directory
switch is restricted to non-symlink/reparse sibling trees under one parent, keeps a
hash-bound pre-apply tree (including empty-directory structure), and uses
compensating renames for rollback. Snapshot creation requires the source database
to be quiescent for the duration of SQLite online backup; source-file hashes are a
mutation alarm, not a substitute for an application-level writer pause. Switch and
rollback receipts are same-process rehearsal contracts, not authorization for a
delayed or cross-process rollback after path identity may have changed. Rehearsals
must use temporary directories unless production deployment is separately
authorized.

## Experimental boundary

Optional retrieval experiments may provide evidence candidates, but remain
subordinate to SoulLink/PCLTM. They cannot become canonical facts, silently alter
access/lifecycle decisions, force top-k admission, or act as fallback.

## Host-adapter boundary

The SoulLink-owned host transformation is explicit and versioned:

- manifest: `adapters/hermes/compatibility-memory-authority.yaml`;
- patch: `adapters/hermes/patches/memory-authority-host-adapter.patch`;
- host targets: `model_tools.py`, `agent/agent_init.py`,
  `agent/tool_executor.py`, and `agent/agent_runtime_helpers.py`;
- patch-created contract assets: `agent/memory_authority.py`,
  `tests/agent/test_exclusive_memory_authority_contract.py`, and
  `tests/agent/test_soullink_memory_authority_adapter.py`.

When `memory.provider: soullink` is configured, the schema seam removes Hermes
`memory` and `session_search`. Initialization records `active` only after the
provider is available; load failure remains `unavailable`. Sequential and
concurrent agent-loop dispatch plus the alternate runtime helper return a
structured forbidden/unavailable result before touching built-in memory,
provider mirroring, or session recall. Other providers and no-provider setups
remain unchanged.

Applying a manifest, activating a profile, restarting Hermes, migrating legacy
records, or switching a production restore remain deployment operations and
require separate approval.
