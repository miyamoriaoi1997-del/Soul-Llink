# PCLTM Storage and Scope Contract

This document records the storage and scope rules absorbed from the EverOS review
without adopting EverOS as a runtime dependency.

## Authority model

PCLTM has more than one storage surface.  Each surface must declare whether it is
source of truth or derived state.

| Surface | Role | Source of truth? | Rebuildable? |
| --- | --- | --- | --- |
| `events` / `summary_nodes` in SQLite | Raw evidence and compressed evidence graph | Yes for captured runtime evidence | No, unless the upstream transcript still exists |
| `memory_claims` / `memory_current` / `memory_governance_events` in SQLite | Versioned governed claims, current lifecycle tuples, and decision receipts | Yes for durable governed memory | No; preserve with SQLite online backup |
| `memory_records` in SQLite | Legacy evidence and migration inventory | No runtime authority | Not applicable; classify/shadow only, never fallback |
| MemFS (`system/`, `pinned/`, `episodic/`, `transient/`, `skills/`) | Human-readable projection and bounded transient/tool evidence | No for governed durable claims | Yes; rebuild governed files from SQLite authority |
| SQLite FTS tables (`event_fts`, `summary_fts`, governed-memory FTS) | Search acceleration | No; derived index | Yes, from canonical SQLite source tables |
| Optional lexical/vector/neural indexes | Candidate retrieval experiments | No; derived and subordinate | Yes; never promotion or injection authority |

Derived state must never become the only copy of a durable memory.  A failed or
stale index is an observability problem, not permission to invent memory content.

## Scope model

PCLTM uses a stable `scope_key` to prevent memory pollution across deployments,
projects, personas, users, and modes.

Recommended dimensions:

```text
profile_id / app_id / project_id / persona_id / user_id / mode_scope
```

Rendered key format:

```text
profile:<profile_id>/app:<app_id>/project:<project_id>/persona:<persona_id>/user:<user_id>/modes:<mode+mode>
```

Example:

```text
profile:example/app:example-agent/project:example-project/persona:example-persona/user:example-user/modes:daily+work
```

Rules:

1. Empty dimensions are omitted; an entirely empty scope renders as `global:default`.
2. Values are lower-case ASCII slugs so they are safe for SQLite metadata,
   MemFS frontmatter, future path components, and derived index IDs.
3. Mode lists are deduplicated and sorted for deterministic keys.
4. `scope_key` belongs in metadata/frontmatter whenever a memory is project,
   persona, runtime, or mode specific.
5. `canonical_key` should include `scope_key` for records that could collide
   across projects or personas.

Recommended canonical form:

```text
<scope_key>/<object_scope>/<stable_slug_or_hash>
```

## Runtime boundaries

- SoulLink-managed SOUL identity remains higher authority than governed memory.
- Mode layers (`daily`, `work`, `sex`, `cron`) narrow retrieval; they do not
  redefine identity.
- Compression and handoff blocks are reference-only and must not become active
  user instructions.
- Reflection and defrag outputs begin as candidate/draft governance artifacts;
  they do not become approved memory without explicit promotion.

## Index observability

PCLTM exposes read-only index observability through:

```text
pcltm index stats
pcltm index doctor
pcltm index doctor --rebuild
```

The commands report:

- SQLite source row counts;
- SQLite FTS derived row counts;
- governed-memory derived-index count;
- MemFS file counts by layer;
- mismatch issues that require rebuild.

`pcltm index doctor --rebuild` rebuilds only the index scope stated by that
command. Full governed projection restore uses the separate restore rehearsal
service: verify backup/config/Git commitments, restore SQLite, rebuild governed
FTS and MemFS, and verify lifecycle/projection receipts. Neither path may edit
durable claim content, SOUL templates, or reviewer decisions.

## EverOS ideas deliberately not adopted

- EverOS runtime is not a production dependency.
- LanceDB is not a default index backend for this private Windows runtime.
- LLM reflection does not directly write approved persona memory.
- SOUL identity templates are not ordinary prompt slots.
