# SoulLink/PCLTM Memory Authority Contract

**Architecture status: canonical** — SoulLink/PCLTM is the only authority for
durable memory, user preferences, cross-session recall, derived memory, and
continuity evidence.

**Architecture status: legacy/non-authoritative and retired for this profile**
— Hermes built-in memory and `session_search` remain host assets for
compatibility/visibility, but are not a provider fallback and must not be used
to answer SoulLink-owned recall questions.

## Runtime boundary

The existing SoulLink memory-provider `system_prompt_block()` is the first
model-facing contract for a new session.  The existing SoulLink tools are the
operational seam:

- recall: `soullink_memory_search`, `soullink_memory_recall_exact`, then
  `soullink_memory_open` when a full body is needed;
- persistence: `soullink_memory_remember`;
- unavailable backend: return an explicit `status: unavailable` result with
  `fallback: forbidden`.

The `pcltm-context` plugin remains the context-engine seam and the `soullink`
plugin remains the exclusive memory-provider seam. The dedicated
`compatibility-memory-authority.yaml` manifest and its generated host patch add
the required schema and execution gates without adding a second provider. This
candidate does not change Hermes configuration, touch the live Hermes checkout,
or write a production database/MemFS.

**Architecture status: experimental** — optional retrieval experiments may
provide evidence candidates, but they remain subordinate to the same
SoulLink/PCLTM authority and cannot become canonical facts or a fallback.

## Host-adapter boundary

The SoulLink-owned host transformation is explicit and versioned:

- manifest: `adapters/hermes/compatibility-memory-authority.yaml`;
- patch: `adapters/hermes/patches/memory-authority-host-adapter.patch`;
- host targets: `model_tools.py`, `agent/agent_init.py`,
  `agent/tool_executor.py`, and `agent/agent_runtime_helpers.py`;
- patch-created contract assets: `agent/soullink_memory_authority.py` and
  `tests/agent/test_soullink_memory_authority_adapter.py`.

When `memory.provider: soullink` is configured, the schema seam removes Hermes
`memory` and `session_search`. Agent initialization records `active` only after
the provider is available; load failure remains `unavailable`. Both sequential
and concurrent agent-loop dispatch plus the alternate runtime helper return a
structured forbidden/unavailable result before touching built-in memory,
provider mirroring, or session recall. Other providers and no-provider setups
remain unchanged.

Applying this manifest, activating a profile, and restarting Hermes remain
parent/deployment operations outside this candidate.