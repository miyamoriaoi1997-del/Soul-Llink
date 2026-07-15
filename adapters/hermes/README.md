# Hermes host adapter boundary

This directory contains SoulLink-owned integration material for Hermes Agent hosts.

SoulLink/PCLTM core packages stay host-neutral under `packages/`. Runtime-specific Hermes seams, patchsets, and host-adapter notes belong here so they are versioned with SoulLink rather than mistaken for generic Hermes upstream work.

## Patchsets

- `patches/pcltm-context-engine-host-adapter.patch` — host-side Hermes changes required for the SoulLink/PCLTM context engine integration:
  - configure plugin context engines from Hermes context config;
  - pass request-budget information into context compression;
  - account for tool-output pressure in compaction decisions;
  - adapt the `plugins/context_engine/pcltm-context` host plugin;
  - add Hermes-side regression tests for the adapter seam.
- `soullink-plugin.yaml` — expected Hermes manifest for the production `soullink` memory-provider plugin. It uses `kind: exclusive`, the host-recognized kind for provider plugins selected through `memory.provider`, and avoids the general plugin scanner's unknown-kind fallback.

These patchsets are source artifacts. Apply them only to an explicitly selected Hermes host checkout, then verify with the relevant Hermes-side tests. The host context-engine file currently carries mixed historical line endings; validate the managed patch against an already-applied Windows checkout with:

```bash
git apply --check --reverse --ignore-space-change --ignore-whitespace adapters/hermes/patches/pcltm-context-engine-host-adapter.patch
```
