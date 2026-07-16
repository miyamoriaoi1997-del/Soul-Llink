# Hermes host adapter boundary

SoulLink core stays host-neutral under `packages/`. Hermes integration assets are owned and versioned here.

## Latest Hermes: profile-local deployment (adapter v2)

For current Hermes Agent releases that expose `MemoryProvider`, `ContextEngine`, user-plugin discovery, and `register_context_engine`, use:

```bash
soullink-hermes-deploy detect \
  --soullink-root /path/to/Soul-Llink \
  --host-root /path/to/hermes-agent \
  --hermes-home "$HERMES_HOME"

soullink-hermes-deploy apply \
  --soullink-root /path/to/Soul-Llink \
  --host-root /path/to/hermes-agent \
  --hermes-home "$HERMES_HOME" \
  --receipt "$HERMES_HOME/soullink-deployment-receipt.json"
```

This strategy does **not** modify Hermes source. It transactionally manages only profile-local paths:

- `plugins/soullink` — exclusive PCLTM memory provider;
- `plugins/pcltm-context` — general plugin registering the governed context engine;
- `config.yaml` — selects both plugins and disables native compression;
- `SOUL.md` — installs the SoulLink identity anchor.

The original bytes of all pre-existing managed paths are retained in a durable backup. Apply/verification failure restores them automatically. Explicit rollback:

```bash
soullink-hermes-deploy rollback \
  --soullink-root /path/to/Soul-Llink \
  --receipt "$HERMES_HOME/soullink-deployment-receipt.json"
```

Hermes must start a fresh process/session after deployment. `verify` launches the host interpreter in a fresh process and proves that Hermes discovers the provider and context engine; it does not hot-reload an already-running agent.

The declarative contract is `compatibility-v2.yaml`.

## Historical core patch (adapter v1)

`patches/pcltm-context-engine-host-adapter.patch` is retained only for older Hermes revisions that predate the required SPIs. Do not force it onto current Hermes. The legacy controller and `compatibility.yaml` remain available for exact historical hosts.
