# Hermes Host Adapter — SoulLink-owned Host Runtime Transformations

**Ownership**: the SoulLink checkout is the single product repository and sole source of truth for an installation. A Hermes checkout is a SoulLink-managed host runtime target, not a second SoulLink production repository.

Every Hermes-side change required by SoulLink is represented by either:

- the active profile deployment controller `soul_link/hermes_deploy.py` plus its packaged entry assets under `soul_link/hermes_assets/`; or
- a bounded host-source compatibility manifest and patch under this directory.

The copies under `adapters/hermes/plugin/` are documentation fixtures and are byte-checked against the packaged assets. They do not contain a second provider implementation or deployment controller. The sole runtime provider implementation is `soul_link/hermes_plugin/memory_provider.py`.

**Purpose**: Install and verify SoulLink state-machine, PCLTM context, exclusive memory authority, final-forward observation, routing, and host verification integration into the Hermes runtime.

The supported production entrypoint is `python -m soul_link.hermes_deploy`. Its
`apply` operation owns both phases as one transaction: it applies the canonical
SoulLink host-source adapter when required, then installs the profile plugins,
configuration, and identity assets. Its receipt binds both rollback scopes.
Individual `soul_link.host_adaptation` commands remain low-level development and
diagnostic tools; they are not a second product installer.

**Status**: PRIVATE PRODUCTION HOST ADAPTERS

## Architecture

This adapter transforms a Hermes host to consume Soul-Llink's TransitionManagerV2 as the sole mode authority:

```
Hermes user input
  → Persona Orchestrator (with TransitionManagerV2)
  → ModeDecision nomination
  → TransitionManagerV2.transition()
  → TransitionDecision.active_mode (sole authority)
  → Soul layer selection
  → Model routing
  → Actual forwarded runtime/model call
```

## Components

- `soul_link/hermes_update.py` - lossless host update preflight, recovery point, transaction, and restore controller
- `soul_link/hermes_deploy.py` - transactional profile plugin/config/SOUL deployment
- `soul_link/host_adaptation.py` - transactional Hermes host-source patch controller
- `compatibility-*.yaml` - bounded host-source compatibility manifests
- `patches/*.patch` - SoulLink-owned Hermes host transformations
- `plugin/` - documented plugin entry assets, byte-checked against packaged assets
- `tests/integration/test_hermes_deploy.py` - profile deployment and rollback tests
- `tests/integration/test_host_adapter_controller.py` - host-source adapter tests

## Operations

### Lossless Hermes update

Do not run `hermes update` directly on a SoulLink-managed production host. The
lossless controller freezes the host checkout (including Git metadata and its
virtual environment), profile configuration/identity/plugins, SoulLink source
and MemFS, and consistent PCLTM/Hermes SQLite backups while Hermes Desktop is
stopped. Recovery artifacts must be outside `HERMES_HOME`; an existing receipt
is never overwritten.

```bash
# Read-only admission check
python -m soul_link.hermes_update preflight \
  --soullink-root /path/to/Soul-Llink \
  --host-root /path/to/hermes-agent \
  --hermes-home /path/to/hermes-profile

# Create and verify a recovery point without updating
python -m soul_link.hermes_update prepare \
  --soullink-root /path/to/Soul-Llink \
  --host-root /path/to/hermes-agent \
  --hermes-home /path/to/hermes-profile \
  --receipt /outside/hermes/recovery.json

# Close Hermes Desktop first. Runs official update, redeploys SoulLink,
# verifies the live integration, and restores automatically on failure.
python -m soul_link.hermes_update run \
  --soullink-root /path/to/Soul-Llink \
  --host-root /path/to/hermes-agent \
  --hermes-home /path/to/hermes-profile \
  --receipt /outside/hermes/recovery.json

# Explicit disaster recovery
python -m soul_link.hermes_update restore \
  --soullink-root /path/to/Soul-Llink \
  --host-root /path/to/hermes-agent \
  --hermes-home /path/to/hermes-profile \
  --receipt /outside/hermes/recovery.json
```

`run` fails closed for any host delta not declared by the compatibility
manifest, for missing/invalid Git checkouts, and on Windows while Hermes
Desktop is running. A legacy deployment receipt whose backup disappeared is
reported as `orphaned`; it is not treated as rollback protection. Complete v2
and v3 deployment receipts remain rollback-compatible.

There are two distinct deployment operations. Do not substitute one for the other.

### Profile plugin deployment

This installs the SoulLink memory/context plugin entries and managed profile
configuration. `apply` mutates the target profile and therefore requires an
explicit target and receipt path at the invocation boundary.

### Probe
```bash
python -m soul_link.hermes_deploy detect \
  --soullink-root /path/to/Soul-Llink \
  --host-root /path/to/hermes-agent \
  --hermes-home /path/to/hermes-profile
```
Classifies host as: `supported`, `transformable`, or `incompatible`

### Apply
```bash
python -m soul_link.hermes_deploy apply \
  --soullink-root /path/to/Soul-Llink \
  --host-root /path/to/hermes-agent \
  --hermes-home /path/to/hermes-profile \
  --receipt /path/to/profile-deployment-receipt.json
```
Creates backup, applies changes, verifies, writes durable receipt

### Verify
```bash
python -m soul_link.hermes_deploy verify \
  --soullink-root /path/to/Soul-Llink \
  --host-root /path/to/hermes-agent \
  --hermes-home /path/to/hermes-profile
```

### Rollback
```bash
python -m soul_link.hermes_deploy rollback \
  --soullink-root /path/to/Soul-Llink \
  --receipt /path/to/profile-deployment-receipt.json
```
Byte-level restoration from backup, verifies hash match

### Hermes host-source adaptation (low-level)

The canonical production manifest is
`compatibility-soullink-runtime.yaml`; it covers every currently required
SoulLink host transformation. Normal deployment does not invoke this section
separately—the profile deployment command above applies and rolls it back as
part of the same transaction.

Use the commands below only for adapter development, read-only diagnostics, or
an isolated rehearsal of a reviewed compatibility manifest. `detect` and
`verify` are read-only; `apply` and `rollback` mutate the specified host
checkout.

```bash
python -m soul_link.host_adaptation detect \
  --manifest adapters/hermes/compatibility-soullink-runtime.yaml \
  --host-root /path/to/hermes-agent

python -m soul_link.host_adaptation apply \
  --manifest adapters/hermes/compatibility-soullink-runtime.yaml \
  --host-root /path/to/hermes-agent \
  --receipt /path/to/host-adaptation-receipt.json

python -m soul_link.host_adaptation verify \
  --manifest adapters/hermes/compatibility-soullink-runtime.yaml \
  --host-root /path/to/hermes-agent

python -m soul_link.host_adaptation rollback \
  --manifest adapters/hermes/compatibility-soullink-runtime.yaml \
  --receipt /path/to/host-adaptation-receipt.json
```

## Safety Features

- **Fail-closed**: Any error aborts, no partial state
- **Idempotent**: Repeated apply/rollback safe
- **Atomic backup**: Full receipt before any mutation
- **Hash verification**: Every file backed up and verified
- **Rollback verification**: Byte-level comparison after restore

## Compatibility

Hermes versions: requires the SPI capabilities checked by
`soul_link.hermes_deploy.HermesDeployment.host_contract`; host-source patches
additionally require an applicable compatibility manifest.
