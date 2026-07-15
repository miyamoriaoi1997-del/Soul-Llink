# SoulLink Public 2.0

SoulLink Public 2.0 is an open-source reference runtime for long-running persona agents.
It is designed for agents that need more than a single prompt: stable identity, controllable state,
auditable memory boundaries, emotion-aware mode routing, and safe model selection across long conversations.

This repository is the public extraction of the SoulLink runtime architecture. It keeps the reusable engine,
contracts, and tests, while deliberately excluding private deployment state, private memories, credentials,
logs, and any concrete private character preset.

## What SoulLink Is

SoulLink is a layered persona-agent runtime. Its core idea is simple: a long-running agent should not be
assembled from one unstructured prompt blob. It should be assembled from explicit layers with clear priority,
state ownership, memory contracts, and safety boundaries.

SoulLink separates the runtime into three major capabilities:

- **Persona Engine** — builds the active persona context from layered identity, mode, emotion, task, and style inputs.
- **PCLTM** — governs memory/context continuity, recall boundaries, compaction semantics, and safe active-context assembly.
- **Model Router** — routes OpenAI-compatible requests to upstream models based on explicit runtime metadata.

Together they provide a reference architecture for persona agents that need to stay coherent over time without
letting stale summaries, tool dumps, deployment artifacts, or unsafe memory records silently take over the current turn.

## Why SoulLink Exists

Most persona-agent systems fail in predictable ways once conversations become long-lived:

- identity and behavior are mixed into one prompt and become hard to audit;
- summaries start acting like instructions instead of background context;
- tool outputs leak across turns and become stale evidence;
- memories have unclear priority, provenance, or approval status;
- compression logic changes the meaning of the current request;
- model routing is disconnected from runtime state;
- host-specific deployment details become tangled with reusable agent logic.

SoulLink exists to make those boundaries explicit. The runtime treats persona, memory, context, tools, and routing as
separate layers that can be inspected, tested, and replaced independently.

## Architecture Overview

```text
User / Host Adapter
        |
        v
Persona Engine  <---- runtime mode, emotion state, task scene, style layer
        |
        v
PCLTM Context Assembly  <---- pinned memory, approved records, episodic recall, compaction boundaries
        |
        v
Model Router  <---- explicit metadata, provider policy, model selection
        |
        v
Upstream Model
```

The reusable runtime packages are host-independent. This repository also carries optional reference adapters and
versioned host patchsets under `adapters/`. They are explicit integration surfaces, not hidden dependencies of the core.
Host adaptation must detect capabilities, create a backup, verify the installed result, and support rollback.

## Core Packages

- `packages/persona_engine/` — persona runtime and mode/state orchestration.
- `packages/pcltm/` — memory/context governance primitives, audit helpers, and safe context assembly.
- `packages/model_router/` — OpenAI-compatible routing proxy and example configuration.
- `adapters/` — optional, host-specific integration assets kept outside the reusable package boundary.
- `tests/` — public regression tests for the extracted runtime.

## Persona Engine

The Persona Engine is responsible for assembling the active behavioral layer of the agent. It keeps identity,
mode, emotion, task framing, and style as separate inputs instead of flattening them into an uncontrolled prompt.

Its design goals are:

- **Layered priority** — higher-priority identity and boundary layers cannot be rewritten by lower-priority style or task layers.
- **Mode-aware behavior** — daily, work, intimate, crisis, or other runtime scenes can change expression without replacing identity.
- **Emotion-aware output** — emotion state can influence tone, distance, initiative, and softness while preserving factual discipline.
- **Host independence** — the reusable engine does not require a specific chat platform, file layout, or deployment process.

This lets a persona remain stable while still reacting dynamically to the current relationship, task, and conversation state.

## PCLTM: Persona-Centered Long-Term Memory

PCLTM is SoulLink's exclusive memory and context-governance architecture. It is not just a vector search layer and not just
a summarizer. It is the control plane that decides which continuity records may influence the model, how they are typed,
how they are bounded, and how they are assembled into the active prompt.

The name emphasizes the design goal: long-term memory is centered on the persona runtime. Memory is not allowed to be a
loose pile of retrieved text. It must respect identity, user preference, runtime boundaries, approval state, and the current
conversation's actual latest request.

### PCLTM Responsibilities

PCLTM owns the memory/context boundary for long-running agents:

- **Typed memory records** — memory is treated as structured records with state, bucket, provenance, and intended use.
- **Pinned continuity** — stable identity, user preferences, and architecture invariants can be pinned without becoming noisy chat history.
- **Progressive recall** — older or larger records can remain searchable without being injected into every turn by default.
- **Compaction governance** — summaries and handoff blocks are reference-only unless explicitly promoted by policy.
- **Tool-result hygiene** — tool output is only valid inside the current assistant-tool chain and should not leak into future turns as fresh evidence.
- **Active-context assembly** — final prompt materialization is a deliberate assembly step, not an accidental concatenation of everything remembered.
- **Auditability** — memory candidates and context decisions can be inspected with read-only audit helpers.

### PCLTM Layer Model

A typical PCLTM deployment separates memory into several layers:

- **Runtime invariants** — rules that define non-negotiable behavior of the memory system itself.
- **Pinned records** — approved facts and preferences that should be available across sessions.
- **Episodic records** — conversation-derived memories that may be recalled when relevant.
- **Compaction capsules** — compressed continuity blocks that preserve context but remain reference-only.
- **Tool-chain evidence** — current-turn tool outputs that expire when the active tool chain closes.
- **Host/runtime state** — deployment-specific files, logs, databases, and adapters that must not be committed into source.

This separation prevents a common failure mode in long-context agents: treating every remembered string as equal. PCLTM
requires memory to carry intent. A durable user preference, a stale tool result, a summary of an old task, and the current
user request are not the same kind of information and should not have the same authority.

### PCLTM Context Invariants

The public package includes context-engine invariants that encode this philosophy:

- Compaction and handoff blocks are background reference material, never the latest user request.
- Tool results are valid only inside the currently open assistant-tool chain.
- User, system, and developer turns close the current tool chain.
- Orphaned, late, duplicate, or historically reused tool results are removed from model/context copies.
- Runtime data is kept outside package source and outside public examples.

These rules make the active context safer and easier to reason about. The model should answer the user's current request,
not a stale handoff, a duplicated tool log, or an old compressed summary that happens to be nearby.

### Why PCLTM Is Different

PCLTM is built around governance, not retrieval alone.

A plain memory system asks, "what text is similar to this turn?" PCLTM asks additional questions before anything reaches
the model:

- What type of memory is this?
- Who approved it, and what is its lifecycle state?
- Is it a durable preference, an episodic fact, a runtime invariant, or temporary evidence?
- Can it affect the current turn, or should it remain searchable background context?
- Does injecting it risk overriding the latest user request?
- Is the record safe for this host, mode, and deployment boundary?

That is why PCLTM is the distinctive architecture inside SoulLink. It gives a persona agent continuity without surrendering
control of the active prompt to untyped memory retrieval or opaque compression.

## Model Router

The Model Router is an OpenAI-compatible routing proxy. It allows runtime metadata to participate in provider/model
selection without hardcoding routing behavior inside the persona or memory layers.

Its goals are:

- route requests based on explicit metadata rather than hidden side effects;
- keep upstream provider configuration separate from persona logic;
- support public tests with synthetic configs;
- make model choice auditable and replaceable.

## Public Boundary

This repository intentionally does not include:

- private memories, state files, logs, evidence dumps, or local operator data;
- deployment configuration for a specific private host;
- private host configuration or unversioned host modifications;
- real API keys, provider secrets, or `.env` files;
- private role/persona content;
- copyrighted or private character presets.

The public edition is meant to demonstrate the reusable architecture, not to publish a private running instance.

## Requirements

- Python 3.11 or newer
- Git, when MemFS history or host patch application is used
- [`uv`](https://docs.astral.sh/uv/) is recommended for development and reproducible environments
- An OpenAI-compatible endpoint is optional and needed only when using the model router or an LLM-backed semantic classifier
- PyTorch and Transformers are optional; install the `ml` dependency group only when local neural emotion inference is required

SoulLink does not require Hermes for its core runtime. Hermes support is provided by explicit adapter assets under `adapters/hermes/`.

## Quick Start

### Install from a release wheel

```bash
python -m venv .venv
# Linux/macOS
. .venv/bin/activate
# Windows PowerShell
# .venv\\Scripts\\Activate.ps1

python -m pip install soullink_public_2_0-2.0.0-py3-none-any.whl
soullink init
soullink doctor
```

### Install from source with uv

```bash
git clone <repository-url>
cd Soul-Llink
uv sync --group dev
uv run pytest -q
```

Optional local neural inference dependencies:

```bash
uv sync --group ml
```

### Initialize into explicit paths

Do this for deployments, CI, and tests so runtime data never lands in the source tree accidentally:

```bash
soullink init --db /srv/soullink/pcltm.db --memfs /srv/soullink/memfs --json
soullink doctor --db /srv/soullink/pcltm.db --memfs /srv/soullink/memfs --json
```

Equivalent environment variables:

```bash
export HERMES_PCLTM_DB=/srv/soullink/pcltm.db
export HERMES_PCLTM_MEMFS_ROOT=/srv/soullink/memfs
```

## Command-Line Interface

The wheel installs four console commands:

| Command | Purpose |
|---|---|
| `soullink` | Initialize, inspect, govern, and monitor the PCLTM runtime |
| `pcltm` | Alias of `soullink` |
| `soullink-continuity-gate` | Evaluate pinned continuity artifacts using deployment-owned baselines and policy |
| `soullink-host-adapt` | Detect, apply, verify, and roll back a versioned host patchset |

Useful health and evidence commands:

```bash
soullink doctor --json
soullink index stats --json
soullink index doctor --json
soullink live-context smoke --mode work --query "current task" --json
soullink live-context evidence-smoke --json
soullink governance run --json
```

`live-context evidence-smoke` uses synthetic tool evidence and verifies that context remains bounded, contains one outer PCLTM block, and does not expose the synthetic secret marker.

## Runtime Data Layout

SoulLink source code and runtime state have separate ownership. A default local initialization creates:

```text
var/
├── pcltm-prod.db       # authoritative SQLite event and memory store
└── memfs/
    ├── system/         # runtime invariants and system continuity
    ├── pinned/         # approved durable records
    ├── episodic/       # recallable episodic records
    ├── transient/      # short-lived or retrieve-only material
    └── skills/         # procedural memory exports
```

The repository ignores runtime databases, MemFS state, logs, backups, `.env` files, keys, and generated evidence. Do not publish a runtime directory as source code.

## Hermes Host Integration

Hermes integration is optional and deliberately separated from the core packages. SoulLink owns its required host adaptations as versioned assets under `adapters/hermes/` rather than relying on undocumented manual edits.

A host-adapter lifecycle is:

```bash
soullink-host-adapt detect \
  --manifest adapters/hermes/compatibility.yaml \
  --host-root /path/to/hermes

soullink-host-adapt apply \
  --manifest adapters/hermes/compatibility.yaml \
  --host-root /path/to/hermes \
  --receipt /safe/path/soullink-adapter-receipt.json

soullink-host-adapt verify \
  --manifest adapters/hermes/compatibility.yaml \
  --host-root /path/to/hermes

soullink-host-adapt rollback \
  --manifest adapters/hermes/compatibility.yaml \
  --receipt /safe/path/soullink-adapter-receipt.json
```

Important rules:

1. Run `detect` before mutation.
2. Test against an isolated host copy or worktree first.
3. Keep the receipt until post-install verification is complete.
4. A failed verification triggers rollback; do not delete backup evidence manually.
5. Host compatibility is version-specific. Never apply a patchset to an unknown host revision merely because paths look similar.

The reference memory-provider and plugin manifests are examples of explicit integration boundaries. They do not silently activate themselves during package installation.

## Model Router

`model_router` is an OpenAI-compatible routing proxy. Start from the synthetic configuration:

```bash
cp packages/model_router/config.example.yaml packages/model_router/config.yaml
# edit the local copy; never commit credentials or private endpoints
python -m model_router.app --config packages/model_router/config.yaml
```

The router accepts only HTTP(S) upstream URLs with a hostname, rejects embedded credentials and query/fragment data, strips SoulLink/Hermes routing metadata before forwarding, and avoids recording raw prompts or authorization headers in audit logs.

## Local Neural Emotion Backend

Neural emotion analysis is optional and never authoritative by default. The rule/state transition path remains available when PyTorch, Transformers, or model weights are absent.

Supported model IDs have pinned default revisions in `sentiment_analyzer.py`. Override them explicitly when auditing a different checkpoint:

```bash
export SOULLINK_SENTIMENT_MODEL_ID=tabularisai/multilingual-emotion-classification
export SOULLINK_SENTIMENT_MODEL_REVISION=<audited-commit-sha>
```

Model downloads may be large. Production deployments should prefetch and review weights rather than allowing an unexpected first-request download.

## Testing

Run the public root suite:

```bash
uv run pytest -q
```

Run the Persona Engine source-tree suite, whose historical imports require the package roots on `PYTHONPATH`:

```bash
# Linux/macOS
PYTHONPATH="packages/persona_engine:packages:adapters" uv run pytest -q packages/persona_engine/tests

# Windows Git Bash
PYTHONPATH="packages/persona_engine;packages;adapters" uv run pytest -q packages/persona_engine/tests
```

Build distributions:

```bash
uv build
```

Before publishing, run the fail-closed public release audit after all tests and after cleaning generated logs:

```bash
python scripts/public_release_audit.py --root . --json
```

The audit rejects private identity markers, host-local absolute paths, runtime databases, logs, backups, key material, private production reports, and missing release-policy files. Release archives should be scanned independently as well; a clean source tree does not prove a clean wheel or sdist.

## Security Model

SoulLink assumes that retrieved memory, compaction text, host messages, and tool output may be untrusted. Important boundaries include:

- the latest real user request is not replaced by a summary or handoff capsule;
- tool evidence expires when the active assistant-tool chain closes;
- memory candidates carry lifecycle, provenance, target, and scope information;
- promotion gates bind to independently configured baseline and policy digests;
- monitoring binds to loopback and exposes read-only methods by default;
- runtime logs and model-router audits do not intentionally store raw prompts or credentials;
- host adaptation rejects paths that escape the selected host root and preserves rollback evidence.

See [SECURITY.md](SECURITY.md) for responsible disclosure and trust-boundary details.

## Troubleshooting

### `soullink doctor` reports a missing database or MemFS layout

Run:

```bash
soullink doctor --fix --db /path/to/pcltm.db --memfs /path/to/memfs
```

The operation creates missing scaffolding and does not intentionally delete existing runtime data.

### Persona Engine tests fail with `No module named persona_orchestrator`

Use the documented source-tree `PYTHONPATH`. Installed wheel imports use `persona_engine.persona_orchestrator`.

### Local neural model is unavailable

Install the `ml` dependency group, verify the model cache and pinned revision, or continue with the rule-based fallback. Do not claim neural inference is active merely because configuration exists.

### Host adaptation reports `incompatible`

Stop. Verify the host root, host revision, required paths, and patchset version. Do not force-apply the patch. An incompatible result is a safety boundary, not an inconvenience to bypass.

### Web monitor refuses a non-loopback address

This is intentional. The bundled monitor is designed as a local read-only surface and rejects external binding by default.

## Project Status and Scope

This repository is an open-source reference runtime, not a copy of a private running persona. It is suitable for development, review, controlled integration, and experimentation. Production adoption still requires deployment-specific threat modeling, persistence/backup policy, provider configuration, and host-version validation.

Public templates are intentionally neutral. Configure your own lawful persona identity, relationship labels, and content policies in deployment-owned overlays; do not commit private overlays or user memories back into the public repository.

## Development Notes

- Keep examples generic and synthetic.
- Keep host-specific adapters explicit and outside the reusable packages.
- Do not commit runtime databases, memory exports, logs, `.env` files, or provider credentials.
- Treat tests as the public contract for the extracted runtime behavior.
- Prefer explicit metadata and typed records over implicit prompt concatenation.

## Safety Notes for Contributors

SoulLink's public value comes from its boundaries. Contributions should preserve those boundaries:

- no private state or secrets in source;
- no real user memory dumps in tests;
- no copyrighted/private persona preset in examples;
- no hidden coupling between host adapters and core packages;
- no behavior that allows stale tool output or compaction summaries to override the latest user request.

## Runtime Initialization

Runtime data is created locally after installation and is not committed to the
repository. The public defaults are:

- DB: `var/pcltm-prod.db`
- MemFS root: `var/memfs`
- MemFS directories: `system`, `pinned`, `episodic`, `transient`, `skills`

Initialize a fresh checkout or installed environment with:

```bash
soullink init
# equivalent alias:
pcltm init
```

Check readiness without modifying existing runtime data:

```bash
soullink doctor
```

Create missing DB/MemFS scaffolding before checking:

```bash
soullink doctor --fix
```

Both commands are idempotent. They create missing parents, bootstrap the SQLite
schema through `EventStore`, and create the MemFS directory layout without
deleting or overwriting runtime data. Use `--db` and `--memfs` for explicit
paths, or set `HERMES_PCLTM_DB` and `HERMES_PCLTM_MEMFS_ROOT`.

Hermes integration is optional. The core runtime exposes initialization,
context governance, MemFS, and host-neutral adapter primitives. Reference Hermes
adapter and patchset assets live under `adapters/`; using them does not make the
core packages depend on Hermes. Treat host compatibility as version-specific and
run detect, apply, verify, and rollback against an isolated host before production use.

## Release and Packaging

This repository builds standard Python source and wheel distributions:

```bash
uv run --with build python -m build
uv run --with twine python -m twine check dist/*
```

The released wheel contains the reusable host-neutral packages:

- `soul_link`
- `pcltm`
- `persona_engine`
- `model_router`

Runtime-specific adapter examples remain in the source tree but are not included
as hidden core dependencies. The `soullink-host-adapt` command applies only an
explicit manifest and records a rollback receipt.

Before publishing, run:

```bash
python scripts/public_release_audit.py --root .
```

## License

SoulLink Public 2.0 is released under the MIT License. See [LICENSE](LICENSE).
