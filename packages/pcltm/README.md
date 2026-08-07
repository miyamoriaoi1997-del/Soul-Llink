# PCLTM package

Host-independent memory/context governance primitives for Soul-Link public 2.0.

Current contents:

- `context_engine.py` — context assembly and sanitization.
- `memory_adapter.py` — layered recall and memory selection helpers.
- `runtime_paths.py` — shared public defaults for DB and MemFS runtime paths.
- `cli.py` — `soullink` / `pcltm` runtime initialization and doctor commands.
- `pcltm_audit.py` — read-only audit helpers for memory stores.
- `continuity_baseline.py` — pure normalization of existing identity/continuity/ADS/summary artifacts.
- `continuity_gate.py` — read-only baseline/candidate shadow comparison and promotion blocking; see `docs/continuity-preservation-gate.md`.

Boundary:

- Core logic lives here.
- Host-specific wrappers and private deployment paths are outside this public package.
- Runtime data is never part of the package source tree.

Important context-engine invariants:

- Compaction/handoff blocks are reference-only and never become the latest user request.
- Tool results are valid only inside the currently open assistant-tool chain.
- User/system/developer turns close the current tool chain.
- Orphan, late, duplicate, or historically reused tool results are removed from model/context copies.
