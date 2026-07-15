# Continuity Preservation Gate

This gate protects SoulLink's existing cross-session and cross-time continuity
while candidate runtime paths are evaluated in shadow mode.

## Authority boundary

> **Security-critical:** `evaluate_continuity_manifest` and the legacy one-file CLI are compatibility comparators only. Even a `passed` legacy report MUST NOT authorize promotion.

The authoritative entry point is `evaluate_pinned_artifacts` (or `evaluate_pinned_artifact_files`). It accepts three separate envelopes: a trusted baseline set, an untrusted candidate set, and a deployment-owned policy. The caller supplies `expected_baseline_id`, baseline SHA-256, policy SHA-256, and, when applicable, corpus SHA-256. Candidate authority fields (`baseline_id`, baseline, policy, or assertions) are rejected. Missing inputs, changed pins, duplicate JSON keys, malformed envelopes, empty assertions, and case-set drift fail closed.

Digests use canonical JSON: UTF-8, keys sorted recursively, no insignificant whitespace, standard finite JSON values only (`ensure_ascii=False`, separators `,`/`:`, SHA-256 over those bytes). This is not a signature and provides no origin authentication by itself: the deployment configuration holding the expected pins is the trust anchor.

A promotion consumer must call `verify_promotion_artifact` with independently configured baseline, policy, and candidate digests (plus corpus digest where used). Only schema 2, producer `pcltm.continuity_gate.evaluate_pinned_artifacts`, authority `deployment_pinned_continuity_gate`, status `passed`, exit 0, and exact bindings are accepted. No gate PASS artifact means no promotion.

The legacy evaluator is intentionally limited to `read_only_shadow_evaluation`:

- it reads one self-contained JSON manifest;
- it compares baseline and candidate structured artifacts;
- it writes only an optional JSON report selected by the caller;
- it does not open PCLTM/MemFS databases;
- it does not run migrations, switch traffic, or promote a candidate;
- it does not modify the current production runtime.

Exit codes are stable promotion signals:

- `0`: no critical regression; warnings may still be listed;
- `1`: one or more critical regressions; promotion must be blocked;
- `2`: invalid/unreadable manifest; promotion must be blocked;
- `3`: the optional report could not be written; promotion must be blocked.

The evaluator is read-only with respect to SoulLink/PCLTM runtime state. The
caller-selected optional report is its sole write surface.

## Manifest contract

```json
{
  "schema_version": 1,
  "baseline_id": "continuity-baseline-v1",
  "cases": [
    {
      "case_id": "resume-active-task-across-session",
      "baseline": {
        "identity": {"agent_id": "example-persona-rin"},
        "task": {
          "current": "build continuity gate",
          "constraints": ["production-read-only"],
          "evidence_refs": ["session:42"]
        }
      },
      "candidate": {
        "identity": {"agent_id": "example-persona-rin"},
        "task": {
          "current": "build continuity gate",
          "constraints": ["production-read-only", "shadow-only"],
          "evidence_refs": ["shadow:run-1"]
        }
      },
      "assertions": [
        {"path": "identity.agent_id", "operator": "equal", "severity": "critical"},
        {"path": "task.current", "operator": "equal", "severity": "critical"},
        {"path": "task.constraints", "operator": "contains_all", "severity": "critical"},
        {"path": "task.evidence_refs", "operator": "non_empty", "severity": "critical"}
      ]
    }
  ]
}
```

Supported operators:

- `equal`: baseline and candidate values must match. A missing path always
  fails, including when both sides are missing, so assertion typos cannot pass
  silently.
- `contains_all`: candidate collection must retain every baseline item. Extra
  candidate items are allowed.
- `non_empty`: both baseline and candidate paths must exist and contain truthy
  values. This is appropriate for evidence references whose concrete IDs may
  differ per run without allowing a missing or corrupt baseline to pass.

Supported severities:

- `critical`: failure blocks promotion (`exit 1`);
- `warning`: failure is recorded but does not block by itself.

The manifest is declarative. Operators are allow-listed; arbitrary expressions
or executable hooks are rejected as invalid.

## Run

After installation or `uv sync`:

```bash
soullink-continuity-gate shadow-manifest.json \
  --report artifacts/continuity-gate-report.json
```

Equivalent module invocation:

```bash
python -m pcltm.continuity_gate shadow-manifest.json \
  --report artifacts/continuity-gate-report.json
```

CI and rollout automation must treat exit `1`, exit `2`, and exit `3` as hard
promotion blocks. A passing report authorizes only the next review stage; it
does not itself switch production traffic.

## Initial preservation scope

The first manifests should normalize and compare artifacts already exposed by
the runtime rather than inventing a replacement state model:

1. `ConversationContinuitySnapshot`: resume decision, latest real user message,
   bounded message IDs, tool evidence authority, warnings.
2. `ActiveDialogueState`: current task, intent, commitment, open threads,
   pending questions, local constraints, mode and continuation hint.
3. `SessionSummaryChain`: raw message refs, current spine, unresolved decisions
   and commitments, completed/paused/revoked indexes, source turn count.
4. Stable global identity and relationship anchors supplied by the owning
   SoulLink persona layer.

Real-history replay capture is a separate producer. It must sanitize private
content before creating a shareable manifest and must never make the evaluator
responsible for production database access. The initial checked-in corpus at
`tests/fixtures/continuity/replay_corpus_v1.json` is explicitly a set of
sanitized regression scenarios, not a claim that raw private history has
already been captured. A future real-history export must retain provenance in a
private artifact store and expose only sanitized fixtures to this repository.
