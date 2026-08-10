# SoulLink Public 2.2.2

SoulLink Public 2.2.2 is a memory-governance and observability patch release.

## Added

- Natural-language durable facts can enter the governed candidate pipeline without requiring command-shaped wording.
- Admission tiers distinguish durable preferences, identity facts, and system conventions from transient conversation.
- Multi-source candidate evidence is preserved through pending review and authority reopening.
- New regression coverage exercises positive durable-memory promotion and negative transient-message rejection.

## Changed

- The WebUI now treats active governed claims as the canonical durable-memory metric.
- Event-derived memory is shown as a lineage breakdown of Active claims rather than as a separate storage stage.
- Raw events and legacy memory records remain visible as evidence and compatibility data without inflating the Active count.
- Hermes history ingestion binds classification output into its source hash so classification changes are reprocessed deterministically.

## Safety and privacy

- Event-derived writes continue to require reopenable source snapshots, active governance, matching payload hashes, and allowed source roles.
- Short-lived observations and ordinary conversational anecdotes remain retrieval-only and do not create durable candidates.
- Public fixtures use neutral synthetic examples. The release contains no private persona data, user memories, conversations, credentials, runtime databases, logs, telemetry, or deployment state.

## Verification

The final release tree is verified with:

- focused memory and monitoring tests;
- the complete public test suite;
- JavaScript syntax and Python compilation checks;
- the strict public release/privacy audit;
- wheel and source-distribution member inspection;
- checksum verification;
- clean-environment wheel installation and CLI lifecycle smoke tests.

Exact final counts and artifact hashes are recorded in `SHA256SUMS.txt` and the GitHub Actions run for the release commit.

## Install

Download these assets from the `v2.2.2` release:

- `soullink_public-2.2.2-py3-none-any.whl`
- `soullink_public-2.2.2.tar.gz`
- `SHA256SUMS.txt`

Verify the checksums before installing, then install the wheel in a clean Python 3.11+ environment:

```bash
python -m pip install soullink_public-2.2.2-py3-none-any.whl
soullink init
soullink doctor
```
