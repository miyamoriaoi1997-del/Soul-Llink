# Soul State Orchestrator

Soul State Orchestrator is the planned layered prompt runtime for Hermes Persona Engine. It keeps one immutable core identity while selecting lightweight behavior layers for the current conversational state.

## Principle: one identity, many behavior layers

The system is not multiple personas competing with each other. It is one continuous persona with mode-specific behavior deltas:

```text
Core Identity Layer        # identity, relationship anchor, global discipline
Mode SOUL Layer        # daily / work / intimacy / repair / conflict / system_maintenance
Memory Profile         # selected by mode; names/candidates only in Phase 1
Emotion Modifier       # existing dynamic emotion block, appended last
Final Prompt Preview   # composed by PromptComposer; shadow-only in Phase 1
```

Mode layers must not redefine identity. They only adjust expression, priorities, and scene rules.

## Current Phase: Shadow Mode

Phase 1 is intentionally non-invasive:

- It does not modify Hermes core files.
- It does not replace the active system prompt.
- It does not enable sex mode.
- It only produces a `StatePacket`, prompt preview hash, selected layers, and JSONL audit logs.

This lets maintainers observe classification accuracy before giving the orchestrator control over live prompt injection.

## Modes

| Mode | Purpose |
|------|---------|
| `daily` | normal private conversation |
| `work` | technical/debugging/execution work |
| `system_maintenance` | Hermes/gateway/persona-engine/config/memory/prompt work |
| `intimacy` | verbal closeness, jealousy, reassurance, hugs/kisses; not sex |
| `repair` | apology, crisis, emotional stabilization, damage repair |
| `conflict` | relationship rupture, hostility, defensive state |
| `creative` | writing, design, stories, critique |
| `sex_candidate` | explicit sex-scene progression detected, blocked to intimacy in Phase 1 |

## Safety gates

`sex_candidate` is never an active adult boundary layer in Phase 1. It maps to `intimacy` with `sex_shadow_only`.

Future active sex mode must require all of:

1. explicit user progression;
2. desire gate permission from emotion score;
3. no crisis/repair/work/system/public-context block;
4. dedicated regression tests proving hugs/kisses/cuddling do not trigger sex;
5. explicit user approval before enabling.

Crisis/self-harm/collapse markers force `repair` mode and block sexual escalation.

## Shadow pipeline

```python
from persona_orchestrator import StateOrchestrator

orchestrator = StateOrchestrator(base_dir='/path/to/hermes-persona-engine')
packet = orchestrator.analyze_turn(
    user_message='帮我检查 gateway 日志',
    emotion_state={'emotion_score': 1.0},
)
print(packet.mode, packet.selected_layers, packet.prompt_hash)
```

Pipeline:

```text
user message
  -> ModeClassifier
  -> TransitionManager
  -> MemorySelector
  -> PromptComposer
  -> StatePacket
  -> JSONL log
```

## CLI probe

```bash
python scripts/orchestrator_probe.py "帮我检查 gateway 日志"
python scripts/orchestrator_probe.py "[assistant name]我想你了" --score 2.5
python scripts/orchestrator_probe.py "我们成人亲密" --score 4.5
python scripts/orchestrator_probe.py "[pet name]帮我看 gateway 日志" --semantic-shadow --semantic-backend local
```

Expected Phase-1 behavior:

- gateway/system messages select `system_maintenance`;
- affectionate messages select `intimacy`;
- explicit sex messages report `sex_candidate` semantics but active mode remains `intimacy` and selected layers do not include `sex`.

## SOUL layer validation

Layer templates have a small contract validator so multi-SOUL changes fail before prompt composition drifts:

```bash
python scripts/validate_soul_layers.py
python scripts/validate_soul_layers.py --json
```

The validator checks that:

- all required `SOUL.<layer>.template.md` files exist;
- each layer has the expected heading;
- `core` is the only layer allowed to define core identity;
- non-core layers explicitly say they must not redefine identity;
- `intimacy` does not auto-escalate to sex;
- `overlay_intimacy` does not replace the active task mode;
- `sex` remains disabled by default in Phase 1.

## Observability

For live-runtime preparation, use `RuntimeShadowAdapter` to compute a candidate prompt and write a redacted JSONL audit record without installing that candidate into Hermes or deciding runtime switching:

```python
from persona_orchestrator import RuntimeShadowAdapter

adapter = RuntimeShadowAdapter('/path/to/hermes-persona-engine')
record = adapter.analyze_runtime_turn(
    host_system_prompt=current_system_prompt,
    user_message=user_message,
    emotion_state={'emotion_score': 1.0},
    emotion_modifier=current_emotion_modifier,
    previous_mode='system_maintenance',
    platform='telegram',
    session_id=session_id,
)
assert record['active'] is False
```

The runtime shadow log hashes user messages and stores mode/layer/safety metadata only. It does not replace the live prompt or decide switching policy.

A safe Hermes plugin scaffold can be prepared with:

```bash
python scripts/runtime_shadow_plugin_probe.py --dry-run
python scripts/runtime_shadow_plugin_probe.py --write
```

The generated plugin registers only `pre_llm_call`, returns `None`, and therefore does not inject context, mutate the live prompt, or control runtime switching. Enabling it still requires an explicit Hermes config change and gateway restart; keep it disabled until the dry-run output and plugin files have been reviewed.

Each turn can be logged as JSONL with:

- mode;
- submode;
- confidence;
- reason;
- transition;
- selected_layers;
- memory_profile;
- safety_flags;
- emotion_score;
- desire_tier;
- prompt_hash;
- shadow_only.

These logs are for audit/debugging only and should not be inserted into normal user-visible replies.

## Rollout plan

1. Shadow only: collect decisions and prompt hashes.
2. Enable active `daily` / `work` only after shadow accuracy is acceptable.
3. Enable `repair` / `conflict` after crisis and relationship regression tests pass.
4. Enable `intimacy` after bracket-ban and non-sex-boundary tests pass.
5. Enable `sex` only after explicit approval and strict desire/crisis/work/public gates.

## Integration warning

Do not modify Hermes core `run_agent.py` unless deliberately integrating into a host agent and the user has approved that scope. The public persona engine should remain a plugin-layer reference first.
