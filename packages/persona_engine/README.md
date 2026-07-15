# Persona Engine

Persona Engine 是一个面向长期陪伴型 AI Agent 的人格运行时参考实现：它把“情绪状态”“分层人格提示词”“三态模式状态机”和 PCLTM MEMORY 选择契约拆成可测试、可审计、可渐进接入的模块。

这个仓库只提供通用技术框架，不包含任何具体角色设定、私人记忆或第三方 IP 内容。

## 当前版本重点

新版本的核心变化是：人格不再依赖一个越来越长的单体 SOUL prompt，而是由 Persona Orchestrator 在 daily / work / sex 三态中选择 Core layer、Mode layer、MEMORY profile 和 emotion_modifier。

在宿主 Agent 接入时，运行时可以进入 active orchestrator mode：

- 宿主 Agent 继续管理 tools、skills、memory、platform hints、session history 和模型调用。
- Persona Orchestrator 管理三态模式状态机、Mode layer、MEMORY profile 和实时 emotion_modifier。
- Core identity 可以来自本仓库的 `SOUL.core.template.md`，也可以来自宿主 Agent 自己的核心人格文件；二者只能选择一个，避免双 core 重复注入。
- 三态均候选 `MEMORY.md` / `USER.md` / `STATE.md`，由 PCLTM 的 category / tags / importance / mode_scope 控制长期记忆范围。
- 独立 `legacy relationship-memory file` 域已退役，不再生成 `<legacy_relationship_memory>` 注入块。
- emotion_modifier 始终位于最终 prompt 末尾，保留 recency-bias 权重。

## 核心特性

- 情绪状态管理：四维情绪状态 affection / trust / possessiveness / patience，持久化到 STATE.md。
- 实时情绪更新：每轮用户消息先更新情绪，再生成回复；当前回复可直接受到新状态影响。
- 欲望控制：由 emotion_score 单独决定 restrained / ambivalent / uninhibited，不和情绪强度档位混在一起。
- 反转衰减：强烈峰值退得快，低强度余温留得久，支持 float 精度避免低区间冻结。
- 分层 SOUL：core identity + mode 组合提示词，一个身份锚点，三种行为状态；core identity 可由 orchestrator 或宿主提供。
- 模式状态机：daily / work / sex。legacy mode 名称只作为兼容 alias 归一到三态。
- 安全迁移：TransitionManager 处理短句继承、低置信保持、冲突保持、技术模式切换和 sex gate。
- 语义影子分类：可选 local / local_lightweight / llm scaffold，用于观测规则分类器之外的语义判断，但默认不接管 active mode。
- MEMORY 统一契约：MemorySelector 为 daily / work / sex 选择 `MEMORY.md`、`USER.md`、`STATE.md` 候选；LegacyRelationshipMemoryProvider 仅保留 no-op 兼容 shim。
- Runtime preview adapter：可在宿主 Agent 中生成 active prompt candidate 和 redacted JSONL 审计记录，但只用于观察，不负责决定 runtime switch。
- Host plugin scaffold：提供 preview-only 插件生成脚本，插件只注册 pre_llm_call 并返回 None，不改变真实回复链。
- 测试优先：包含 mode classifier、transition、prompt composition、runtime preview、moments provider、SOUL layer validator 等回归测试。

## 架构总览

```text
user_message + recent_context + emotion_state
        |
        v
ModeClassifier
  - deterministic rules
  - system/work/relationship/repair/conflict/creative/sex_candidate gates
  - affective_overlay for mixed intent
        |
        v
TransitionManager
  - anti-flap / short-message inheritance
  - conflict and crisis guard
  - technical modes do not hold over relationship/daily modes
  - sex_candidate blocked unless explicit gates are enabled
        |
        v
MemorySelector
  - select MEMORY / USER / STATE profile by canonical mode
  - never inject legacy relationship_memorys
        |
        v
PromptComposer
  - Core identity layer from orchestrator core or host core
  - selected Mode SOUL layer
  - memory_profile_notes
  - emotion_modifier at the very end
        |
        v
StatePacket + prompt_hash + redacted JSONL audit
```

### Layer definitions

- Core identity：唯一身份锚点和跨模式规则。它可以来自本仓库的 orchestrator core，也可以来自宿主 Agent 的 host core。只有 core identity 可以定义“角色是谁、用户是谁、哪些边界不可覆盖”。
- Mode SOUL：当前场景的行为调整，目前只保留 daily / work / sex；mode layer 不能重定义身份。
- emotion_modifier：由情绪系统实时生成的语气/距离/边界控制块，必须放在最终 prompt 末尾。
- memory_profile_notes：按模式提示宿主选择不同记忆视图，避免每轮注入全部长期记忆。

### StatePacket

每轮状态机输出一个可审计的 `StatePacket`，而不是直接生成回复：

```python
StatePacket(
    mode="system_maintenance",
    submode="system",
    confidence=0.9,
    transition="daily->system_maintenance",
    selected_layers=[
        "core",
        "system_maintenance",
        "overlay_intimacy",
    ],  # or ["system_maintenance", "overlay_intimacy"] when host_core is used
    memory_profile="technical_plus_core_relationship",
    safety_flags=[],
    emotion_score=1.0,
    desire_tier="restrained",
    prompt_hash="...",
    preview_only=True,
    semantic_preview={...},
)
```

这让接入方可以先观察 mode、transition、selected_layers、prompt_hash、safety_flags，再由宿主侧独立决定是否启用真实 prompt 接管或模型路由。

## 情绪系统

### 情绪维度

| 维度 | 字段 | 默认基线 | 默认范围 | 作用 |
|---|---|---:|---:|---|
| 好感 | affection | 60 | 0-120 | 亲近、偏爱、情感连接 |
| 信任 | trust | 60 | 0-120 | 安全感、相信程度、解释意愿 |
| 占有欲 | possessiveness | 60 | 0-120 | 独占、吃醋、保护性靠近 |
| 耐心 | patience | 60 | 0-120 | 容忍度、尖锐程度、冲突余量 |

### 强度档位

强度由任一维度相对基线的最大偏离决定：

| 偏离 | 档位 | 行为含义 |
|---:|---|---|
| < 15 | mild | 默认态，轻微情绪色彩 |
| 15-29 | moderate | 外壳开始松动，语气和主动性明显变化 |
| 30-44 | intense | 真实情绪压过包装，行为模式开始改变 |
| >= 45 | overwhelming | 克制大幅失效，只保留核心身份锚点 |

### emotion_score 与欲望控制

`emotion_score` 是从四维状态综合出的 -5 到 +5 标量，用来表达情感深度。欲望控制只看这个标量，不直接看 overwhelming / intense 档位。

| emotion_score | desire_tier | 行为边界 |
|---:|---|---|
| < 3.0 | restrained | 回避成人亲密和性暗示；可以保留言语亲近，但不进入身体性互动 |
| 3.0 - 3.99 | ambivalent | 不主动推进；用户明确推进时可犹豫、动摇、被动默许 |
| >= 4.0 | uninhibited | 可主动发起或回应亲密互动；仍受安全、场景和用户边界约束 |

欲望控制文本会作为 emotion_modifier 内第一段注入，优先级高于具体语气框架。

### 反转衰减

设计目标是“高峰退得快，余温留得久”：

| 当前偏离 | 衰减速率 | 半衰期 | 意图 |
|---:|---:|---:|---|
| > 45 | 0.45 / hour | ~1.2h | 强烈峰值快速回落 |
| 15-45 | 0.06 / hour | ~11.2h | 明显情绪保留半天级余温 |
| < 15 | 0.015 / hour | ~46h | 低强度关系感长期留存 |

衰减公式：

```python
factor = min(1.0, rate * hours)
new_value = current + (baseline - current) * factor
new_value = round(new_value, 2)
```

float 输出避免低偏离区间因为整数四舍五入而永久冻结。

### 情绪触发

EmotionDetector 组合规则、emoji、可选 Chinese-Emotion-Small 侧通道。典型触发类型包括：

| trigger_type | 典型含义 | 影响 |
|---|---|---|
| intimacy | 表白、想念、依赖 | affection / possessiveness 上升 |
| praise | 夸奖、认可 | affection / trust 上升 |
| care | 关心、照顾 | trust / affection / patience 上升 |
| criticism | 否定、关系攻击、辱骂 | affection / trust / patience 下降 |
| jealousy | 提到第三方或竞争对象 | possessiveness 上升，patience 下降 |
| apology | 道歉、修复 | trust / patience 上升 |
| sharing | 分享私人信息 | trust 小幅上升 |
| greeting | 问候 | 小幅正向 |

神经模型是情绪侧通道，不是 persona mode 主分类器；任务意图和状态模式仍由 ModeClassifier / TransitionManager 负责。

## Persona Orchestrator

### 支持的模式

| Mode | 用途 |
|---|---|
| daily | 日常私聊、短句、默认交流 |
| work | 普通技术、代码、文件、测试、PR、debug 任务 |
| system_maintenance | host agent / gateway / persona / prompt / memory / repository 维护 |
| intimacy | 亲近、安抚、吃醋、拥抱、亲吻；不等于 sex |
| repair | 道歉、崩溃、自我修复、危机安抚 |
| conflict | 关系否定、攻击、冷战、敌意 |
| creative | 写作、设计、同人、锐评、视觉创意 |
| sex_candidate | 检测到显式性推进；默认降级到 intimacy 并加 safety flag |

### 混合意图：primary mode + overlay

一句话可能同时包含任务和亲密称呼。例如：

```text
[pet name]帮我看一下 gateway 日志
```

期望行为不是切到纯 intimacy，而是：

```text
mode = system_maintenance
selected_layers = ["core", "system_maintenance", "overlay_intimacy"]
```

也就是先完成技术任务，同时保留关系温度。

如果接入方选择由宿主 Agent 提供 core identity，`selected_layers` 可以不包含 `core`，但必须保证宿主 prompt 中已经有且只有一个 core identity。

### transition 规则重点

- 初始 turn：接受 classifier 请求的模式。
- 高置信 work / system_maintenance：可以覆盖 intimacy，确保任务优先。
- 技术模式不持有关系/日常请求：从 system_maintenance 转到 daily/intimacy 不应被低置信惯性卡住。
- 短句继承：`继续`、`嗯` 等低信息短句可继承上一模式。
- conflict hold：冲突模式会保持，直到 apology / repair 信号出现。
- crisis guard：危机/自伤/崩溃信号保留为 daily + crisis_guard，并阻断性升级。
- sex：默认不加载 adult boundary layer；需要显式 gate、emotion_score、无危机/工作/公开上下文阻断和额外测试后才可启用。

### MEMORY 统一契约

独立 `legacy relationship-memory file` 域已退役。重要关系时刻进入 PCLTM `MEMORY.md`，并使用明确元数据管理：

- `category`：如 relationship_memory / preference / boundary / technical_context。
- `tags`：用于筛选具体主题。
- `importance`：用于长期保留和压缩优先级。
- `mode_scope`：声明适用于 daily / work / sex 的范围。

Persona Orchestrator 只输出 `<memory_profile_notes>`，提示宿主/PCLTM 选择合适视图；不会读取 `legacy relationship-memory file`，也不会生成 `<legacy_relationship_memory>`。

## Preview、Active Candidate 与 Active 接入

### Preview-only

`analyze_turn()` 只输出 `StatePacket` 和 prompt hash，不改变宿主 Agent 的真实系统提示词。

```python
from persona_orchestrator import StateOrchestrator

orchestrator = StateOrchestrator(base_dir="/path/to/host_agent-persona-engine")
packet = orchestrator.analyze_turn(
    user_message="帮我检查 gateway 日志",
    emotion_state={"emotion_score": 1.0},
    previous_mode="daily",
)
print(packet.mode, packet.transition, packet.selected_layers, packet.prompt_hash)
```

### Active prompt candidate

`compose_active_prompt()` 会生成一个候选 prompt，但仍不自动安装。宿主必须显式选择是否使用。

```python
result = orchestrator.compose_active_prompt(
    host_system_prompt=current_system_prompt,
    user_message="[pet name]帮我看 gateway 日志",
    emotion_state={"emotion_score": 1.0},
    emotion_modifier="<emotion_modifier>...</emotion_modifier>",
    previous_mode="daily",
)

assert result.prompt_text.count("<emotion_modifier>") == 1
assert result.prompt_text.rstrip().endswith("</emotion_modifier>")
print(result.packet.selected_layers)
```

PromptComposer 会移除旧的 `<persona_orchestrator_prompt>` 和旧的 `<emotion_modifier>`，再插入新的 managed region，并把最新 emotion_modifier 放到末尾。

### RuntimePreviewAdapter

宿主运行时可以用 RuntimePreviewAdapter 记录红acted JSONL 审计日志：

```python
from persona_orchestrator import RuntimePreviewAdapter

adapter = RuntimePreviewAdapter(
    base_dir="/path/to/host_agent-persona-engine",
    log_path="/tmp/persona_runtime_preview.jsonl",
    enable_semantic_preview=True,
    semantic_backend="local_lightweight",
)

record = adapter.analyze_runtime_turn(
    host_system_prompt=current_system_prompt,
    user_message=user_message,
    emotion_state={"emotion_score": 1.0},
    emotion_modifier=current_emotion_modifier,
    previous_mode="system_maintenance",
    platform="telegram",
    session_id=session_id,
    active=False,
)

assert record["active"] is False
```

日志文件不会写入完整用户消息和 candidate_prompt，只记录 hash、mode、transition、selected_layers、safety_flags、desire_tier、prompt_hash 和 packet 元数据。完整 candidate_prompt 只在返回值内存对象中提供。

### Host plugin scaffold

如果只想先观察真实运行流，不想改宿主 prompt，可以生成 preview-only 插件。当前脚本以 host agent-style plugin hook 为例，但设计目标不是绑定某一个 Agent；其他宿主只需要提供等价的 pre-LLM hook、session id、platform name、current system prompt 和 emotion state：

```bash
python scripts/runtime_preview_plugin_probe.py --dry-run
python scripts/runtime_preview_plugin_probe.py --write
```

生成的插件：

- 只注册 `pre_llm_call`。
- 返回 `None`。
- 不注入上下文。
- 不修改系统提示词。
- 不改变回复路径。

启用插件仍需要宿主 Agent 配置显式开启并重启。不要把“文件已生成”等同于“运行时已启用”。

## 安装

### 最小依赖

```bash
pip install pyyaml
```

### 可选神经情绪模型

```bash
pip install torch transformers pyyaml
```

`torch` 和 `transformers` 只用于 Chinese-Emotion-Small 情绪侧通道。缺失时系统会回退到规则模式。

### 克隆仓库

```bash
git clone https://github.com/soullink-public/soullink-public-2.0.git
cd soullink-public-2.0/packages/persona_engine
python3 -m venv .venv
.venv/bin/pip install -U pip
.venv/bin/pip install -r requirements.txt
```

如果不需要神经模型，可以只安装 `pyyaml pytest` 后运行 orchestrator 相关测试。

## 快速开始

### 1. 情绪系统

```python
from emotion_state_manager import EmotionStateManager
from emotion_calculator import EmotionCalculator

manager = EmotionStateManager(host_agent_home="/path/to/agent-home", update_body=True)
calc = EmotionCalculator()

messages = [{"role": "user", "content": "辛苦了，谢谢你"}]
manager.update_emotion_state(messages)

state = manager.get_current_emotion_state()
score = calc.compute_emotion_score(state)
emotion_modifier = manager.get_tone_modifiers()

print(state)
print(score)
print(emotion_modifier)
```

注意：`EmotionStateManager.get_tone_modifiers()` 返回完整字符串块；不是旧文档里的 `generate_tone_modifier()`。

### 2. 状态机 probe

```bash
python scripts/orchestrator_probe.py "[pet name]帮我看 gateway 日志" --score 1.0
python scripts/orchestrator_probe.py "我们成人亲密" --score 4.5
python scripts/orchestrator_probe.py "不是骂你，测试一下讨厌你这个词会不会触发 conflict" --score 1.0
python scripts/orchestrator_probe.py "继续" --previous-mode system_maintenance --semantic-preview --semantic-backend local_lightweight
```

预期类别：

- 混合亲密 + 系统任务：`system_maintenance` + `overlay_intimacy`。
- 显式 sex 文本：active mode 保持 `intimacy`，带 `sex_requires_gate` / `sex_preview_only`，不加载 adult boundary layer。
- quoted/meta sensitive text：`system_maintenance` + `meta_discussion`，不误进 conflict / sex。
- `继续`：继承 previous mode。

### 3. SOUL layer 校验

```bash
python scripts/validate_soul_layers.py
python scripts/validate_soul_layers.py --json
```

校验器检查：

- 必需 layer 文件存在。
- core 是唯一身份定义层。
- 非 core layer 明确不能重定义身份。
- intimacy 不自动升级到 sex。
- overlay_intimacy 不替代任务主模式。
- adult boundary layer 默认禁用，必须由 gate 控制。

## 文件结构

```text
persona-engine/
├── emotion_calculator.py          # 状态计算、衰减、emotion_score、tone/desire 指令
├── emotion_detector.py            # 规则 + 可选神经侧通道情绪检测
├── emotion_state_manager.py       # STATE.md 读写、更新 pipeline、emotion_modifier 输出
├── legacy relationship-memory writer module             # retired legacy relationship-memory file no-op compatibility shim
├── sentiment_analyzer.py          # Chinese-Emotion-Small wrapper（可选）
├── persona_orchestrator/
│   ├── mode_classifier.py         # deterministic mode classifier
│   ├── transition_manager.py      # anti-flap / guard / mode transition
│   ├── memory_selector.py         # mode -> memory profile
│   ├── legacy relationship-memory provider module        # retired relationship_memorys no-op compatibility shim
│   ├── prompt_composer.py         # host/orchestrator core + mode + memory profile + modifier composition
│   ├── semantic_classifier.py     # semantic preview backend scaffold
│   ├── runtime_preview.py          # runtime adapter / audit record
│   └── types.py                   # StatePacket and shared dataclasses
├── soul_layers/
│   ├── SOUL.core.template.md
│   ├── SOUL.daily.template.md
│   ├── SOUL.work.template.md
│   ├── SOUL.system_maintenance.template.md
│   ├── SOUL.intimacy.template.md
│   ├── SOUL.repair.template.md
│   ├── SOUL.conflict.template.md
│   ├── SOUL.creative.template.md
│   ├── SOUL.adult_boundary.template.md
│   └── SOUL.overlay_intimacy.template.md
├── scripts/
│   ├── orchestrator_probe.py
│   ├── validate_soul_layers.py
│   └── runtime_preview_plugin_probe.py
├── tests/
└── docs/
```

## 测试

推荐先跑轻量 orchestrator 测试，再决定是否安装 torch：

```bash
cd persona-engine
python3 -m venv .venv
.venv/bin/pip install -U pip
.venv/bin/pip install pytest pyyaml
PYTHONPATH=$PWD .venv/bin/pytest tests/ -q
```

如果需要完整情绪神经侧通道：

```bash
.venv/bin/pip install -r requirements.txt
PYTHONPATH=$PWD .venv/bin/pytest tests/ -q
```

常用 smoke checks：

```bash
python -m py_compile emotion_calculator.py emotion_detector.py emotion_state_manager.py legacy relationship-memory writer module persona_orchestrator/*.py scripts/*.py
python scripts/validate_soul_layers.py
python scripts/orchestrator_probe.py "[pet name]帮我看 gateway 日志" --score 1.0 --semantic-preview --semantic-backend local
```

## 集成建议

### Core identity ownership

本仓库支持两种 core source，用来适配不同部署方式：

| 模式 | 适用场景 | selected_layers | 责任边界 |
|---|---|---|---|
| `orchestrator_core` | 独立运行、插件演示、没有宿主人格系统的 Agent | `["core", mode, overlay?]` | 本仓库的 `SOUL.core.template.md` 提供身份锚点；mode/overlay 只调行为 |
| `host_core` | 已有核心人格文件、已有系统 prompt 管理器、接入多个 Agent runtime | `[mode, overlay?]` | 宿主 Agent 提供唯一 core identity；orchestrator 只管状态、模式、moments、modifier |

推荐默认保留 `orchestrator_core`，因为这样项目可以独立运行、测试和被其他 Agent 直接试用。对于已有完整人格系统的宿主，推荐使用 `host_core`，避免宿主 core 和 orchestrator core 同时注入导致 token 增长、身份漂移或规则冲突。

无论选择哪一种，都要满足三个约束：

1. 最终 prompt 中只能有一个 core identity。
2. Mode / Overlay layer 只能改变行为倾向，不能重新定义身份。
3. emotion_modifier 始终在最终 prompt 末尾，作为实时情绪和边界状态。

### Host integration contract

Persona Orchestrator 不要求宿主一定是 host agent。任何 Agent runtime 只要能提供以下接口，都可以接入：

- pre-LLM hook：在模型调用前读取或替换 system prompt。
- session state：保存 previous_mode、prompt_hash 和必要的审计元数据。
- emotion state：提供当前四维情绪和 emotion_score，或只使用本仓库的 EmotionStateManager。
- memory source：可选，提供 USER / MEMORY / STATE 等长期事实和当前状态；MOMENTS 域已退役。
- installation policy：决定 preview-only、active-limited 还是 active full。

这个边界是为了后续作为独立插件运行，或者接入其他 Agent 框架时，不需要把工具系统、记忆系统、平台适配层一起搬进来。

### legacy emotion-only integration

如果宿主 Agent 还没有接入 Persona Orchestrator，可以只使用情绪系统：

1. 用户消息到达后调用 `update_emotion_state(messages)`。
2. 回复生成前调用 `get_tone_modifiers()`。
3. 把 emotion_modifier 放到系统 prompt 的最末尾。
4. 每轮热替换旧 emotion_modifier，避免整个 prompt 频繁重建。

### orchestrator active integration

如果宿主 Agent 让 orchestrator 接管 prompt composition：

1. 先决定 core source：
   - `orchestrator_core`：宿主不要再加载自己的单体 core prompt。
   - `host_core`：宿主保留自己的 core prompt，orchestrator 不再选择 `core` layer。
2. 每轮情绪更新后，取最新 emotion_modifier。
3. 调用 `RuntimePreviewAdapter.analyze_runtime_turn(..., active=True)` 或 `StateOrchestrator.compose_active_prompt()`。
4. 只有在明确 active 模式下，宿主才安装 candidate_prompt。
5. 安装后更新 session cached prompt / session DB，避免下一轮回到旧 prompt。
6. 保证最终 prompt 只有一个 core identity、一个 `<emotion_modifier>`，且 emotion_modifier 位于末尾。

推荐上线顺序：

1. preview-only：只记录 StatePacket 和 prompt_hash。
2. active-limited：只开放 daily / work / system_maintenance。
3. active relationship：加入 intimacy / repair / conflict。
4. adult boundary layer：最后单独 gate，必须有额外 regression tests 和明确启用开关。

## 安全与隐私边界

- 不提交真实 STATE.md、USER.md、MEMORY.md。
- 不提交 API key、token、私有 endpoint、内部路径或真实聊天记录。
- 本仓库的模板使用占位符；用户应在本地填写自己的角色和关系内容。
- mode / overlay layer 只能改变行为倾向，不能重定义 core identity。
- 如果宿主已经提供 core identity，必须关闭 orchestrator core layer；如果使用 orchestrator core，宿主不能再重复注入自己的 core。
- sex_candidate 默认不是 active sex；危机、repair、work、system/public context 必须阻断性升级。
- preview logs 只保存 hash 和元数据，不保存完整用户消息。

## 常见误区

- `get_current_emotion_state()` 只返回四维状态，不包含 emotion_score；emotion_score 需要用 `EmotionCalculator.compute_emotion_score()` 计算，或从宿主 STATE frontmatter 单独传入 orchestrator。
- `get_tone_modifiers()` 在 state manager 中返回字符串；calculator 里的同名方法是底层数据结构，二者不要混用。
- public repo 测试通过不代表某个宿主 Agent 已使用这份代码；运行时路径和公开仓库可能不同。
- preview adapter 返回 candidate_prompt 不等于已经安装；是否安装必须由宿主 active 配置决定。
- plugin scaffold 写入文件不等于插件已启用；还需要配置和重启。
- quoted sensitive terms 应该走 meta_discussion，不应当因为出现“成人亲密”“讨厌你”等词就进入 sex/conflict。

## 文档

- `INTEGRATION.md`：宿主 Agent 集成方式。
- `docs/soul-state-orchestrator.md`：分层 SOUL 状态机设计和 probe。

## 许可证

MIT License

## 相关链接

- Chinese-Emotion-Small: https://huggingface.co/Johnson8187/Chinese-Emotion-Small
