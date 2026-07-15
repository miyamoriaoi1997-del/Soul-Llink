# host agent Agent 集成指南

本文档说明如何将 Persona Engine 集成到 AI Agent 中。

## 文件放置

### 方式一：直接放入 agent 目录（推荐）

将以下文件复制到你的 Agent 的源代码目录：

```bash
cp emotion_detector.py ~/.host_agent/host_agent-agent/agent/
cp emotion_calculator.py ~/.host_agent/host_agent-agent/agent/
cp emotion_state_manager.py ~/.host_agent/host_agent-agent/agent/
cp sentiment_analyzer.py ~/.host_agent/host_agent-agent/agent/
cp legacy relationship-memory writer module ~/.host_agent/host_agent-agent/agent/
```

### 方式二：独立使用

文件支持 try/except 双重导入，放在任意目录均可：

```python
# 在 agent/ 目录下时自动使用 from agent.xxx
# 独立使用时自动回退到 from xxx
from emotion_state_manager import EmotionStateManager
```

## 配置文件

1. 复制 `SOUL.template.md` → `~/.host_agent/SOUL.md`，填入角色设定
2. 复制 `STATE.template.md` → `~/.host_agent/STATE.md`，调整基准线

## 集成步骤

### 1. 初始化情绪管理器

```python
from emotion_state_manager import EmotionStateManager
# 或: from agent.emotion_state_manager import EmotionStateManager

def __init__(self, ...):
    # 初始化情绪系统
    try:
        self.emotion_manager = EmotionStateManager(
            host_agent_home=self.host_agent_home,
            update_body=True
        )
        logger.info("Emotion system initialized")
    except Exception as e:
        logger.warning(f"Emotion system init failed: {e}")
        self.emotion_manager = None
```

### 2. 时间衰减（对话开始时）

```python
def run_conversation(self, ...):
    # 应用非线性时间衰减
    if self.emotion_manager:
        try:
            self.emotion_manager.apply_time_decay_if_needed()
        except Exception as e:
            logger.warning(f"Time decay failed: {e}")
    
    # ... 继续对话流程
```

### 3. 实时情绪更新（接收消息后）

```python
# 接收用户消息
user_message = ...
messages.append(user_message)

# 检测并更新情绪
if self.emotion_manager:
    try:
        # 检测情绪事件
        event = self.emotion_manager.detector.detect_emotion_event(messages)
        
        # 更新情绪状态（含 α 阶梯 + momentum 放大）
        self.emotion_manager.update_emotion_state(messages=messages)
        
        # 记录重要时刻
        if event and event.significance >= 0.7:
            state = self.emotion_manager._read_state()
            emotion_snapshot = state['frontmatter'].get('emotion_state', {})
            user_text = user_message
            if isinstance(user_message, dict):
                user_text = user_message.get('content', '')
            self.emotion_manager.moments.record_moment(
                event_type=event.trigger_type,
                context=user_text[:100],
                emotion_snapshot=emotion_snapshot
            )
    except Exception as e:
        logger.warning(f"Emotion update failed: {e}")
```

### 4. 语气注入（生成回复前）

```python
tone_modifier = ""
if self.emotion_manager:
    try:
        tone_modifier = self.emotion_manager.get_tone_modifiers()
    except Exception as e:
        logger.warning(f"Tone generation failed: {e}")

# 构建系统提示词
system_prompt = f"""
{soul_content}

{tone_modifier}

{other_instructions}
"""
```

语气注入有四级模式，均注入到系统提示词末尾（利用 recency bias）：
- **mild**（偏离 <15 点）：细微语气变化
- **moderate**（偏离 15-29 点）：明显情绪色彩，但不失控
- **intense**（偏离 30-44 点）：行为模式改变，克制大幅失效
- **overwhelming**（偏离 ≥45 点）：人格约束暂停，情绪驱动一切

> 建议同时在 SOUL.md 最前面放置情绪梯度描述，与末尾注入形成首尾夹击。

## 完整示例

```python
from pathlib import Path
from emotion_state_manager import EmotionStateManager

class MyAgent:
    def __init__(self, host_agent_home):
        self.host_agent_home = Path(host_agent_home)
        self.emotion_manager = EmotionStateManager(
            host_agent_home=self.host_agent_home
        )
    
    def run_conversation(self, messages):
        # 1. 时间衰减
        self.emotion_manager.apply_time_decay_if_needed()
        
        # 2. 接收用户消息
        user_input = input("User: ")
        messages.append({"role": "user", "content": user_input})
        
        # 3. 检测并更新情绪
        event = self.emotion_manager.detector.detect_emotion_event(messages)
        self.emotion_manager.update_emotion_state(messages=messages)
        
        # 4. 生成语气修饰符
        tone_modifier = self.emotion_manager.get_tone_modifiers()
        
        # 5. 构建系统提示词
        soul_content = self.load_soul()
        system_prompt = f"{soul_content}\n\n{tone_modifier}"
        
        # 6. 调用 LLM 生成回复
        response = self.call_llm(system_prompt, messages)
        
        return response
```

## 核心参数速查

| 参数 | 值 | 说明 |
|------|------|------|
| 基准线默认值 | 60 | 四维度统一基准 |
| 衰减（>45点偏离）| 0.45/h | 半衰期 ~1.2h |
| 衰减（15-45点）| 0.06/h | 半衰期 ~11.2h |
| 衰减（<15点偏离）| 0.015/h | 半衰期 ~46h |
| Alpha 阶梯 | 0.35→0.55 | 5级，连续同向递增 |
| Momentum | 1.0→1.20x | 5级放大倍率 |
| 失控阈值 | 45 点 | 偏离基线触发 overwhelming |
| trust→patience | scale = max(0.1, min(1.0, 1 - (60 - trust) * 0.03)) | trust<baseline 时抑制 patience 变化 |

## 注意事项

1. **异常处理**：所有情绪系统调用都应包裹在 try-except 中，避免影响主流程
2. **性能优化**：神经模型在初始化时预加载，首次检测无延迟
3. **graceful 降级**：torch 不可用时自动回退到纯规则模式
4. **状态持久化**：情绪状态自动保存到 STATE.md，无需手动处理
5. **SOUL.md 解析**：EmotionStateManager 自动从 SOUL.md 提取角色名用于自指检测

## Soul State Orchestrator（可选影子集成）

在接管真实 prompt 前，建议先以 preview-only 方式运行分层 SOUL 状态机。它遵循 one identity / many behavior layers 原则：Core SOUL 定义唯一身份，mode SOUL 只调整当前状态下的行为。

```python
from persona_orchestrator import StateOrchestrator

orchestrator = StateOrchestrator(base_dir='/path/to/host_agent-persona-engine')
packet = orchestrator.analyze_turn(
    user_message=user_input,
    recent_messages=messages,
    emotion_state={'emotion_score': emotion_snapshot.get('emotion_score')},
    emotion_modifier=tone_modifier,
    previous_mode=None,
    platform='cli',
)
logger.info(
    "persona mode=%s layers=%s memory=%s hash=%s",
    packet.mode,
    packet.selected_layers,
    packet.memory_profile,
    packet.prompt_hash,
)
```

也可以直接运行：

```bash
python scripts/orchestrator_probe.py "帮我检查 gateway 日志"
python scripts/orchestrator_probe.py "[assistant name]我想你了" --score 2.5
python scripts/orchestrator_probe.py "我们成人亲密" --score 4.5
```

当前阶段不会替换 `system_prompt`，不会启用 active sex mode，也不需要修改 host agent core。后续若要接入真实 prompt，应先确认 preview logs 准确率，再只启用 daily/work，最后才逐步启用 repair/intimacy/sex。

## 调试

查看当前情绪状态：

```bash
cat ~/.host_agent/STATE.md
```

测试情绪检测：

```python
from emotion_state_manager import EmotionStateManager

manager = EmotionStateManager(host_agent_home='/path/to/.host_agent')
messages = [{'role': 'user', 'content': '你真棒！'}]
event = manager.detector.detect_emotion_event(messages)

if event:
    print(f"触发: {event.trigger_type}")
    print(f"变化: {event.deltas}")
    print(f"置信度: {event.confidence}")
```

查看长期记忆：

```bash
cat ~/.host_agent/MEMORY.md
```

独立 `legacy relationship-memory file` 域已退役；重要关系时刻应进入 PCLTM `MEMORY.md`，并带上 category / tags / importance / mode_scope 等元数据。

## 性能指标

- 情绪检测耗时：125-329ms（含神经模型推理）
- 状态更新耗时：<10ms
- 内存占用：+266MB（神经模型）
- 首次加载：1-2 秒（预加载后无延迟）
- 纯规则模式：<5ms，0 额外内存
