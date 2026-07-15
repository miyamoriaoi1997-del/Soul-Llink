# 状态机识别率/精准度提升路线

生成时间：2026-04-30

## 当前状态

当前 persona 状态机以规则分类器为主，神经/模型打分主要处于 shadow/辅助观察状态。

### Active 决策链

`StateOrchestrator._analyze()` 当前 active packet 的 mode 来自：

1. `ModeClassifier.classify()` 规则分类
2. `TransitionManager.transition()` 根据 previous_mode、confidence、desire_tier、安全 flags 做切换
3. `_selected_layers()` 根据 active_mode 选择 SOUL layers

`semantic_shadow` 会被记录到 packet.reason 和 packet.semantic_shadow，但不会覆盖 active mode。

### 神经/语义接入点

- `SemanticModeClassifier`
  - backend=`local`：规则镜像，非神经
  - backend=`local_lightweight` / `rules+local`：规则 + 本地 SentimentAnalyzer
  - backend=`llm`：OpenAI-compatible LLM scaffold，但默认 disabled，且仍 shadow-only
- `SentimentAnalyzer`
  - 通过 `agent.sentiment_analyzer.SentimentAnalyzer.get_instance()` 加载
  - 输出 label / confidence / valence / inference_ms
  - 只在 semantic shadow 中用于 correction/boost/disagree 记录

### 当前日志观察

最近 runtime logs 中，semantic_shadow 出现较多，backend 为 `rules+local-lightweight`。
但 selected_layers/mode 仍由规则 + transition 决定。

## 当前神经模型参与程度

结论：中等偏低。

- 有本地轻量情绪/语气模型参与 shadow 观察。
- 有 sentiment -> mode hint 逻辑：angry/disgusted -> conflict，sad -> repair，caring -> intimacy。
- 有 confidence boost 和 disagreement logging。
- 但不会直接改 active mode，也不会直接改 selected_layers。
- LLM semantic classifier 只是 scaffold，默认关闭，且不会掌权。

## 提升路线

### Phase 1：评测集优先

先建立可复现 fixture，不先让模型掌权。

增加数据集字段：

- id
- message
- recent_context
- previous_mode
- emotion_score/desire_tier
- expected_mode
- expected_transition
- expected_layers
- acceptable_overlays
- forbidden_layers
- rationale

重点覆盖：

- daily vs intimacy
- intimacy vs sex_candidate
- repair vs conflict
- work vs system_maintenance
- meta discussion vs sensitive content
- short continuation hold
- high emotion + technical task
- crisis + sex request should repair not sex

### Phase 2：离线 shadow 对比

对每条 fixture 同时跑：

- rules result
- semantic_shadow result
- transition result
- final selected_layers

输出 confusion matrix：

- rule_correct / semantic_correct
- semantic_can_fix_rule_miss
- semantic_would_break_rule_match
- high-risk false positive
- low-confidence ambiguous

### Phase 3：有限仲裁，不直接全权交给神经模型

只允许 semantic 在低风险条件下影响 active：

1. 规则是 daily fallback 且 confidence < 0.65
2. semantic confidence >= 阈值
3. 目标 mode 不属于高风险：sex_candidate/conflict/crisis 不直接升格
4. 不覆盖 system_maintenance/work 的高置信命中
5. 所有覆盖必须写入 reason_codes 和日志

建议先允许：

- daily -> intimacy
- daily -> repair（非 crisis）
- daily -> work（如果语义强且无亲密/冲突风险）

暂不允许：

- semantic 直接进入 sex_candidate
- semantic 直接进入 conflict 后压过技术任务
- semantic 覆盖 system_maintenance

### Phase 4：transition 精准化

增加切换层面的测试与参数：

- mode inertia by mode pair
- short-message hold only for allowed previous modes
- relationship/daily 从 technical mode 退出的条件更细
- conflict hold 的解除条件
- repair 后回 daily/intimacy 的冷却
- sex_candidate 的进入、退出、aftercare overlay 条件

### Phase 5：生产 shadow A/B

在 active 不改变的情况下跑一段时间：

- 每轮记录 rules vs semantic vs expected/manual label
- 用户可用命令标记误判样本
- 定期汇总 top error patterns

再决定是否打开 limited arbitration。

## 推荐下一步

不要直接把神经模型接管 active。

下一步应该做：

1. 新增 `tests/fixtures/state_machine_eval_cases.jsonl`
2. 新增 `scripts/eval_state_machine.py`
3. 输出准确率、混淆矩阵、规则/语义差异
4. 根据误判样本补规则和 transition
5. 再设计 semantic limited arbitration
