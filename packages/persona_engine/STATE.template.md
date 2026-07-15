---
controller:
  enabled: true
  platforms:
  - cli
  - telegram
  title: 当前状态
emotion_state:
  affection: 60
  trust: 60
  possessiveness: 60
  patience: 60
  emotion_score: 0.0
  current_emotion: 0.0
  last_update: '2026-01-01T00:00:00.000000'
  baselines:
    affection: 60
    trust: 60
    possessiveness: 60
    patience: 60
  decay_rate: 2.0
  inertia:
    consecutive_same: 0
    last_direction: 0
    history: []
---

## 当前情绪状态

好感度: 60/120 (基准)
信任度: 60/120 (基准)
占有欲: 60/120 (基准)
耐心值: 60/120 (基准)
情绪分值: 0.00 / 5.00

最近触发: 无
语气倾向: 正常

---

**配置说明：**

1. 基准线（baselines）：四维度默认均为 60，可根据角色性格调整
   - 高傲冷淡角色：降低 affection 基准（如 40）
   - 热情开朗角色：提高 affection 基准（如 80）
   - 防备心强角色：降低 trust 基准（如 30）
   - 占有欲强角色：提高 possessiveness 基准（如 80）

2. 范围：所有维度的有效范围为 0-120

3. 衰减系统会自动根据偏离量选择速率：
   - <15 点偏离：0.35 点/小时（快速恢复）
   - 15-45 点偏离：0.12 点/小时（正常衰减）
   - >45 点偏离：0.03 点/小时（缓慢衰减）

4. inertia 字段由系统自动维护，无需手动修改
