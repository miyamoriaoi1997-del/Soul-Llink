# SoulLink WebUI 监控

只读、本机旁路监控服务。默认地址：`http://127.0.0.1:8765/`。

## 启动

```bash
soullink webui --no-open-browser
soullink webui --host 127.0.0.1 --port 8765
```

可用参数：`--db`、`--memfs`、`--config`、`--persona-log`、`--router-log`、`--state-path`、`--mode`、`--memory-body-limit`、`--refresh-seconds`。

## Windows 登录自启

仓库内提供可回滚的当前用户登录任务管理脚本：

```powershell
# 注册登录自启
powershell -NoProfile -ExecutionPolicy Bypass -File scripts/windows/manage-webui-autostart.ps1 install

# 查看状态
powershell -NoProfile -ExecutionPolicy Bypass -File scripts/windows/manage-webui-autostart.ps1 status

# 移除自启（回滚）
powershell -NoProfile -ExecutionPolicy Bypass -File scripts/windows/manage-webui-autostart.ps1 remove
```

任务名为 `SoulLink WebUI`，以当前用户的交互式登录会话、普通权限运行。启动脚本先检查 `GET /api/v1/health`，已运行时直接退出，避免重复实例；服务仍只绑定 `127.0.0.1:8765`。日志位于 Hermes 本机日志目录的 `soullink-webui.stdout.log` 与 `soullink-webui.stderr.log`。

## 私有监控边界

- 仅绑定 loopback IP，拒绝 `0.0.0.0` 和外部地址。
- API 只接受 GET；POST/PUT/PATCH/DELETE 返回 405。
- 这是单机私有监控台，默认直接展示本机情绪值、记忆正文与注入块内容，不设置正文遮罩或二次解锁。
- 注入块必须标明证据类型：宿主精确捕获或 `sidecar_reconstruction_preview` 旁路重建；旁路结果不得冒充已发送内容。
- `--mode` 只限定旁路预览的记忆检索 scope，不是状态机输出；实时 mode 只读取 `exact_host_capture`。
- Collector 强制使用 `fix=False`、`rebuild=False`、`dry_run=True`、`rebuild_indexes=False`。
- WebUI 不执行批准、删除、修复、模式切换或索引重建。

## 证据语义

WebUI 不把“能推测”写成“已观察”。主要状态含义：

| 状态 | 含义 |
|---|---|
| `exact_host_capture` | 在受支持的宿主边界捕获，可归属于具体宿主轮次。 |
| `sidecar_reconstruction_preview` | 根据旁路可用状态只读重建的诊断预览；不是最终模型输入证据。 |
| `unavailable` / `not observed` | 没有足够证据，界面保留缺口，不从最终回答或邻近状态反推。 |
| `stale` | 观察真实存在，但超过新鲜度阈值。 |

首页把当前姿态、决策权威、语义融合、记忆影响路径、情绪状态、状态轨迹、判断证据、上下文构成与事实底座放在同一证据面。`ARCHIVE → RECALL → GOVERN → FORWARD` 只连接同一轮真实捕获的阶段；其中任一阶段未捕获时，就显示未捕获。

## 公开截图

私有生产页面会直接显示本机情绪/关系值、记忆规模、注入正文、宿主轮次、时间戳和 token 遥测。公开截图前必须逐项审查，不能因为服务只绑定 loopback 就把截图视为无隐私。

仓库的 `docs/assets/soullink-observatory-demo.png` 来自生产提供的 2.2 前端，但在浏览器内先替换为明确的合成 demo 值，并只保留首页首屏结构。图片不包含生产轮次、时间、情绪、关系、记忆、token、对话或宿主数据。

## API

- `GET /api/v1/health`
- `GET /api/v1/snapshot`
- `GET /api/v1/issues`
- `GET /`

## 数据新鲜度

配置值标记为 `config`；具有 UTC `observed_at` 的最近上下文遥测标记为 `config+last_telemetry`。超过陈旧阈值时显示 stale，不把旧值伪装成实时值。

## 故障排查

- 端口占用：修改 `--port`。
- DB 锁或 collector 错误：相关板块降级并保留其他数据。
- 日志缺失或损坏尾行：显示 warning，不导致服务退出。
- 静态资源缺失：确认 wheel 中包含 `pcltm/monitoring/static/*`。

## 停止与回滚

前台运行时按 Ctrl+C。WebUI 不迁移 schema；回滚只需停止服务并恢复此前备份的源码/安装包。生产同步前仍需完整备份、SHA-256 校验与恢复演练。
