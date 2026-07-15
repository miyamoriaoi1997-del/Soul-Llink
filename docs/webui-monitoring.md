# SoulLink WebUI 监控

只读、本机旁路监控服务。默认地址：`http://127.0.0.1:8765/`。

## 启动

```bash
soullink webui --no-open-browser
soullink webui --host 127.0.0.1 --port 8765
```

可用参数：`--db`、`--memfs`、`--config`、`--persona-log`、`--router-log`、`--state-path`、`--mode`、`--memory-body-limit`、`--refresh-seconds`。

## 私有监控边界

- 仅绑定 loopback IP，拒绝 `0.0.0.0` 和外部地址。
- API 只接受 GET；POST/PUT/PATCH/DELETE 返回 405。
- 这是单机私有监控台，默认直接展示本机情绪值、记忆正文与注入块内容，不设置正文遮罩或二次解锁。
- 注入块必须标明证据类型：宿主精确捕获或 `sidecar_reconstruction_preview` 旁路重建；旁路结果不得冒充已发送内容。
- Collector 强制使用 `fix=False`、`rebuild=False`、`dry_run=True`、`rebuild_indexes=False`。
- WebUI 不执行批准、删除、修复、模式切换或索引重建。

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
