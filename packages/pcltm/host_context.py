from __future__ import annotations

from collections.abc import Callable


_EXPLICIT_SEX_TERMS = ("做爱", "性交", "性器", "高潮", "舔下面", "成人内容", "sex")
_WORK_TERMS = (
    "soullink", "pcltm", "hermes", "github", "git", "commit", "push", "main", "token",
    "仓库", "代码", "测试", "验证", "审计", "运行", "配置", "数据库", "日志", "模型",
    "状态机", "情绪值", "注入", "修复", "优化", "生产", "目录", "文件", "分支", "全绿",
    "工作", "互相干扰", "干扰", "切换", "mode", "memory-context", "检索",
)
_DAILY_TERMS = ("我爱你", "喜欢你", "爱你", "亲", "抱", "摸", "揉", "想你", "小凛", "凛", "老婆")


def conservative_mode_hint(query: str) -> str | None:
    """Return a conservative recall hint; active persona routing stays authoritative."""
    text = (query or "").strip().lower()
    if not text:
        return None
    if any(term in text for term in _EXPLICIT_SEX_TERMS):
        return "sex"
    if any(term in text for term in _WORK_TERMS):
        return "work"
    if any(term in text for term in _DAILY_TERMS):
        return "daily"
    return None


class PCLTMContextPort:
    """Host-neutral prefetch port around the governed PCLTM prompt-context loader."""

    def __init__(self, *, loader: Callable[..., str]) -> None:
        self._loader = loader

    def prefetch(
        self,
        query: str,
        *,
        active_mode: str | None = None,
        session_id: str | None = None,
        continuity_evidence: object | None = None,
    ) -> str:
        """Load memory with the state machine mode when the host provides it."""
        mode = active_mode if active_mode in {"daily", "work", "sex"} else conservative_mode_hint(query)
        kwargs = {"mode": mode, "query": query}
        if session_id is not None or continuity_evidence is not None:
            kwargs.update(session_id=session_id, continuity_evidence=continuity_evidence)
        return self._loader(**kwargs) or ""
