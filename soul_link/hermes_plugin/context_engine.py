"""PCLTM-context engine for Hermes.

This bridge restores the Soul-Link PCLTM-context compression engine as a
Hermes context-engine plugin. It intentionally keeps the native Hermes
compressor out of the main path; the engine is selected via
``context.engine = pcltm-context`` and delegates the actual shadow context
construction to the Soul-Link package.
"""

from __future__ import annotations

import ast
import hashlib
import json
import logging
import os
import sqlite3
import sys
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Mapping, Sequence

try:
    from agent.context_engine import ContextEngine
    try:
        from agent.context_compressor import SUMMARY_PREFIX
    except ModuleNotFoundError:
        SUMMARY_PREFIX = "Previous conversation summary:"
    try:
        from agent.model_metadata import estimate_messages_tokens_rough, get_model_context_length
    except ModuleNotFoundError:
        def estimate_messages_tokens_rough(messages: list[dict[str, Any]]) -> int:
            return sum(len(str(message.get("content", ""))) for message in messages) // 4

        def get_model_context_length(model: str | None = None) -> int:
            return 128000
except ModuleNotFoundError as exc:  # pragma: no cover - host-neutral tests
    if exc.name != "agent":
        raise

    class ContextEngine:  # type: ignore[no-redef]
        def __init__(self, config: Any | None = None) -> None:
            self.config = config or {}

        def get_status(self) -> dict[str, Any]:
            return {"engine": self.__class__.__name__}

    SUMMARY_PREFIX = "Previous conversation summary:"

    def estimate_messages_tokens_rough(messages: list[dict[str, Any]]) -> int:
        return sum(len(str(message.get("content", ""))) for message in messages) // 4

    def get_model_context_length(model: str | None = None) -> int:
        return 128000


def import_pcltm_module(module_name: str):
    """Import SoulLink modules without relying on Hermes-private shims."""
    import importlib

    return importlib.import_module(f"pcltm.{module_name}")


def import_pcltm_memory_adapter():
    import importlib

    return importlib.import_module("pcltm.memory_adapter")

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class ActiveFrameGovernanceSettings:
    """Typed PCLTM active-frame budget and eviction controls.

    This is the PCLTM analogue of Letta's ``CompactionSettings``: policy is
    explicit and agent/config driven, but older transcript material is archived
    as typed capsules instead of being reinserted as an always-on summary
    sidecar in the hot prompt.
    """

    active_frame_budget: int = 0
    core_budget: int = 0
    pinned_budget: int = 0
    episodic_budget: int = 0
    current_task_budget: int = 0
    continuity_budget: int = 0
    # Keep a small natural conversation window in the active continuity capsule:
    # enough real user/assistant turns to preserve conversational inertia, but
    # still bounded so archived dialogue cannot overrule the latest request.
    continuity_turns: int = 5
    continuity_min_dialogue_turns: int = 3
    continuity_max_dialogue_turns: int = 5
    tool_evidence_budget: int = 0
    tool_result_capsule_threshold_chars: int = 2400
    recall_budget: int = 0
    partial_evict_percentage: float = 0.30
    eviction_policy: str = "partial_archival_capsule"
    max_archival_capsule_tokens: int = 700

    @classmethod
    def from_mapping(
        cls,
        config: Mapping[str, Any] | None,
        previous: "ActiveFrameGovernanceSettings | None" = None,
    ) -> "ActiveFrameGovernanceSettings":
        base = previous or cls()
        if not isinstance(config, Mapping):
            return base
        return cls(
            active_frame_budget=_positive_int(
                config.get("active_frame_budget")
                or config.get("active_frame_budget_tokens")
                or config.get("message_budget_tokens")
                or config.get("hot_path_budget_tokens"),
                base.active_frame_budget,
            ),
            core_budget=_positive_int(config.get("core_budget") or config.get("core_budget_tokens"), base.core_budget),
            pinned_budget=_positive_int(config.get("pinned_budget") or config.get("pinned_budget_tokens"), base.pinned_budget),
            episodic_budget=_positive_int(config.get("episodic_budget") or config.get("episodic_budget_tokens"), base.episodic_budget),
            current_task_budget=_positive_int(
                config.get("current_task_budget") or config.get("current_task_budget_tokens"),
                base.current_task_budget,
            ),
            continuity_budget=_positive_int(
                config.get("continuity_budget") or config.get("continuity_budget_tokens"),
                base.continuity_budget,
            ),
            continuity_turns=_positive_int(
                config.get("continuity_turns") or config.get("continuity_turn_count"),
                base.continuity_turns,
            ),
            continuity_min_dialogue_turns=_positive_int(
                config.get("continuity_min_dialogue_turns")
                or config.get("continuity_min_real_dialogue_turns"),
                base.continuity_min_dialogue_turns,
            ),
            continuity_max_dialogue_turns=_positive_int(
                config.get("continuity_max_dialogue_turns")
                or config.get("continuity_max_real_dialogue_turns"),
                base.continuity_max_dialogue_turns,
            ),
            tool_evidence_budget=_positive_int(
                config.get("tool_evidence_budget") or config.get("tool_evidence_budget_tokens"),
                base.tool_evidence_budget,
            ),
            tool_result_capsule_threshold_chars=_positive_int(
                config.get("tool_result_capsule_threshold_chars"),
                base.tool_result_capsule_threshold_chars,
            ),
            recall_budget=_positive_int(config.get("recall_budget") or config.get("recall_budget_tokens"), base.recall_budget),
            partial_evict_percentage=_percentage(
                config.get("partial_evict_percentage")
                or config.get("partial_evict_summarizer_percentage")
                or config.get("partial_evict_archival_percentage"),
                base.partial_evict_percentage,
            ),
            eviction_policy=str(config.get("eviction_policy") or base.eviction_policy or "partial_archival_capsule"),
            max_archival_capsule_tokens=_positive_int(
                config.get("max_archival_capsule_tokens"),
                base.max_archival_capsule_tokens,
            ),
        )

    def as_dict(self) -> Dict[str, Any]:
        return {
            "active_frame_budget": self.active_frame_budget,
            "core_budget": self.core_budget,
            "pinned_budget": self.pinned_budget,
            "episodic_budget": self.episodic_budget,
            "current_task_budget": self.current_task_budget,
            "continuity_budget": self.continuity_budget,
            "continuity_turns": self.continuity_turns,
            "continuity_min_dialogue_turns": self.continuity_min_dialogue_turns,
            "continuity_max_dialogue_turns": self.continuity_max_dialogue_turns,
            "tool_evidence_budget": self.tool_evidence_budget,
            "tool_result_capsule_threshold_chars": self.tool_result_capsule_threshold_chars,
            "recall_budget": self.recall_budget,
            "partial_evict_percentage": self.partial_evict_percentage,
            "eviction_policy": self.eviction_policy,
            "max_archival_capsule_tokens": self.max_archival_capsule_tokens,
        }


def _positive_int(value: Any, default: int = 0) -> int:
    try:
        parsed = int(value or 0)
    except Exception:
        return default
    return parsed if parsed > 0 else default


def _percentage(value: Any, default: float = 0.30) -> float:
    try:
        parsed = float(value)
    except Exception:
        parsed = default
    if parsed <= 0:
        return default
    return min(1.0, max(0.01, parsed))

def _candidate_soul_link_package_roots() -> List[Path]:
    """Return candidate Soul-Link package roots for both source and installed plugins.

    The same adapter file can run from two legitimate locations:
    - <repo>/adapters/hermes/context_engine/pcltm-context/ (source)
    - <hermes-root>/plugins/context_engine/pcltm-context/ (installed artifact)

    The old implementation assumed the source layout only and derived
    ``parents[4] / "packages"``.  In the installed Hermes layout that becomes
    ``<hermes-home>/packages`` and silently loses the real Soul-Link package
    root.  Keep this resolver local to the adapter so production does not rely
    on gateway-level PYTHONPATH injection.
    """
    here = Path(__file__).resolve()
    candidates: List[Path] = []

    for env_name in ("SOUL_LINK_PACKAGES", "HERMES_SOUL_LINK_PACKAGES"):
        raw = os.environ.get(env_name)
        if raw:
            candidates.append(Path(raw).expanduser())

    for parent in here.parents:
        candidates.append(parent / "packages")
        candidates.append(parent / "soul-link" / "packages")

    seen: set[str] = set()
    unique: List[Path] = []
    for candidate in candidates:
        resolved = candidate.resolve()
        key = str(resolved)
        if key in seen:
            continue
        seen.add(key)
        unique.append(resolved)
    return unique


def _install_soul_link_packages_path() -> Path | None:
    for packages_root in _candidate_soul_link_package_roots():
        if (packages_root / "pcltm").is_dir():
            text = str(packages_root)
            if text not in sys.path:
                sys.path.insert(0, text)
            return packages_root
    return None


_SOUL_LINK_PACKAGES = _install_soul_link_packages_path()

try:  # imported lazily enough for plugin discovery, but fail closed if absent
    _context_engine_module = import_pcltm_module("context_engine")
    PCLTMContextEngine = _context_engine_module.PCLTMContextEngine
    is_compaction_handoff = _context_engine_module.is_compaction_handoff
    is_runtime_control_message = _context_engine_module.is_runtime_control_message
    runtime_visible_user_text = _context_engine_module.runtime_visible_user_text
    sanitize_tool_chain = _context_engine_module.sanitize_tool_chain
except Exception as exc:  # pragma: no cover - availability checked separately
    PCLTMContextEngine = None  # type: ignore[assignment]
    is_compaction_handoff = None  # type: ignore[assignment]
    is_runtime_control_message = None  # type: ignore[assignment]
    runtime_visible_user_text = None  # type: ignore[assignment]
    sanitize_tool_chain = None  # type: ignore[assignment]
    _IMPORT_ERROR = exc
else:
    _IMPORT_ERROR = None


class PCLTMContextCompressionEngine(ContextEngine):
    """Hermes ContextEngine backed solely by PCLTM-context."""

    def __init__(
        self,
        *,
        model: str = "",
        threshold_percent: float = 0.50,
        protect_first_n: int = 0,
        protect_last_n: int = 0,
        target_ratio: float = 0.20,
        quiet_mode: bool = True,
    ) -> None:
        if _IMPORT_ERROR is not None:
            raise RuntimeError(f"PCLTM-context unavailable: {_IMPORT_ERROR}")
        self.model = model or ""
        self.threshold_percent = threshold_percent

        # Kept as inert compatibility attributes for status/config callers;
        # PCLTM active-frame assembly never uses fixed head/tail retention.
        self.protect_first_n = 0
        self.protect_last_n = 0
        self.target_ratio = target_ratio
        self.quiet_mode = quiet_mode
        self.context_length = get_model_context_length(self.model or "unknown")
        self._configured_budget_tokens = 0
        self._configured_message_budget_tokens = 0
        self._request_budget_safety_margin_tokens = 512
        self.threshold_tokens = int(self.context_length * self.threshold_percent)
        self.last_prompt_tokens = 0
        self.last_completion_tokens = 0
        self.last_total_tokens = 0
        self.compression_count = 0
        self.session_id = ""
        self._hermes_home = Path(os.getenv("HERMES_HOME") or Path.home() / ".hermes")
        self._last_context_render = ""
        self._last_continuity_tokens = 0
        self._last_continuity_message_count = 0
        self._last_continuity_omitted_count = 0
        self._last_continuity_summary = ""
        self._last_dropped_tool_results = 0
        self._last_ignored_handoffs = 0
        self._task_generation = 0
        self._last_active_user_request = ""
        self._last_tool_capsules = 0
        self._last_dropped_tool_chars = 0
        self._last_message_budget_tokens = 0
        self._last_tail_tokens = 0
        self._last_overweight_reason = ""
        self._last_budget_breakdown: Dict[str, int] = {}
        self._last_budget_total_tokens = 0
        self._last_budget_limit_tokens = 0
        self._last_fail_closed_reason = ""
        self.strict_fail_closed = True
        self._evidence_capsules_written: set[str] = set()
        self.governance = ActiveFrameGovernanceSettings()
        self._last_archival_capsules_written = 0
        self._last_archival_messages_considered = 0
        self._last_archival_messages_evicted = 0
        self._last_archival_eviction_policy = self.governance.eviction_policy

    @property
    def name(self) -> str:
        return "pcltm-context"

    def configure(self, config: Mapping[str, Any] | None = None) -> None:
        """Apply Hermes ``context`` config to the plugin instance.

        The host constructs context engines before model metadata is finalized;
        this hook lets the runtime pass Letta-style hot-path budget controls
        without changing plugin discovery semantics.
        """
        if not isinstance(config, Mapping):
            return
        # Letta/MemGPT-style active frames do not honor legacy fixed head/tail
        # retention knobs.  Older Hermes configs may still contain
        # ``protect_first_n`` / ``protect_last_n`` for the native compressor;
        # this engine intentionally ignores them so archival/recall memory, not
        # raw transcript pinning, is the source of older context.
        unified_budget = _positive_int(
            config.get("budget_tokens")
            or config.get("total_budget_tokens")
            or config.get("context_budget_tokens"),
            0,
        )
        self._configured_budget_tokens = unified_budget

        self.governance = ActiveFrameGovernanceSettings.from_mapping(config, self.governance)
        derived_active_frame_budget = 0
        if unified_budget > 0:
            derived_active_frame_budget = max(256, int(unified_budget * 0.70))
            derived_continuity_budget = max(256, int(unified_budget * 0.05))
            derived_tool_evidence_budget = max(256, int(unified_budget * 0.05))
            derived_recall_budget = max(256, int(unified_budget * 0.08))
            self.governance = ActiveFrameGovernanceSettings(
                active_frame_budget=self.governance.active_frame_budget or derived_active_frame_budget,
                core_budget=self.governance.core_budget,
                pinned_budget=self.governance.pinned_budget,
                episodic_budget=self.governance.episodic_budget,
                current_task_budget=self.governance.current_task_budget,
                continuity_budget=self.governance.continuity_budget or derived_continuity_budget,
                continuity_turns=self.governance.continuity_turns,
                continuity_min_dialogue_turns=self.governance.continuity_min_dialogue_turns,
                continuity_max_dialogue_turns=self.governance.continuity_max_dialogue_turns,
                tool_evidence_budget=self.governance.tool_evidence_budget or derived_tool_evidence_budget,
                tool_result_capsule_threshold_chars=self.governance.tool_result_capsule_threshold_chars,
                recall_budget=self.governance.recall_budget or derived_recall_budget,
                partial_evict_percentage=self.governance.partial_evict_percentage,
                eviction_policy=self.governance.eviction_policy,
                max_archival_capsule_tokens=self.governance.max_archival_capsule_tokens,
            )

        budget = (
            config.get("message_budget_tokens")
            or config.get("hot_path_budget_tokens")
            or self.governance.active_frame_budget
            or derived_active_frame_budget
        )
        if budget is not None:
            try:
                self._configured_message_budget_tokens = max(256, int(budget))
            except (TypeError, ValueError):
                self._configured_message_budget_tokens = 0
        if unified_budget > 0:
            self.threshold_tokens = unified_budget
        if "strict_fail_closed" in config:
            self.strict_fail_closed = _coerce_bool(config.get("strict_fail_closed"), default=True)
        if "request_budget_safety_margin_tokens" in config:
            try:
                self._request_budget_safety_margin_tokens = max(
                    0,
                    int(config.get("request_budget_safety_margin_tokens") or 0),
                )
            except (TypeError, ValueError):
                self._request_budget_safety_margin_tokens = 512
        elif unified_budget > 0:
            self._request_budget_safety_margin_tokens = min(8192, max(1024, int(unified_budget * 0.03)))

    def is_available(self) -> bool:
        return _IMPORT_ERROR is None

    def update_model(
        self,
        model: str,
        context_length: int,
        base_url: str = "",
        api_key: str = "",
        provider: str = "",
        api_mode: str = "",
    ) -> None:
        self.model = model or self.model
        if context_length and context_length > 0:
            self.context_length = context_length
        else:
            self.context_length = get_model_context_length(self.model or "unknown")
        if self._configured_budget_tokens > 0:
            self.threshold_tokens = self._configured_budget_tokens
        else:
            self.threshold_tokens = int(self.context_length * self.threshold_percent)

    def update_from_response(self, usage: Dict[str, Any]) -> None:
        self.last_prompt_tokens = int(usage.get("prompt_tokens") or usage.get("input_tokens") or 0)
        self.last_completion_tokens = int(usage.get("completion_tokens") or usage.get("output_tokens") or 0)
        self.last_total_tokens = int(
            usage.get("total_tokens") or (self.last_prompt_tokens + self.last_completion_tokens)
        )
        payload = {
            "source": "exact_host_context_usage",
            "observed_at": datetime.now(timezone.utc).isoformat(),
            "session_id": self.session_id,
            "engine": self.name,
            "prompt_tokens": self.last_prompt_tokens,
            "completion_tokens": self.last_completion_tokens,
            "total_tokens": self.last_total_tokens,
            "input_tokens": int(usage.get("input_tokens") or self.last_prompt_tokens),
            "output_tokens": int(usage.get("output_tokens") or self.last_completion_tokens),
            "cache_read_tokens": int(usage.get("cache_read_tokens") or 0),
            "cache_write_tokens": int(usage.get("cache_write_tokens") or 0),
            "reasoning_tokens": int(usage.get("reasoning_tokens") or 0),
            "context_length": int(self.context_length or 0),
            "budget_tokens": int(self._configured_budget_tokens or self.threshold_tokens or 0),
            "compression_count": int(self.compression_count or 0),
            "last_budget_breakdown": dict(self._last_budget_breakdown),
            "last_budget_total_tokens": int(self._last_budget_total_tokens or 0),
            "last_budget_limit_tokens": int(self._last_budget_limit_tokens or 0),
        }
        path = self._hermes_home / "runtime" / "soullink-context-telemetry.json"
        try:
            path.parent.mkdir(parents=True, exist_ok=True)
            temp = path.with_name(path.name + f".{os.getpid()}.tmp")
            temp.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
            os.replace(temp, path)
        except OSError:
            logger.debug("Failed to write SoulLink context usage telemetry", exc_info=True)

    def should_compress(self, prompt_tokens: int = None) -> bool:
        tokens = int(prompt_tokens or self.last_prompt_tokens or 0)
        return bool(tokens and tokens >= self.threshold_tokens)

    def should_compress_preflight(self, messages: List[Dict[str, Any]]) -> bool:
        if not self.has_content_to_compress(messages):
            return False
        tokens = estimate_messages_tokens_rough(messages)
        threshold = int(self.threshold_tokens or 0)
        if threshold <= 0:
            threshold = 16_000
        return tokens >= threshold

    def has_content_to_compress(self, messages: List[Dict[str, Any]]) -> bool:
        latest = self._latest_real_user_index(messages)
        if latest is None:
            return False
        # Rebuild whenever there is archival material before the current turn,
        # a pending state capsule to preserve, or an open assistant/tool chain
        # after the latest real user.  This is intentionally not based on
        # legacy protected-message windows.
        if latest > 0:
            return True
        if self._extract_pending_state(messages):
            return True
        return len(messages) > latest + 1

    def on_session_start(self, session_id: str, **kwargs) -> None:
        self.session_id = session_id or self.session_id
        if kwargs.get("hermes_home"):
            self._hermes_home = Path(kwargs["hermes_home"])

    def on_session_reset(self) -> None:
        super().on_session_reset()
        self._last_context_render = ""
        self._last_continuity_tokens = 0
        self._last_continuity_message_count = 0
        self._last_continuity_omitted_count = 0
        self._last_continuity_summary = ""
        self._last_dropped_tool_results = 0
        self._last_ignored_handoffs = 0
        self._task_generation = 0
        self._last_active_user_request = ""
        self._last_tool_capsules = 0
        self._last_dropped_tool_chars = 0
        self._last_message_budget_tokens = 0
        self._last_tail_tokens = 0
        self._last_overweight_reason = ""
        self._last_budget_breakdown = {}
        self._last_budget_total_tokens = 0
        self._last_budget_limit_tokens = 0
        self._last_fail_closed_reason = ""
        self._evidence_capsules_written.clear()

    def compress(
        self,
        messages: List[Dict[str, Any]],
        current_tokens: int = None,
        focus_topic: str = None,
        request_budget: Mapping[str, Any] | None = None,
    ) -> List[Dict[str, Any]]:
        """Assemble a strict Letta-style active frame.

        This engine deliberately does *not* create a compressed transcript
        summary/sidecar and does *not* preserve fixed head/tail windows.  Old
        conversation history belongs in PCLTM/MemFS archival/recall memory and
        the active model-call message list is rebuilt from the latest real user
        turn plus any current assistant/tool chain that follows it.  Durable
        memories are injected separately by the system-prompt/PCLTM memory view.
        """
        sanitized, dropped = sanitize_tool_chain(messages)  # type: ignore[misc]
        latest_user_idx = self._latest_real_user_index(sanitized)

        message_budget = self._message_budget_tokens(current_tokens, request_budget=request_budget)
        self._last_message_budget_tokens = message_budget
        self._last_tool_capsules = 0
        self._last_dropped_tool_chars = 0
        self._last_archival_capsules_written = 0
        self._last_archival_messages_considered = 0
        self._last_archival_messages_evicted = 0
        self._last_archival_eviction_policy = self.governance.eviction_policy
        self._last_continuity_tokens = 0
        self._last_continuity_message_count = 0
        self._last_continuity_omitted_count = 0
        self._last_continuity_summary = ""
        self._last_tail_tokens = 0
        self._last_overweight_reason = ""
        self._last_fail_closed_reason = ""
        self._last_dropped_tool_results = int(dropped or 0)
        self._last_ignored_handoffs = sum(
            1
            for m in sanitized
            if _is_compaction_handoff_text(_content_text(m.get("content")))
        )


        if latest_user_idx is not None:
            self._archive_removed_tool_evidence(sanitized[:latest_user_idx])

        compressed = self._assemble_active_frame(
            sanitized,
            latest_user_idx=latest_user_idx,
            message_budget=message_budget,
        )
        if latest_user_idx is not None:
            self._inject_continuity_capsule(
                compressed,
                sanitized,
                latest_user_idx=latest_user_idx,
                message_budget=message_budget,
            )

        pending_state = self._extract_pending_state(sanitized)

        if pending_state:
            pending_msg = self._pending_state_message(pending_state)
            pending_cost = estimate_messages_tokens_rough([pending_msg])
            if self._last_tail_tokens + pending_cost <= message_budget or not compressed:
                compressed.insert(0, pending_msg)
                self._last_tail_tokens += pending_cost
            else:
                self._last_overweight_reason = self._last_overweight_reason or "pending_state"

        if len(compressed) >= len(messages) and len(compressed) > 1:
            # Active-frame rebuild must still converge when a diagnostic/system
            # frame offsets messages removed by tool-chain sanitization.
            for drop_idx, msg in enumerate(compressed):
                if msg.get("role") == "system":
                    continue
                removed = compressed.pop(drop_idx)
                self._last_overweight_reason = self._last_overweight_reason or "messages"
                self._last_tail_tokens = estimate_messages_tokens_rough(compressed)
                break

        compressed = self._fit_messages_to_request_budget(compressed, request_budget=request_budget)
        self._record_budget_breakdown(compressed, request_budget=request_budget)
        if self.strict_fail_closed and self._last_budget_limit_tokens > 0 and self._last_budget_total_tokens > self._last_budget_limit_tokens:
            self._last_fail_closed_reason = "request_budget_exceeded"
            raise RuntimeError(
                "PCLTM-context strict fail-closed: assembled active frame exceeds request budget "
                f"({self._last_budget_total_tokens} > {self._last_budget_limit_tokens}); "
                f"breakdown={self._last_budget_breakdown}"
            )

        # Expose only diagnostics, not replayable history.  The actual memory
        # surface is the active PCLTM memory view injected elsewhere.
        removed_count = max(0, len(sanitized) - len(compressed))
        self._last_context_render = "\n".join([
            "[PCLTM strict Letta active-frame diagnostics]",
            "legacy_handoff_sidecar: disabled",
            "fixed_head_tail_retention: disabled",
            f"removed_message_count: {removed_count}",
            f"latest_real_user_index: {latest_user_idx if latest_user_idx is not None else 'none'}",
            f"dropped_orphan_tool_results: {self._last_dropped_tool_results}",
            f"ignored_compaction_handoffs: {self._last_ignored_handoffs}",
            f"budget_total_tokens: {self._last_budget_total_tokens}",
            f"budget_limit_tokens: {self._last_budget_limit_tokens}",
            f"budget_breakdown: {self._last_budget_breakdown}",
            f"continuity_messages: {self._last_continuity_message_count}",
            f"continuity_tokens: {self._last_continuity_tokens}",
            f"continuity_omitted_messages: {self._last_continuity_omitted_count}",
        ])

        if self._last_continuity_summary:
            self._mirror_compression_summary_to_pcltm(
                content=self._last_continuity_summary,
            )

        self.compression_count += 1
        self.last_prompt_tokens = estimate_messages_tokens_rough(compressed)
        self.last_completion_tokens = 0

        return compressed

    def _mirror_compression_summary_to_pcltm(self, *, content: str) -> None:
        """Mirror compression continuity state into PCLTM short-term storage."""
        db_path = os.getenv("HERMES_PCLTM_DB")
        if not db_path or not content:
            return

        session_id = self.session_id or "unknown-session"
        conversation_id = os.getenv("HERMES_CONVERSATION_ID") or session_id
        platform = os.getenv("HERMES_PLATFORM") or "telegram"

        try:
            with sqlite3.connect(db_path) as conn:
                short_term_events = conn.execute(
                    "SELECT 1 FROM sqlite_master WHERE type='table' AND name='short_term_events'"
                ).fetchone()
                if short_term_events:
                    existing = conn.execute(
                        """
                        SELECT 1 FROM short_term_events
                        WHERE session_id = ? AND source = ? AND category = ?
                          AND subcategory = ? AND content = ?
                        LIMIT 1
                        """,
                        (session_id, "compression", "conversation", "context_summary", content),
                    ).fetchone()
                    if existing:
                        return
                    conn.execute(
                        """
                        INSERT INTO short_term_events (
                            session_id, conversation_id, platform, role, source, content,
                            persona_mode, route_bucket, model_hint, sensitivity,
                            category, subcategory, inject_policy, retention_policy, ttl_hours
                        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                        """,
                        (
                            session_id,
                            conversation_id,
                            platform,
                            "system",
                            "compression",
                            content,
                            None,
                            None,
                            None,
                            "normal",
                            "conversation",
                            "context_summary",
                            "no_memory",
                            "ttl",
                            168,
                        ),
                    )
                    conn.commit()
                    return

                legacy_short_term = conn.execute(
                    "SELECT 1 FROM sqlite_master WHERE type='table' AND name='pcltm_short_term_memory'"
                ).fetchone()
                if legacy_short_term:
                    existing = conn.execute(
                        """
                        SELECT 1 FROM pcltm_short_term_memory
                        WHERE session_id = ? AND source = ? AND category = ?
                          AND subcategory = ? AND content = ?
                        LIMIT 1
                        """,
                        (session_id, "compression", "conversation", "context_summary", content),
                    ).fetchone()
                    if existing:
                        return
                    conn.execute(
                        """
                        INSERT INTO pcltm_short_term_memory (
                            session_id, conversation_id, platform, role, source, category,
                            subcategory, content, inject_policy, retention_policy, ttl_hours
                        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                        """,
                        (
                            session_id,
                            conversation_id,
                            platform,
                            "system",
                            "compression",
                            "conversation",
                            "context_summary",
                            content,
                            "no_memory",
                            "ttl",
                            168,
                        ),
                    )
                    conn.commit()
                    return
        except Exception:
            logger.debug("PCLTM direct short-term mirror failed", exc_info=True)

        try:
            pcltm_store_module = import_pcltm_module("store")
            store_cls = getattr(pcltm_store_module, "PcltmStore", None)
            if store_cls is not None:
                store = store_cls(db_path)
                try:
                    store.append_short_term_event(
                        session_id=session_id,
                        conversation_id=conversation_id,
                        platform=platform,
                        role="system",
                        source="compression",
                        content=content,
                        category="conversation",
                        subcategory="context_summary",
                        inject_policy="no_memory",
                        retention_policy="ttl",
                        ttl_hours=168,
                    )
                finally:
                    close = getattr(store, "close", None)
                    if callable(close):
                        close()
        except Exception:
            logger.debug("PCLTM short-term store mirror failed", exc_info=True)



    def get_status(self) -> Dict[str, Any]:
        status = super().get_status()
        status.update(
            {
                "engine": self.name,
                "last_dropped_tool_results": self._last_dropped_tool_results,
                "last_ignored_handoffs": self._last_ignored_handoffs,
                "last_tool_capsules": self._last_tool_capsules,
                "last_dropped_tool_chars": self._last_dropped_tool_chars,
                "last_message_budget_tokens": self._last_message_budget_tokens,
                "last_tail_tokens": self._last_tail_tokens,
                "last_continuity_tokens": self._last_continuity_tokens,
                "last_continuity_message_count": self._last_continuity_message_count,
                "last_continuity_omitted_count": self._last_continuity_omitted_count,
                "last_continuity_summary": self._last_continuity_summary,
                "last_overweight_reason": self._last_overweight_reason,
                "last_budget_breakdown": dict(self._last_budget_breakdown),
                "last_budget_total_tokens": self._last_budget_total_tokens,
                "last_budget_limit_tokens": self._last_budget_limit_tokens,
                "configured_budget_tokens": self._configured_budget_tokens,
                "request_budget_safety_margin_tokens": self._request_budget_safety_margin_tokens,
                "last_fail_closed_reason": self._last_fail_closed_reason,
                "strict_fail_closed": self.strict_fail_closed,
                "governance": self.governance.as_dict(),
            }
        )
        return status

    def handle_tool_call(self, name: str, args: Dict[str, Any], **kwargs) -> str:
        if name == "pcltm_context_status":
            return json.dumps(self.get_status(), ensure_ascii=False)
        return super().handle_tool_call(name, args, **kwargs)

    def _latest_real_user_index(self, messages: List[Mapping[str, Any]]) -> int | None:
        for idx in range(len(messages) - 1, -1, -1):
            msg = messages[idx]
            if msg.get("role") != "user":
                continue
            text = _content_text(msg.get("content"))
            if is_runtime_control_message is not None and is_runtime_control_message(text):
                continue
            if _is_compaction_handoff_text(text):
                continue
            return idx
        return None

    def _active_frame_message(self, messages: list[dict[str, Any]], latest_user_idx: int) -> dict[str, Any] | None:
        if PCLTMContextEngine is None:
            if self.strict_fail_closed:
                raise RuntimeError(f"PCLTM-context engine unavailable: {_IMPORT_ERROR}")
            latest_text = _content_text(messages[latest_user_idx].get("content")) if latest_user_idx < len(messages) else ""
            rendered = "\n".join([
                "<pcltm_context>",
                f"【latest_real_user_message】{latest_text}",
                f"【current_user_request】{latest_text}",
                "</pcltm_context>",
            ])
            return {"role": "system", "content": rendered}
        context = PCLTMContextEngine(mode="work").build_shadow_context(messages)
        request_text = context.current_user_request or context.latest_real_user_message
        rendered = "\n".join([
            "<pcltm_context>",
            f"【latest_real_user_message】{context.latest_real_user_message}",
            f"【current_user_request】{request_text}",
            "【runtime_context_status】",
            f"  dropped_tool_results: {context.dropped_tool_results}",
            f"  ignored_handoffs: {context.ignored_handoffs}",
            "</pcltm_context>",
        ])
        return {"role": "system", "content": rendered}

    def _message_budget_tokens(
        self,
        current_tokens: int | None,
        *,
        request_budget: Mapping[str, Any] | None = None,
    ) -> int:
        explicit_message_budget = 0
        if request_budget:
            try:
                explicit_message_budget = int(request_budget.get("message_budget_tokens") or 0)
            except Exception:
                explicit_message_budget = 0
        configured_budget = int(getattr(self, "_configured_message_budget_tokens", 0) or 0)
        threshold = int(getattr(self, "threshold_tokens", 0) or 0)
        if explicit_message_budget > 0:
            base = explicit_message_budget
        elif configured_budget > 0:
            base = max(256, configured_budget)
        elif self._configured_budget_tokens > 0:
            safety_margin = int(getattr(self, "_request_budget_safety_margin_tokens", 512) or 0)
            base = max(256, self._configured_budget_tokens - max(0, safety_margin))
        else:
            base = int(threshold * 0.50)
            if current_tokens and current_tokens < threshold:
                base = min(base, int(max(2048, current_tokens * 0.70)))
            base = max(4096, min(base, 32_000))

        residual = self._request_residual_message_budget(request_budget)
        if residual > 0:
            return max(1, min(base, residual))
        if request_budget:
            return max(1, explicit_message_budget or 64)
        return base

    def _request_residual_message_budget(self, request_budget: Mapping[str, Any] | None) -> int:
        if not request_budget:
            return 0
        configured = request_budget or {}
        explicit_total = _safe_int(configured.get("total_budget_tokens"))
        if explicit_total > 0:
            total_limit = explicit_total
        else:
            total_limit = sum(
                max(0, _safe_int(configured.get(key)))
                for key in (
                    "message_budget_tokens",
                    "system_prompt_budget_tokens",
                    "tool_schema_budget_tokens",
                    "memory_prompt_budget_tokens",
                    "response_budget_tokens",
                )
            )
        if total_limit <= 0:
            return 0

        reserved = sum(
            max(0, _safe_int(configured.get(key)))
            for key in (
                "system_prompt_budget_tokens",
                "tool_schema_budget_tokens",
                "memory_prompt_budget_tokens",
                "response_budget_tokens",
            )
        )
        available = total_limit - reserved
        if available <= 0:
            return 0

        safety_margin = int(getattr(self, "_request_budget_safety_margin_tokens", 512) or 0)
        applied_margin = min(max(0, safety_margin), max(0, available - 1))
        return max(1, available - applied_margin)

    def _fit_messages_to_request_budget(
        self,
        messages: List[Dict[str, Any]],
        *,
        request_budget: Mapping[str, Any] | None = None,
    ) -> List[Dict[str, Any]]:
        if not request_budget:
            return messages
        fitted = list(messages)
        while len(fitted) > 1:
            self._record_budget_breakdown(fitted, request_budget=request_budget)
            if self._last_budget_limit_tokens <= 0 or self._last_budget_total_tokens <= self._last_budget_limit_tokens:
                return fitted
            drop_idx = None
            for idx, msg in enumerate(fitted):
                if msg.get("role") != "system":
                    drop_idx = idx
                    break
            if drop_idx is None:
                drop_idx = len(fitted) - 1
            fitted.pop(drop_idx)
            self._last_overweight_reason = self._last_overweight_reason or "request_budget"
        return fitted

    def _record_budget_breakdown(
        self,
        messages: List[Dict[str, Any]],
        *,
        request_budget: Mapping[str, Any] | None = None,
    ) -> None:
        message_tokens = estimate_messages_tokens_rough(messages)
        configured = request_budget or {}
        pending_state_tokens = estimate_messages_tokens_rough([
            m
            for m in messages
            if m.get("role") == "system" and "[PCLTM pending state capsule]" in _content_text(m.get("content"))
        ])
        buckets = {
            "messages": message_tokens,
            "system_prompt_budget": _safe_int(configured.get("system_prompt_budget_tokens")),
            "tool_schema_budget": _safe_int(configured.get("tool_schema_budget_tokens")),
            "memory_prompt_budget": _safe_int(configured.get("memory_prompt_budget_tokens")),
            "response_budget": _safe_int(configured.get("response_budget_tokens")),
        }
        self._last_budget_breakdown = buckets
        self._last_budget_total_tokens = sum(max(0, int(v or 0)) for v in buckets.values())
        explicit_total = _safe_int(configured.get("total_budget_tokens"))
        if explicit_total > 0:
            self._last_budget_limit_tokens = explicit_total
        elif request_budget:
            self._last_budget_limit_tokens = sum(
                max(0, _safe_int(configured.get(key)))
                for key in (
                    "message_budget_tokens",
                    "system_prompt_budget_tokens",
                    "tool_schema_budget_tokens",
                    "memory_prompt_budget_tokens",
                    "response_budget_tokens",
                )
            )
        else:
            self._last_budget_limit_tokens = int(self.threshold_tokens or self.context_length or 0)

    def _extract_pending_state(self, messages: List[Mapping[str, Any]]) -> str:
        for msg in messages:
            if msg.get("role") != "system":
                continue
            text = _content_text(msg.get("content"))
            if "[PCLTM pending state capsule]" in text:
                return text
        return ""

    def _pending_state_message(self, pending_state: str) -> Dict[str, Any]:
        return {"role": "system", "content": pending_state}


    def strict_healthcheck(self) -> Dict[str, Any]:
        checks = {
            "engine_name": self.name == "pcltm-context",
            "pcltm_import_available": _IMPORT_ERROR is None,
            "legacy_handoff_sidecar_disabled": True,
            "fixed_head_tail_retention_disabled": True,
            "continuity_capsule_enabled": True,
            "strict_fail_closed": bool(self.strict_fail_closed),
            "message_budget_configured": bool(self._configured_message_budget_tokens or self.threshold_tokens),
        }
        return {
            "ok": all(checks.values()),
            "checks": checks,
            "import_error": repr(_IMPORT_ERROR) if _IMPORT_ERROR is not None else "",
            "status": self.get_status(),
        }


    def _assemble_active_frame(
        self,
        messages: List[Mapping[str, Any]],
        *,
        latest_user_idx: int | None,
        message_budget: int,
    ) -> List[Dict[str, Any]]:
        """Build the Letta-style hot frame without legacy head/tail pinning.

        The active prompt is analogous to Letta/MemGPT RAM: a typed PCLTM core
        frame plus the current turn's still-open interaction chain.  Anything
        before the latest real user turn is archival/recall material and must
        not be kept merely because it appeared near the beginning or end of the
        transcript.
        """
        if latest_user_idx is None:
            self._last_tail_tokens = 0
            return []

        active_frame_msg = self._active_frame_message([dict(m) for m in messages], latest_user_idx)
        compressed: List[Dict[str, Any]] = []
        used_tokens = 0
        if active_frame_msg is not None:
            compressed.append(active_frame_msg)
            used_tokens += estimate_messages_tokens_rough([active_frame_msg])

        for idx in range(latest_user_idx, len(messages)):
            cleaned = self._clean_tail_message(messages[idx], preserve_tool_edges=True)
            if cleaned is None:
                continue
            text = _content_text(cleaned.get("content"))
            if _is_compaction_handoff_text(text):
                self._last_ignored_handoffs += 1
                continue
            cost = estimate_messages_tokens_rough([cleaned])
            is_latest = idx == latest_user_idx
            if not is_latest and used_tokens + cost > message_budget:
                self._last_overweight_reason = self._last_overweight_reason or "messages"
                continue
            compressed.append(cleaned)
            used_tokens += cost

        self._last_tail_tokens = used_tokens
        return compressed

    def _inject_continuity_capsule(
        self,
        compressed: List[Dict[str, Any]],
        messages: List[Mapping[str, Any]],
        *,
        latest_user_idx: int,
        message_budget: int,
    ) -> None:
        """Add a bounded, typed continuity capsule before the current turn.

        This is the fast-path continuity layer inspired by Letta/MemGPT active
        memory, Generative Agents recency retrieval, Graphiti temporal episodes,
        and MemoryBank rolling user/session memory.  It is deliberately typed as
        PCLTM context, not a legacy compression handoff: the goal is to preserve
        enough recent unresolved dialogue for "continue" and pronoun references
        without reviving fixed head/tail transcript retention.
        """
        if latest_user_idx <= 0 or not compressed:
            return
        if message_budget < 512:
            return
        budget = int(self.governance.continuity_budget or 0)
        remaining = max(0, message_budget - self._last_tail_tokens)
        if remaining < 64:
            self._last_continuity_omitted_count = 0
            return
        remaining = message_budget - self._last_tail_tokens
        if remaining < 256:
            return
        if budget <= 0:
            budget = max(128, min(2400, int(message_budget * 0.20)))
        budget = max(128, min(budget, remaining))
        if budget < 64:
            return
        if remaining <= 0:
            return
        budget = max(64, min(budget, remaining))
        if remaining < 256:
            self._last_continuity_omitted_count = 0
            return
        budget = max(64, min(budget, remaining))
        if remaining < 64:
            self._last_continuity_omitted_count = max(self._last_continuity_omitted_count, latest_user_idx)
            return
        budget = max(64, min(budget, remaining))
        configured_turn_limit = max(1, int(self.governance.continuity_turns or 5))
        min_dialogue_turns = max(1, int(self.governance.continuity_min_dialogue_turns or 3))
        max_dialogue_turns = max(min_dialogue_turns, int(self.governance.continuity_max_dialogue_turns or 5))
        dialogue_turn_limit = min(max_dialogue_turns, max(configured_turn_limit, min_dialogue_turns))
        total_turn_limit = max(configured_turn_limit, dialogue_turn_limit)

        selected: list[Dict[str, Any]] = []
        selected_tokens = 0
        seen_dialogue_turns = 0
        seen_user_turns = 0
        seen_user_texts: set[str] = set()
        deduped_repeated_user_turns = 0
        omitted = 0
        for msg in reversed(messages[:latest_user_idx]):
            cleaned = self._clean_tail_message(msg, preserve_tool_edges=False)
            if cleaned is None:
                continue
            text = _content_text(cleaned.get("content"))
            if not text or _is_compaction_handoff_text(text):
                continue
            role = str(cleaned.get("role") or "").lower()
            if role not in {"user", "assistant", "tool", "system"}:
                continue
            is_dialogue_turn = role in {"user", "assistant"}
            if role == "user":
                normalized_user_text = " ".join(text.split())
                if normalized_user_text in seen_user_texts:
                    deduped_repeated_user_turns += 1
                    continue
                if seen_user_turns >= total_turn_limit or seen_dialogue_turns >= dialogue_turn_limit:
                    omitted += 1
                    continue
                seen_user_texts.add(normalized_user_text)
                seen_user_turns += 1
            elif is_dialogue_turn:
                if seen_dialogue_turns >= dialogue_turn_limit or seen_user_turns >= total_turn_limit:
                    omitted += 1
                    continue
            elif len(selected) >= total_turn_limit:
                omitted += 1
                continue
            line = self._continuity_line(cleaned)
            if not line:
                continue
            candidate = {"role": "system", "content": line}
            cost = estimate_messages_tokens_rough([candidate])
            if selected_tokens + cost > budget:
                omitted += 1
                if role == "user" and seen_user_turns > 0:
                    seen_user_turns -= 1
                    seen_user_texts.discard(" ".join(text.split()))
                continue
            selected.append({"role": role, "text": line})
            if is_dialogue_turn:
                seen_dialogue_turns += 1
            selected_tokens += cost

        if not selected:
            self._last_continuity_omitted_count = omitted
            return

        selected.reverse()
        lines = [
            "[PCLTM continuity capsule]",
            "source: current_open_task_frame",
            "status: open",
            "delta_policy: delta-only",
            "semantics: typed_active_memory_not_legacy_handoff",
            "usage: reference_only_background_context",
            (
                "directive: the turns listed below are already-delivered history, kept only "
                "for pronoun resolution and 'continue'-style references. They are NOT open "
                "tasks. Do not re-execute or re-answer any request that was already answered "
                "below. Only the current (latest) user message is the active request."
            ),
            f"dialogue_window_target: {min_dialogue_turns}-{max_dialogue_turns}",
            f"max_dialogue_turns: {dialogue_turn_limit}",
            f"max_total_items: {total_turn_limit}",
            f"deduped_repeated_user_turns: {deduped_repeated_user_turns}",
        ]
        lines.extend(item["text"] for item in selected)
        if omitted:
            lines.append(f"omitted_older_or_over_budget_messages: {omitted}")
        capsule = {"role": "system", "content": "\n".join(lines)}
        capsule_cost = estimate_messages_tokens_rough([capsule])
        if self._last_tail_tokens + capsule_cost > message_budget:
            self._last_overweight_reason = self._last_overweight_reason or "continuity"
            self._last_continuity_omitted_count = omitted + len(selected)
            return
        insert_at = 1 if compressed and compressed[0].get("role") == "system" else 0
        compressed.insert(insert_at, capsule)
        self._last_tail_tokens += capsule_cost
        self._last_continuity_tokens = capsule_cost
        self._last_continuity_message_count = len(selected)
        self._last_continuity_omitted_count = omitted
        self._last_continuity_summary = _truncate(" | ".join(item["text"] for item in selected), 500)

    def _continuity_line(self, msg: Mapping[str, Any]) -> str:
        role = str(msg.get("role") or "unknown").lower()
        text = _content_text(msg.get("content"))
        if not text:
            return ""
        text = _truncate(" ".join(text.split()), 900)
        if role == "tool":
            name = str(msg.get("name") or msg.get("tool_call_id") or "tool")
            return f"- tool[{name}]: {text}"
        if role == "system":
            if "[PCLTM" not in text and "<pcltm_context>" not in text:
                return ""
            return f"- system_memory: {_truncate(text, 700)}"
        return f"- {role}: {text}"

    def _clean_tail_message(
        self,
        msg: Mapping[str, Any],
        *,
        preserve_tool_edges: bool = False,
    ) -> Dict[str, Any] | None:
        cleaned = dict(msg)
        text = _content_text(cleaned.get("content"))
        if is_compaction_handoff(text):
            return None
        if is_runtime_control_message is not None and is_runtime_control_message(text):
            return None
        if cleaned.get("role") == "tool":
            return self._tool_result_capsule(cleaned, preserve_edges=preserve_tool_edges)
        if cleaned.get("role") != "user" or runtime_visible_user_text is None:
            if len(text) > 4000:
                cleaned["content"] = _truncate(text, 4000)
            return cleaned
        visible_text = runtime_visible_user_text(cleaned.get("content"))
        if not visible_text:
            return None
        if len(visible_text) > 4000:
            visible_text = _truncate(visible_text, 4000)
        cleaned["content"] = visible_text
        return cleaned

    def _tool_result_capsule(
        self,
        msg: Mapping[str, Any],
        *,
        preserve_edges: bool = False,
    ) -> Dict[str, Any]:
        cleaned = dict(msg)
        raw_content = cleaned.get("content")
        text = _content_text(raw_content)
        tool_name = str(cleaned.get("name") or cleaned.get("tool_name") or "tool")
        tool_call_id = cleaned.get("tool_call_id") or cleaned.get("id") or "unknown"

        capsule_kind = _tool_capsule_kind(tool_name, text)
        if capsule_kind == "skill":
            dropped = len(text)
            cleaned["content"] = "\n".join([
                "[Skill result omitted from compressed active context]",
                f"tool={tool_name}",
                f"tool_call_id={tool_call_id}",
                f"original_chars={len(text)}",
                f"dropped_chars={dropped}",
                "reason=skill_view results contain procedural instructions already applied when loaded; replaying them after context compression can trigger redundant skill_view calls.",
                "Full result remains available in the session transcript/tool logs.",
            ])
            self._last_tool_capsules += 1
            self._last_dropped_tool_chars += dropped
            return cleaned

        max_chars = int(self.governance.tool_result_capsule_threshold_chars or 2400)
        if len(text) <= max_chars:
            return cleaned
        capsule_text = _render_tool_capsule(
            tool_name=tool_name,
            tool_call_id=str(tool_call_id),
            text=text,
            raw_content=raw_content,
            kind=capsule_kind,
            preserve_edges=preserve_edges,
        )
        cleaned["content"] = capsule_text
        self._maybe_write_evidence_capsule(tool_name=tool_name, tool_call_id=str(tool_call_id), kind=capsule_kind, original_text=text, capsule_text=capsule_text)
        self._last_tool_capsules += 1
        self._last_dropped_tool_chars += max(0, len(text) - len(capsule_text))
        return cleaned

    def _archive_removed_tool_evidence(self, messages: List[Mapping[str, Any]]) -> None:
        """Archive large removed tool evidence without retaining it in active context."""
        for msg in messages:
            if msg.get("role") != "tool":
                continue
            text = _content_text(msg.get("content"))
            if len(text) <= 2400:
                continue
            tool_name = str(msg.get("name") or msg.get("tool_name") or "tool")
            tool_call_id = str(msg.get("tool_call_id") or msg.get("id") or "unknown")
            kind = _tool_capsule_kind(tool_name, text)
            if kind == "skill":
                continue
            capsule_text = _render_tool_capsule(
                tool_name=tool_name,
                tool_call_id=tool_call_id,
                text=text,
                raw_content=msg.get("content"),
                kind=kind,
                preserve_edges=True,
            )
            self._maybe_write_evidence_capsule(
                tool_name=tool_name,
                tool_call_id=tool_call_id,
                kind=kind,
                original_text=text,
                capsule_text=capsule_text,
            )

    def _maybe_write_evidence_capsule(
        self,
        *,
        tool_name: str,
        tool_call_id: str,
        kind: str,
        original_text: str,
        capsule_text: str,
    ) -> None:
        if os.getenv("HERMES_PCLTM_AUTO_EVIDENCE", "").lower() not in {"1", "true", "yes", "on"}:
            return
        try:
            min_chars = int(os.getenv("HERMES_PCLTM_EVIDENCE_MIN_CHARS", "10000") or "10000")
        except ValueError:
            min_chars = 10000
        if len(original_text or "") < max(1, min_chars):
            return
        if kind == "skill":
            return
        if kind not in {"terminal", "file", "web", "json"}:
            return
        if os.getenv("HERMES_PCLTM_EVIDENCE_ERROR_ONLY", "1").lower() not in {"0", "false", "no", "off"}:
            if not _tool_capsule_indicates_error(kind, capsule_text, original_text):
                return
        try:
            pcltm_memory_adapter = import_pcltm_memory_adapter()
            writer = getattr(pcltm_memory_adapter, "write_evidence_capsule", None)
            if not callable(writer):
                return
            evidence_key = f"{tool_name}:{tool_call_id}"
            if evidence_key in self._evidence_capsules_written:
                return
            self._evidence_capsules_written.add(evidence_key)
            writer(
                title=f"{kind} evidence from {tool_name}",
                body=capsule_text,
                mode="work",
                buckets=["tool_evidence"],
                source_tool=tool_name,
                evidence_id=evidence_key,
            )
        except Exception:
            logger.debug("failed to write PCLTM evidence capsule", exc_info=True)

    def _build_handoff(
        self,
        middle: List[Mapping[str, Any]],
        *,
        current_user_request: str,
        dropped_tool_results: int,
        ignored_handoffs: int,
        focus_topic: str = None,
        max_tokens: int | None = None,
    ) -> str:
        role_counts: dict[str, int] = {}
        snippets: list[str] = []
        user_snippets: list[str] = []

        for msg in middle:
            role = str(msg.get("role") or "message")
            role_counts[role] = role_counts.get(role, 0) + 1

        priority_user_candidates: list[str] = []
        user_candidates: list[str] = []
        candidates: list[str] = []
        seen_user_facts: set[str] = set()
        for msg in reversed(middle):
            if len(candidates) >= 10 and len(user_candidates) + len(priority_user_candidates) >= 8:
                break
            role = str(msg.get("role") or "message")
            text = _content_text(msg.get("content")).strip()
            if not text:
                continue
            if _is_compaction_handoff_text(text):
                for fact in _extract_handoff_user_facts(text):
                    if _is_task_like_handoff_fact(fact) and fact.strip() != (current_user_request or "").strip():
                        continue
                    _add_user_fact_candidate(
                        fact,
                        priority_user_candidates=priority_user_candidates,
                        user_candidates=user_candidates,
                        seen_user_facts=seen_user_facts,
                    )
                continue
            if role == "user" and is_runtime_control_message is not None and is_runtime_control_message(text):
                continue
            if role == "user":
                _add_user_fact_candidate(
                    text,
                    priority_user_candidates=priority_user_candidates,
                    user_candidates=user_candidates,
                    seen_user_facts=seen_user_facts,
                )
                continue
            if len(candidates) >= 10:
                continue
            if role == "tool":
                candidates.append(f"- {role}: {_truncate(text, 240)}")
                continue
            if role == "assistant" and msg.get("tool_calls"):
                candidates.append(f"- {role}: {_truncate(text, 240)}")
                continue

        user_snippets = list(reversed(priority_user_candidates)) + list(reversed(user_candidates))
        user_snippets = user_snippets[:8]
        snippets = list(reversed(candidates))

        lines = [
            f"{SUMMARY_PREFIX}",
            "## PCLTM-context Reference Sidecar",
            "PCLTM-context is now the only active context compression layer for this session.",
            "The native Hermes compressor was not used for this handoff.",
            "This block is reference-only compressed context, not a user request, task command, or resume instruction.",
            "No active request is restored from compressed context; the current request must come only from the latest real retained user message after this block.",
            "Preserved todos, weak resumes, and older handoff evidence are background evidence only unless the latest real user message explicitly names and resumes them.",
            "",
            "## Compressed Middle Window",
            f"message_count: {len(middle)}",
            f"role_counts: {role_counts}",
            f"dropped_orphan_tool_results: {dropped_tool_results}",
            f"ignored_compaction_handoffs: {ignored_handoffs}",
        ]
        if focus_topic:
            lines.extend(["", "## Focus Topic", str(focus_topic)])
        if user_snippets:
            lines.extend(["", "## Reference User Facts From Removed Window", *user_snippets])
        if snippets:
            lines.extend(["", "## Reference Evidence From Removed Window", *snippets])
        lines.append("\n--- END OF PCLTM-context reference sidecar — respond only to the latest real user message below ---")
        return "\n".join(lines)

    def _assemble_valid_sequence(
        self,
        head: List[Dict[str, Any]],
        handoff: str,
        tail: List[Dict[str, Any]],
    ) -> List[Dict[str, Any]]:
        if not head:
            return [{"role": "user", "content": handoff}, *tail]

        result = list(head)
        first_tail_role = tail[0].get("role") if tail else None
        last_head_role = result[-1].get("role")
        # Prefer a system handoff for context-compression summaries. It is
        # neutral in OpenAI role alternation, keeps the boundary explicit, and
        # prevents tests/runtime readers from mistaking a retained tail user
        # message for the handoff body when user/assistant alternation would
        # otherwise conflict.
        handoff_role = "system"
        if not tail or first_tail_role == "system":
            handoff_role = "user" if last_head_role != "user" else "assistant"
            if first_tail_role == handoff_role:
                flipped = "assistant" if handoff_role == "user" else "user"
                if flipped != last_head_role:
                    handoff_role = flipped
                else:
                    handoff_role = "system"
        result.append({"role": handoff_role, "content": handoff})
        result.extend(tail)
        return result


def _safe_int(value: Any, default: int = 0) -> int:
    try:
        return int(value or 0)
    except Exception:
        return default


def _coerce_bool(value: Any, *, default: bool = False) -> bool:
    if value is None:
        return default
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, float)):
        return bool(value)
    if isinstance(value, str):
        lowered = value.strip().lower()
        if lowered in {"1", "true", "yes", "on"}:
            return True
        if lowered in {"0", "false", "no", "off"}:
            return False
    return default


def _content_text(content: Any) -> str:
    if content is None:
        return ""
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts = []
        for item in content:
            if isinstance(item, str):
                parts.append(item)
            elif isinstance(item, Mapping):
                value = item.get("text") or item.get("content") or ""
                if isinstance(value, str):
                    parts.append(value)
        return "\n".join(parts)
    return str(content)


def _is_compaction_handoff_text(text: str) -> bool:
    markers = (
        "[CONTEXT COMPACTION",
        "[CONTEXT SUMMARY]:",
        "🗜️ Compacting context",
        "## PCLTM-context Reference Sidecar",
        "--- END OF PCLTM-context reference sidecar",
        "## Active Task",
    )
    stripped = text.lstrip()
    return any(marker in stripped for marker in markers)


def _extract_handoff_user_facts(text: str) -> list[str]:
    markers = (
        "## Recent User Facts/Requests From Removed Window",
        "## Reference User Facts From Removed Window",
    )
    marker = next((candidate for candidate in markers if candidate in text), "")
    if not marker:
        return []
    section = text.split(marker, 1)[1]
    for stop in ("\n## ", "\n--- END OF PCLTM-context", "\n--- END OF PCLTM-context reference sidecar"):
        if stop in section:
            section = section.split(stop, 1)[0]
    facts: list[str] = []
    for line in section.splitlines():
        stripped = line.strip()
        if not stripped.startswith("- user:"):
            continue
        fact = stripped[len("- user:"):].strip()
        if fact and not _is_compaction_handoff_text(fact):
            facts.append(fact)
    return facts


def _add_user_fact_candidate(
    text: str,
    *,
    priority_user_candidates: list[str],
    user_candidates: list[str],
    seen_user_facts: set[str],
    limit: int = 16,
) -> None:
    fact = " ".join(text.split())
    if not fact or fact in seen_user_facts:
        return
    seen_user_facts.add(fact)
    rendered = f"- user: {_truncate(fact, 240)}"
    if _looks_important_user_fact(fact):
        if len(priority_user_candidates) < limit:
            priority_user_candidates.append(rendered)
        return
    if len(user_candidates) < limit:
        user_candidates.append(rendered)


def _looks_important_user_fact(text: str) -> bool:
    lowered = text.lower()
    markers = (
        "critical",
        "important",
        "must",
        "keep",
        "anchor",
        "do not lose",
        "don't lose",
        "不能丢",
        "不要丢",
        "别丢",
        "重要",
        "记住",
        "保留",
    )
    return any(marker in lowered for marker in markers)


def _is_task_like_handoff_fact(text: str) -> bool:
    lowered = (text or "").lower()
    markers = (
        "做",
        "修",
        "改",
        "查",
        "排查",
        "验证",
        "重启",
        "优化",
        "实现",
        "处理",
        "修复",
        "继续",
        "task",
        "fix",
        "debug",
        "verify",
        "restart",
        "implement",
        "optimize",
    )
    return any(marker in lowered for marker in markers)


def _is_skill_tool_result(tool_name: str, text: str) -> bool:
    normalized = (tool_name or "").strip().lower()
    if normalized in {"skill_view", "skills_view"}:
        return True
    stripped = (text or "").lstrip()
    return (
        '"readiness_status"' in stripped
        and '"linked_files"' in stripped
        and '"usage_hint"' in stripped
        and '"content"' in stripped
    )


def _tool_capsule_kind(tool_name: str, text: str) -> str:
    normalized = (tool_name or "").strip().lower()
    stripped = (text or "").lstrip()
    if _is_skill_tool_result(tool_name, text):
        return "skill"
    if normalized in {"read_file", "search_files"} or '"total_lines"' in stripped or '"matches"' in stripped:
        return "file"
    if normalized in {"terminal", "execute_code", "process"} or '"exit_code"' in stripped or '"stdout"' in stripped:
        return "terminal"
    if normalized in {"web_extract", "web_search", "browser_console", "browser_snapshot"} or '"url"' in stripped:
        return "web"
    if stripped.startswith("{") or stripped.startswith("["):
        return "json"
    return "generic"


def _parse_structured_tool_content(raw_content: Any, text: str = "") -> Any:
    if isinstance(raw_content, (Mapping, list)):
        return raw_content
    if isinstance(raw_content, str):
        stripped = raw_content.strip()
        if stripped.startswith("{") or stripped.startswith("["):
            try:
                return json.loads(stripped)
            except Exception:
                try:
                    return ast.literal_eval(stripped)
                except Exception:
                    return None
    if text:
        stripped = text.strip()
        if stripped.startswith("{") or stripped.startswith("["):
            try:
                return json.loads(stripped)
            except Exception:
                try:
                    return ast.literal_eval(stripped)
                except Exception:
                    return None
    return None


def _json_capsule_excerpt(text: str, limit: int = 900, raw_content: Any = None) -> str:
    parsed = _parse_structured_tool_content(raw_content, text)
    if parsed is None:
        return _truncate(text, limit)
    if isinstance(parsed, dict):
        safe_keys = [
            "path", "url", "title", "description", "total_lines", "exit_code",
            "error", "status", "count", "memory_id", "line", "offset", "limit", "command",
        ]
        safe = {key: parsed.get(key) for key in safe_keys if key in parsed}
        if "content" in parsed and isinstance(parsed.get("content"), str):
            safe["content_excerpt"] = _truncate(parsed["content"], 300)
        if "output" in parsed and isinstance(parsed.get("output"), str):
            safe["output_excerpt"] = _truncate(parsed["output"], 300)
        if safe:
            return json.dumps(safe, ensure_ascii=False)
    return _truncate(text, limit)


def _render_tool_capsule(
    *,
    tool_name: str,
    tool_call_id: str,
    text: str,
    raw_content: Any = None,
    kind: str,
    preserve_edges: bool,
) -> str:
    labels = {
        "file": "[File tool evidence capsule for active context]",
        "terminal": "[Terminal tool evidence capsule for active context]",
        "web": "[Web/browser tool evidence capsule for active context]",
        "json": "[Structured tool evidence capsule for active context]",
        "generic": "[Tool result truncated for active context]",
    }
    head_limit = 900 if preserve_edges else 700
    tail_limit = 120 if preserve_edges else 0
    if kind in {"json", "file", "terminal", "web"}:
        head = _json_capsule_excerpt(text, head_limit, raw_content=raw_content)
    else:
        head = _truncate(text, head_limit)
    tail = ""
    # Letta-style evidence capsules intentionally avoid preserving arbitrary
    # raw tail bytes for structured/tool outputs. The full transcript remains
    # the retrieval source; active context keeps only typed excerpts.
    if tail_limit and kind == "generic":
        tail = _truncate(text[-tail_limit:], tail_limit)
    dropped = max(0, len(text) - len(head) - len(tail))
    capsule = [
        labels.get(kind, labels["generic"]),
        f"tool={tool_name}",
        f"tool_call_id={tool_call_id}",
        f"capsule_kind={kind}",
        f"original_chars={len(text)}",
        f"dropped_chars={dropped}",
        "head_excerpt:",
        head,
    ]
    if tail:
        capsule.extend(["tail_excerpt:", tail])
    capsule.append("Full result remains available in the session transcript/tool logs; use explicit file/session/PCLTM retrieval if more detail is needed.")
    return "\n".join(capsule)


def _tool_capsule_indicates_error(kind: str, capsule_text: str, original_text: str = "") -> bool:
    parsed = _parse_structured_tool_content(None, original_text)
    if isinstance(parsed, dict):
        exit_code = parsed.get("exit_code")
        if exit_code is not None:
            try:
                return int(exit_code) != 0
            except Exception:
                return str(exit_code).strip() not in {"", "0", "None", "none"}
        for key in ("error", "exception", "traceback"):
            value = parsed.get(key)
            if isinstance(value, str) and value.strip():
                return True
        status = parsed.get("status")
        if isinstance(status, str) and status.lower() in {"error", "failed", "failure"}:
            return True
    lowered = f"{capsule_text or ''}\n{original_text or ''}".lower()
    if kind == "terminal":
        if '"exit_code": 0' in lowered or "exit_code=0" in lowered:
            return False
        if "exit_code" in lowered:
            return True
    error_markers = (
        "traceback",
        "exception",
        "error",
        "failed",
        "failure",
        "non-zero",
        "exit_code",
    )
    return any(marker in lowered for marker in error_markers)


def _truncate(text: str, limit: int) -> str:
    text = " ".join(text.split())
    if len(text) <= limit:
        return text
    return text[: max(0, limit - 1)] + "…"


def register(ctx) -> None:
    ctx.register_context_engine(PCLTMContextCompressionEngine())