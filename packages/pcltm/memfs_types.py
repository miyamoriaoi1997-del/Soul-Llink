"""MemFS view types for PCLTM layered memory.

Borrowed architectural ideas from Letta (formerly MemGPT):
  - system/  always-loaded context
  - pinned/   high-priority stable facts
  - episodic/ event-based memory selected by query/recency
  - transient/ current-session/task evidence

Authority order (highest to lowest):
  1. Core identity / SOUL contract
  2. Runtime structured state (mode, emotion, route)
  3. PCLTM layered memory views
  4. Recent real user messages
  5. Compression summaries as reference-only
"""

from __future__ import annotations

import enum
from dataclasses import dataclass, field
from typing import Any, Iterable


def _as_tuple(value: object) -> tuple[str, ...]:
    if value is None:
        return ()
    if isinstance(value, str):
        return (value,)
    if isinstance(value, Iterable):
        return tuple(str(v) for v in value)
    return (str(value),)


class MemoryAuthority(str, enum.Enum):
    """Memory layer authority level.  Lower value == higher authority."""

    SYSTEM = "system"
    PINNED = "pinned"
    EPISODIC = "episodic"
    TRANSIENT = "transient"
    COMPRESSION = "compression"  # reference-only, never authoritative


VALID_AUTHORITY_VALUES = {a.value for a in MemoryAuthority}
VALID_MODE_SCOPES = {"daily", "work", "sex", "cron", "default"}


class MemoryRecordType(str, enum.Enum):
    """Typed memory categories for PCLTM/MemFS governance."""

    USER_PREFERENCE = "UserPreference"
    PROJECT_PATH = "ProjectPath"
    RUNTIME_INVARIANT = "RuntimeInvariant"
    PERSONA_BOUNDARY = "PersonaBoundary"
    TOOL_QUIRK = "ToolQuirk"
    RELATIONSHIP_ANCHOR = "RelationshipAnchor"
    FINANCIAL_PROFILE = "FinancialProfile"
    WORKFLOW_CONVENTION = "WorkflowConvention"
    RISK_POLICY = "RiskPolicy"
    TEMPORARY_TASK_STATE = "TemporaryTaskState"


class MemoryLifecycleState(str, enum.Enum):
    """Stable lifecycle states for Letta-inspired memory governance."""

    OBSERVED = "observed"
    CANDIDATE = "candidate"
    SCORED = "scored"
    PENDING_REVIEW = "pending_review"
    APPROVED = "approved"
    ACTIVE = "active"
    DECAYED = "decayed"
    MERGED = "merged"
    SUPERSEDED = "superseded"
    ARCHIVED = "archived"
    REJECTED = "rejected"


VALID_MEMORY_TYPES = {item.value for item in MemoryRecordType}
VALID_LIFECYCLE_STATES = {item.value for item in MemoryLifecycleState}


@dataclass(frozen=True)
class MemoryTypePolicy:
    """Policy attached to a typed memory record.

    The policy is control-plane metadata.  It tells selectors/governors how a
    record may be injected or reviewed; it does not mutate memory by itself.
    """

    type_name: str
    ttl: str = "none"
    authority: str = "medium"
    injectable_modes: tuple[str, ...] = ("daily", "work", "sex")
    conflict_policy: str = "merge_similar"
    injection_policy: str = "mode_aware"
    review_required: bool = False
    compression_visible: bool = False
    prompt_bucket: str = "generic"
    priority: int = 50

    @classmethod
    def for_type(cls, type_name: str | MemoryRecordType) -> "MemoryTypePolicy":
        name = type_name.value if isinstance(type_name, MemoryRecordType) else str(type_name)
        policies = {
            "UserPreference": cls("UserPreference", authority="medium", prompt_bucket="user_preference", priority=70),
            "ProjectPath": cls("ProjectPath", authority="medium", injectable_modes=("work", "cron"), prompt_bucket="project_path", priority=60),
            "RuntimeInvariant": cls("RuntimeInvariant", authority="high", conflict_policy="strict", injection_policy="always", prompt_bucket="runtime_boundary", priority=90),
            "PersonaBoundary": cls("PersonaBoundary", authority="high", conflict_policy="strict", review_required=True, prompt_bucket="emotion_boundary", priority=95),
            "ToolQuirk": cls("ToolQuirk", authority="medium", injectable_modes=("work", "cron"), prompt_bucket="runtime_boundary", priority=60),
            "RelationshipAnchor": cls("RelationshipAnchor", authority="high", injectable_modes=("daily", "sex"), conflict_policy="strict", review_required=True, prompt_bucket="relationship", priority=85),
            "FinancialProfile": cls("FinancialProfile", authority="medium", injectable_modes=("work",), prompt_bucket="investment", priority=50),
            "WorkflowConvention": cls("WorkflowConvention", authority="medium", injectable_modes=("work", "cron"), prompt_bucket="current_task", priority=65),
            "RiskPolicy": cls("RiskPolicy", authority="high", conflict_policy="strict", review_required=True, prompt_bucket="runtime_boundary", priority=90),
            "TemporaryTaskState": cls("TemporaryTaskState", ttl="short", authority="low", injectable_modes=("work", "cron"), conflict_policy="replace_newer", injection_policy="transient_only", prompt_bucket="current_task", priority=20),
        }
        if name not in policies:
            raise ValueError(f"invalid memory_type {name!r}; expected one of {sorted(VALID_MEMORY_TYPES)}")
        return policies[name]


@dataclass(frozen=True)
class MemoryFileFrontmatter:
    """Parsed YAML frontmatter of a MemFS memory file."""

    description: str
    authority: str = "pinned"
    mode_scope: tuple[str, ...] = ("daily", "work", "sex")
    buckets: tuple[str, ...] = ()
    source: str = "pcltm"
    last_reviewed: str = ""
    char_limit: int | None = None
    read_only: bool = False
    metadata: dict[str, Any] = field(default_factory=dict)
    updated_at: str = ""
    memory_type: str = "UserPreference"
    lifecycle_state: str = "active"
    ttl: str = "none"
    conflict_policy: str = ""
    injection_policy: str = ""
    evidence_refs: tuple[dict[str, Any], ...] = ()
    extra: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not self.description or not self.description.strip():
            raise ValueError("description is required")
        if self.authority not in VALID_AUTHORITY_VALUES:
            raise ValueError(
                f"invalid authority {self.authority!r}; "
                f"expected one of {sorted(VALID_AUTHORITY_VALUES)}"
            )
        object.__setattr__(self, "mode_scope", _as_tuple(self.mode_scope))
        object.__setattr__(self, "buckets", _as_tuple(self.buckets))
        object.__setattr__(self, "evidence_refs", tuple(dict(ref) for ref in self.evidence_refs))
        if self.char_limit is not None and int(self.char_limit) < 0:
            raise ValueError("char_limit must be >= 0")
        if self.memory_type not in VALID_MEMORY_TYPES:
            raise ValueError(
                f"invalid memory_type {self.memory_type!r}; expected one of {sorted(VALID_MEMORY_TYPES)}"
            )
        if self.lifecycle_state not in VALID_LIFECYCLE_STATES:
            raise ValueError(
                f"invalid lifecycle_state {self.lifecycle_state!r}; expected one of {sorted(VALID_LIFECYCLE_STATES)}"
            )
        for m in self.mode_scope:
            if m not in VALID_MODE_SCOPES:
                raise ValueError(
                    f"invalid mode_scope {m!r}; "
                    f"expected one of {sorted(VALID_MODE_SCOPES)}"
                )

    @property
    def authority_enum(self) -> MemoryAuthority:
        return MemoryAuthority(self.authority)

    @property
    def type_policy(self) -> MemoryTypePolicy:
        policy = MemoryTypePolicy.for_type(self.memory_type)
        return MemoryTypePolicy(
            type_name=policy.type_name,
            ttl=self.ttl or policy.ttl,
            authority=policy.authority,
            injectable_modes=policy.injectable_modes,
            conflict_policy=self.conflict_policy or policy.conflict_policy,
            injection_policy=self.injection_policy or policy.injection_policy,
            review_required=policy.review_required,
            compression_visible=policy.compression_visible,
            prompt_bucket=policy.prompt_bucket,
            priority=policy.priority,
        )


@dataclass(frozen=True)
class MemoryLayerItem:
    """A single memory item within a layer view."""

    path: str
    id: str = ""
    description: str = ""
    body: str = ""
    authority: str = "pinned"
    buckets: tuple[str, ...] = ()
    mode_scope: tuple[str, ...] = ("daily", "work", "sex")
    char_count: int = 0
    char_limit: int = 0
    read_only: bool = False
    metadata: dict[str, Any] = field(default_factory=dict)
    updated_at: str = ""
    score: float = 0.0  # relevance score, higher = more relevant
    memory_type: str = "UserPreference"
    lifecycle_state: str = "active"
    ttl: str = "none"
    conflict_policy: str = ""
    injection_policy: str = ""
    evidence_refs: tuple[dict[str, Any], ...] = ()

    def __post_init__(self) -> None:
        if not self.id:
            object.__setattr__(self, "id", self.path)
        if self.authority not in VALID_AUTHORITY_VALUES:
            raise ValueError(f"invalid authority {self.authority!r}")
        if self.char_limit < 0:
            raise ValueError("char_limit must be >= 0")
        object.__setattr__(self, "buckets", _as_tuple(self.buckets))
        object.__setattr__(self, "mode_scope", _as_tuple(self.mode_scope))
        object.__setattr__(self, "evidence_refs", tuple(dict(ref) for ref in self.evidence_refs))
        if self.memory_type not in VALID_MEMORY_TYPES:
            raise ValueError(f"invalid memory_type {self.memory_type!r}")
        if self.lifecycle_state not in VALID_LIFECYCLE_STATES:
            raise ValueError(f"invalid lifecycle_state {self.lifecycle_state!r}")

    @property
    def type_policy(self) -> MemoryTypePolicy:
        policy = MemoryTypePolicy.for_type(self.memory_type)
        return MemoryTypePolicy(
            type_name=policy.type_name,
            ttl=self.ttl or policy.ttl,
            authority=policy.authority,
            injectable_modes=policy.injectable_modes,
            conflict_policy=self.conflict_policy or policy.conflict_policy,
            injection_policy=self.injection_policy or policy.injection_policy,
            review_required=policy.review_required,
            compression_visible=policy.compression_visible,
            prompt_bucket=policy.prompt_bucket,
            priority=policy.priority,
        )


@dataclass
class MemoryLayerView:
    """View of one memory layer (system/pinned/episodic/transient/compression)."""

    layer: str
    items: list[MemoryLayerItem] = field(default_factory=list)
    omitted_count: int = 0
    budget_chars: int = 0
    used_chars: int = 0
    is_reference_only: bool = False  # True for compression layers

    def __post_init__(self) -> None:
        if self.layer not in VALID_AUTHORITY_VALUES:
            raise ValueError(f"invalid layer {self.layer!r}")

    @property
    def total_chars(self) -> int:
        return sum(i.char_count for i in self.items)

    def render(self) -> str:
        """Render layer as prompt-safe labelled text.

        Control-plane metadata (record ids, candidate ids, scores, source paths,
        review timestamps, etc.) stays available through ``context_summary()``
        and audits.  The active prompt receives only the memory value plus the
        layer label so metadata cannot become high-salience instructions.
        """
        ref_tag = " [reference-only]" if self.is_reference_only else ""
        header = f"[{self.layer}]{ref_tag}"
        if not self.items:
            body = "(empty)"
        else:
            parts: list[str] = []
            for item in self.items:
                prefix_parts: list[str] = []
                if item.memory_type:
                    prefix_parts.append(f"type={item.memory_type}")
                if item.lifecycle_state:
                    prefix_parts.append(f"state={item.lifecycle_state}")
                if item.buckets:
                    prefix_parts.append("buckets=" + ",".join(sorted(item.buckets)))
                prefix = f"[{'; '.join(prefix_parts)}] " if prefix_parts else ""
                if item.body:
                    parts.append(f"{prefix}{item.body.strip()}")
                else:
                    parts.append(f"{prefix}{(item.description.strip() or item.path)}")
            body = "\n".join(part for part in parts if part)
        if self.omitted_count > 0:
            body += f"\n[省略 {self.omitted_count} 条超出预算的记录]"
        return f"{header}\n{body}"

    def to_dict(self) -> dict:
        return {
            "layer": self.layer,
            "item_count": len(self.items),
            "omitted_count": self.omitted_count,
            "budget_chars": self.budget_chars,
            "used_chars": self.used_chars,
            "is_reference_only": self.is_reference_only,
            "memory_types": sorted({item.memory_type for item in self.items}),
            "lifecycle_states": sorted({item.lifecycle_state for item in self.items}),
            "rendered_preview": self.render()[:160],
        }

    def selection_audit(self) -> dict[str, Any]:
        """Return prompt-safe control-plane details about layer selection."""
        selected_items = []
        for item in self.items:
            selected_items.append(
                {
                    "id": item.id,
                    "path": item.path,
                    "description": item.description,
                    "memory_type": item.memory_type,
                    "lifecycle_state": item.lifecycle_state,
                    "authority": item.authority,
                    "buckets": list(item.buckets),
                    "mode_scope": list(item.mode_scope),
                    "score": item.score,
                    "char_count": item.char_count,
                    "char_limit": item.char_limit,
                    "injection_policy": item.injection_policy,
                    "ttl": item.ttl,
                    "read_only": item.read_only,
                }
            )
        omitted_reasons = []
        if self.omitted_count:
            omitted_reasons.append(
                {
                    "reason": "budget_exceeded_or_filter_not_selected",
                    "count": self.omitted_count,
                }
            )
        return {
            "layer": self.layer,
            "selected_count": len(self.items),
            "omitted_count": self.omitted_count,
            "budget_chars": self.budget_chars,
            "used_chars": self.used_chars,
            "remaining_chars": max(0, self.budget_chars - self.used_chars),
            "selected_items": selected_items,
            "omitted_reasons": omitted_reasons,
            "is_reference_only": self.is_reference_only,
        }


@dataclass
class ContextSelectionSnapshot:
    """Host-neutral snapshot of a PCLTM active-context selection.

    The snapshot is control-plane data: it describes what entered the active
    frame, what stayed reference-only, and why items were omitted.  It is safe
    for host adapters to serialize or dashboard without depending on Hermes
    sessions, prompt builders, or compression internals.
    """

    schema_version: int = 1
    selection_source: str = "default"
    active_layers: tuple[str, ...] = ()
    selected_layers: tuple[str, ...] = ()
    selected_buckets: tuple[str, ...] = ()
    reference_only_layers: tuple[str, ...] = ()
    total_selected_items: int = 0
    total_omitted_items: int = 0
    total_used_chars: int = 0
    layers: tuple[dict[str, Any], ...] = ()
    compression: dict[str, Any] = field(default_factory=dict)
    warnings: tuple[str, ...] = ()

    @classmethod
    def from_prompt_memory_view(cls, view: "PromptMemoryView") -> "ContextSelectionSnapshot":
        summary = view.context_summary()
        audit = summary.get("selection_audit") if isinstance(summary, dict) else {}
        if not isinstance(audit, dict):
            audit = {}

        def _string_tuple(mapping: dict[str, Any], key: str) -> tuple[str, ...]:
            value = mapping.get(key, [])
            if not isinstance(value, list | tuple):
                return ()
            return tuple(str(item) for item in value if item)

        def _int_value(*values: Any) -> int:
            for value in values:
                try:
                    return int(value)
                except (TypeError, ValueError):
                    continue
            return 0

        raw_layers = audit.get("layers", [])
        if not isinstance(raw_layers, list):
            raw_layers = []
        compression = summary.get("compression")
        if not isinstance(compression, dict):
            compression = {}
        return cls(
            schema_version=_int_value(audit.get("schema_version"), 1),
            selection_source=str(summary.get("selection_source") or audit.get("selection_source") or "default"),
            active_layers=_string_tuple(summary, "active_layers"),
            selected_layers=_string_tuple(summary, "selected_layers"),
            selected_buckets=_string_tuple(summary, "selected_buckets"),
            reference_only_layers=_string_tuple(summary, "reference_only_layers"),
            total_selected_items=_int_value(audit.get("total_selected_items"), summary.get("total_items")),
            total_omitted_items=_int_value(audit.get("total_omitted_items"), summary.get("total_omitted")),
            total_used_chars=_int_value(audit.get("total_used_chars"), summary.get("total_chars")),
            layers=tuple(dict(layer) for layer in raw_layers if isinstance(layer, dict)),
            compression=dict(compression),
            warnings=_string_tuple(audit, "warnings"),
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "object_type": "pcltm_context_selection_snapshot",
            "schema_version": self.schema_version,
            "selection_source": self.selection_source,
            "active_layers": list(self.active_layers),
            "selected_layers": list(self.selected_layers),
            "selected_buckets": list(self.selected_buckets),
            "reference_only_layers": list(self.reference_only_layers),
            "total_selected_items": self.total_selected_items,
            "total_omitted_items": self.total_omitted_items,
            "total_used_chars": self.total_used_chars,
            "layers": [dict(layer) for layer in self.layers],
            "compression": dict(self.compression),
            "warnings": list(self.warnings),
        }


@dataclass
class ActiveContextFrame:
    """Soul-Link active context frame for prompt assembly and audit.

    Inspired by Letta/MemGPT's split between in-context memory and external
    memory, but this is PCLTM's own frame contract rather than a 1:1 Letta type.

    ``active_text`` is the only memory text intended for the model prompt.
    ``reference_summary``/``audit`` describe archival/compression state without
    granting those layers active authority.
    """

    active_text: str = ""
    reference_summary: dict[str, Any] = field(default_factory=dict)
    audit: dict[str, Any] = field(default_factory=dict)
    selected_layers: tuple[str, ...] = ()
    reference_only_layers: tuple[str, ...] = ()
    warnings: tuple[str, ...] = ()

    def to_dict(self) -> dict[str, Any]:
        return {
            "active_text_chars": len(self.active_text),
            "reference_summary": self.reference_summary,
            "audit": self.audit,
            "prompt_active_layers": list(self.selected_layers),
            "selected_layers": list(self.selected_layers),
            "reference_only_layers": list(self.reference_only_layers),
            "warnings": list(self.warnings),
        }


@dataclass
class PromptMemoryView:
    """Aggregate layered memory view for prompt injection.

    Layers are rendered in PCLTM authority order inspired by Letta/MemGPT:
      system > pinned > episodic > transient

    The legacy compression slot remains on the object only for compatibility
    with older call sites, but it is not part of the active render path.
    """

    system: MemoryLayerView = field(
        default_factory=lambda: MemoryLayerView(layer="system")
    )
    pinned: MemoryLayerView = field(
        default_factory=lambda: MemoryLayerView(layer="pinned")
    )
    episodic: MemoryLayerView = field(
        default_factory=lambda: MemoryLayerView(layer="episodic")
    )
    transient: MemoryLayerView = field(
        default_factory=lambda: MemoryLayerView(layer="transient")
    )
    compression: MemoryLayerView = field(
        default_factory=lambda: MemoryLayerView(
            layer="compression", is_reference_only=True
        )
    )
    selected_layers: tuple[str, ...] = ()
    selected_buckets: tuple[str, ...] = ()
    selection_source: str = "default"

    @property
    def layers(self) -> list[MemoryLayerView]:
        """Return active layers in authority order.

        Compression is intentionally excluded from active context assembly.
        """
        return [
            self.system,
            self.pinned,
            self.episodic,
            self.transient,
        ]

    @property
    def total_chars(self) -> int:
        return sum(l.total_chars for l in self.layers)

    @property
    def total_items(self) -> int:
        return sum(len(l.items) for l in self.layers)

    @property
    def total_omitted(self) -> int:
        return sum(l.omitted_count for l in self.layers)

    def context_summary(self) -> dict[str, object]:
        """Return PCLTM active context audit metadata."""
        active_layers = list(self.selected_layers) if self.selected_layers else [layer.layer for layer in self.layers]
        layer_audits = [layer.selection_audit() for layer in self.layers]
        return {
            "active_layers": active_layers,
            "selected_layers": active_layers,
            "selected_buckets": list(self.selected_buckets),
            "selection_source": self.selection_source,
            "total_items": self.total_items,
            "total_chars": self.total_chars,
            "total_omitted": self.total_omitted,
            "selection_audit": {
                "schema_version": 1,
                "selection_source": self.selection_source,
                "active_layers": active_layers,
                "selected_buckets": list(self.selected_buckets),
                "total_selected_items": self.total_items,
                "total_omitted_items": self.total_omitted,
                "total_used_chars": self.total_chars,
                "layers": layer_audits,
                "warnings": [
                    "transient_layer_empty"
                ] if not self.transient.items else [],
            },
            "reference_only_layers": [layer.layer for layer in self.layers if layer.layer not in active_layers] + ["compression"],
            "layers": [
                {
                    **layer.to_dict(),
                    "buckets": sorted({bucket for item in layer.items for bucket in item.buckets}),
                    "item_paths": [item.path for item in layer.items],
                }
                for layer in self.layers
            ],
            "compression": self.compression.to_dict(),
        }

    def render(self) -> str:
        """Render all active layers with explicit labels.

        Compression is retained only as a compatibility field and never rendered.
        """
        parts: list[str] = []
        for view in self.layers:
            rendered = view.render()
            if rendered and rendered != f"[{view.layer}]\n(empty)":
                parts.append(rendered)
        return "\n\n".join(parts) if parts else ""

    def render_active_frame(self) -> str:
        """Render a PCLTM active working set.

        The active frame is deliberately smaller than ``render()`` when the
        selector marks layers as archival/reference-only.  Archival layers stay
        available via ``context_summary()`` and explicit retrieval/open tools but
        do not pour full bodies into the model prompt.
        """
        active_layers = set(self.selected_layers or [layer.layer for layer in self.layers])
        parts: list[str] = []
        for view in self.layers:
            if view.layer not in active_layers:
                continue
            rendered = view.render()
            if rendered and rendered != f"[{view.layer}]\n(empty)":
                parts.append(rendered)
        return "\n\n".join(parts) if parts else ""

    def context_selection_snapshot(self) -> ContextSelectionSnapshot:
        """Return a host-neutral control-plane snapshot for this selection."""
        return ContextSelectionSnapshot.from_prompt_memory_view(self)

    def active_frame(self) -> ActiveContextFrame:
        summary = self.context_summary()
        raw_reference_layers = summary.get("reference_only_layers", [])
        if not isinstance(raw_reference_layers, list):
            raw_reference_layers = []
        reference_layers = tuple(str(layer) for layer in raw_reference_layers)
        raw_layers = summary.get("layers", [])
        if not isinstance(raw_layers, list):
            raw_layers = []
        reference_summary = {
            "reference_only_layers": list(reference_layers),
            "compression": summary.get("compression"),
            "layers": [
                {
                    "layer": layer.get("layer"),
                    "item_count": layer.get("item_count"),
                    "omitted_count": layer.get("omitted_count"),
                    "buckets": layer.get("buckets", []),
                }
                for layer in raw_layers
                if isinstance(layer, dict) and layer.get("layer") in reference_layers
            ],
        }
        raw_selected_layers = summary.get("selected_layers", [])
        if not isinstance(raw_selected_layers, list):
            raw_selected_layers = []
        return ActiveContextFrame(
            active_text=self.render_active_frame(),
            reference_summary=reference_summary,
            audit=summary,
            selected_layers=tuple(str(layer) for layer in raw_selected_layers),
            reference_only_layers=reference_layers,
        )

    def summary(self) -> dict:
        """Return a compact summary dict for logging."""
        return self.context_summary()
