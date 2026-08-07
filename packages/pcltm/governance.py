"""PCLTM memory governance orchestration.

This module ties together the MemFS layer, pending memory candidates,
reflection drafts, and defrag analysis into one dry-run governance report.
It is intentionally conservative: analysis is read-only by default and never
approves candidates, writes reflection drafts, edits system memory, or applies
defrag actions unless a future caller explicitly opts into those operations.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field
from typing import Any

from .defrag import DefragPlan, MemFSDefragger
from .memfs_store import MemFSStore
from .reflection import MemoryDraft, ReflectionCandidateBuilder, ReflectionWriter


@dataclass(frozen=True)
class GovernanceAction:
    """A proposed memory-governance action."""

    kind: str
    reason: str
    target: str = ""
    requires_human_review: bool = True
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "kind": self.kind,
            "reason": self.reason,
            "target": self.target,
            "requires_human_review": self.requires_human_review,
            "metadata": dict(self.metadata),
        }


@dataclass(frozen=True)
class MemoryLifecycleTransition:
    """Typed lifecycle edge for candidate/reflection/defrag governance.

    The transition is a control-plane object only.  It describes the current
    state, proposed next state, authority boundary, and risk; it does not apply
    the transition by itself.
    """

    transition_id: str
    object_id: str
    object_kind: str
    current_state: str
    proposed_state: str
    trigger: str
    authority_boundary: str
    risk_level: str
    requires_human_review: bool = True
    evidence_refs: tuple[dict[str, Any], ...] = ()
    metadata: dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_action(cls, action: GovernanceAction) -> "MemoryLifecycleTransition":
        metadata = dict(action.metadata)
        object_id = str(
            metadata.get("candidate_id")
            or metadata.get("record_id")
            or metadata.get("source_path")
            or metadata.get("target_path")
            or action.target
            or action.kind
        )
        object_kind = cls._object_kind(action)
        current_state = cls._current_state(action)
        proposed_state = cls._proposed_state(action)
        authority_boundary = cls._authority_boundary(action, object_kind)
        risk_level = cls._risk_level(action, authority_boundary)
        evidence_refs = cls._evidence_refs(action)
        transition_id = cls._transition_id(
            object_id=object_id,
            object_kind=object_kind,
            current_state=current_state,
            proposed_state=proposed_state,
            trigger=action.kind,
            authority_boundary=authority_boundary,
            metadata=metadata,
        )
        return cls(
            transition_id=transition_id,
            object_id=object_id,
            object_kind=object_kind,
            current_state=current_state,
            proposed_state=proposed_state,
            trigger=action.kind,
            authority_boundary=authority_boundary,
            risk_level=risk_level,
            requires_human_review=action.requires_human_review,
            evidence_refs=evidence_refs,
            metadata=metadata,
        )

    @staticmethod
    def _object_kind(action: GovernanceAction) -> str:
        if "candidate" in action.kind:
            return "memory_candidate"
        if "reflection" in action.kind:
            return "reflection_draft"
        if "defrag" in action.kind:
            return "memfs_file"
        return "memory_object"

    @staticmethod
    def _current_state(action: GovernanceAction) -> str:
        metadata = dict(action.metadata)
        if action.kind == "review_memory_candidate":
            return str(metadata.get("status") or "pending")
        if action.kind == "approve_memory_candidate":
            return "pending"
        if action.kind == "write_reflection_draft":
            return "draft_candidate"
        if action.kind.startswith("defrag_"):
            return "active_file"
        if action.kind.startswith("apply_defrag_"):
            return "proposed_file_change"
        if action.kind == "blocked_system_defrag":
            return "blocked"
        return "unknown"

    @staticmethod
    def _proposed_state(action: GovernanceAction) -> str:
        if action.kind == "review_memory_candidate":
            return "review_required"
        if action.kind == "approve_memory_candidate":
            return "approved"
        if action.kind == "write_reflection_draft":
            return "review_required" if action.requires_human_review else "draft_written"
        if action.kind.startswith("defrag_"):
            return "defrag_review_required"
        if action.kind.startswith("apply_defrag_"):
            return "applied"
        if action.kind == "blocked_system_defrag":
            return "blocked_until_explicit_authority"
        return "review_required" if action.requires_human_review else "executed"

    @staticmethod
    def _authority_boundary(action: GovernanceAction, object_kind: str) -> str:
        metadata = dict(action.metadata)
        target = str(action.target or metadata.get("source_path") or metadata.get("target_path") or "")
        target_layer = str(metadata.get("target_layer") or "")
        if target.startswith("system/") or target_layer == "system" or target == "SYSTEM.md":
            return "system_memory"
        if object_kind == "memory_candidate":
            return "candidate_memory"
        if object_kind == "reflection_draft":
            return "reflection_memory"
        if object_kind == "memfs_file":
            return "memfs_defrag"
        return "memory"

    @staticmethod
    def _risk_level(action: GovernanceAction, authority_boundary: str) -> str:
        if authority_boundary == "system_memory":
            return "high"
        if action.requires_human_review:
            return "medium"
        if action.kind.startswith("apply_defrag"):
            return "medium"
        return "low"

    @staticmethod
    def _evidence_refs(action: GovernanceAction) -> tuple[dict[str, Any], ...]:
        metadata = dict(action.metadata)
        refs: list[dict[str, Any]] = []
        if metadata.get("record_id") is not None:
            refs.append({"type": "memory_record", "id": metadata.get("record_id")})
        if metadata.get("candidate_id"):
            refs.append({"type": "candidate", "id": metadata.get("candidate_id")})
        if metadata.get("source_path"):
            refs.append({"type": "memfs_path", "path": metadata.get("source_path")})
        if metadata.get("target_path"):
            refs.append({"type": "memfs_path", "path": metadata.get("target_path")})
        if metadata.get("provenance"):
            refs.append({"type": "provenance", "value": metadata.get("provenance")})
        return tuple(refs)

    @staticmethod
    def _transition_id(**payload: Any) -> str:
        encoded = json.dumps(payload, sort_keys=True, separators=(",", ":"), default=str)
        return "mlt_" + hashlib.sha256(encoded.encode("utf-8")).hexdigest()[:12]

    def to_dict(self) -> dict[str, Any]:
        return {
            "transition_id": self.transition_id,
            "object_id": self.object_id,
            "object_kind": self.object_kind,
            "current_state": self.current_state,
            "proposed_state": self.proposed_state,
            "trigger": self.trigger,
            "authority_boundary": self.authority_boundary,
            "risk_level": self.risk_level,
            "requires_human_review": self.requires_human_review,
            "evidence_refs": [dict(ref) for ref in self.evidence_refs],
            "metadata": dict(self.metadata),
        }


@dataclass(frozen=True)
class MemoryLifecycleLedger:
    """Auditable lifecycle ledger for one governance pass."""

    schema_version: int = 1
    transitions: tuple[MemoryLifecycleTransition, ...] = ()
    authority_boundary: str = "read_only_governance_control_plane"

    @classmethod
    def from_actions(cls, actions: list[GovernanceAction]) -> "MemoryLifecycleLedger":
        transitions = tuple(MemoryLifecycleTransition.from_action(action) for action in actions)
        return cls(transitions=transitions)

    @property
    def transition_count(self) -> int:
        return len(self.transitions)

    @property
    def risk_level(self) -> str:
        order = {"low": 0, "medium": 1, "high": 2}
        if not self.transitions:
            return "low"
        return max((transition.risk_level for transition in self.transitions), key=lambda value: order.get(value, 0))

    @property
    def blocked_count(self) -> int:
        return sum(1 for transition in self.transitions if transition.proposed_state.startswith("blocked"))

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "authority_boundary": self.authority_boundary,
            "transition_count": self.transition_count,
            "risk_level": self.risk_level,
            "blocked_count": self.blocked_count,
            "transitions": [transition.to_dict() for transition in self.transitions],
        }


@dataclass(frozen=True)
class MemoryGovernanceReport:
    """Dry-run governance report for one memory maintenance pass."""

    dry_run: bool
    pending_candidates: int = 0
    reflection_drafts: int = 0
    defrag_plan: DefragPlan = field(default_factory=DefragPlan)
    actions: list[GovernanceAction] = field(default_factory=list)
    lifecycle_ledger: MemoryLifecycleLedger = field(default_factory=MemoryLifecycleLedger)

    @classmethod
    def build(
        cls,
        *,
        dry_run: bool,
        pending_candidates: int = 0,
        reflection_drafts: int = 0,
        defrag_plan: DefragPlan | None = None,
        actions: list[GovernanceAction] | None = None,
    ) -> "MemoryGovernanceReport":
        resolved_actions = list(actions or [])
        return cls(
            dry_run=dry_run,
            pending_candidates=pending_candidates,
            reflection_drafts=reflection_drafts,
            defrag_plan=defrag_plan or DefragPlan(),
            actions=resolved_actions,
            lifecycle_ledger=MemoryLifecycleLedger.from_actions(resolved_actions),
        )


    def to_dict(self) -> dict[str, Any]:
        defrag_actions = [
            {
                "action_type": action.action_type,
                "source_path": action.source_path,
                "target_path": action.target_path,
                "reason": action.reason,
            }
            for action in self.defrag_plan.actions
        ]
        return {
            "dry_run": self.dry_run,
            "pending_candidates": self.pending_candidates,
            "reflection_drafts": self.reflection_drafts,
            "defrag_plan": {
                "total_files_analyzed": self.defrag_plan.total_files_analyzed,
                "duplicate_count": self.defrag_plan.duplicate_count,
                "oversized_count": self.defrag_plan.oversized_count,
                "actions": defrag_actions,
            },
            "actions": [action.to_dict() for action in self.actions],
            "lifecycle_ledger": self.lifecycle_ledger.to_dict(),
        }


class MemoryGovernanceOrchestrator:
    """Coordinate PCLTM memory lifecycle analysis without side effects.

    The orchestrator is the single place where Stateful memory lifecycle
    concepts meet Soul-Link/PCLTM primitives:
    - pending ``memory_records`` are surfaced for review;
    - reflection events become reviewable drafts, not automatic writes;
    - MemFS defrag is analyzed as a dry-run plan;
    - every suggested action declares whether human review is required.
    """

    def __init__(self, *, memfs_store: MemFSStore, event_store: Any | None = None) -> None:
        self.memfs_store = memfs_store
        self.event_store = event_store

    def analyze(self, *, reflection_events: list[dict] | None = None) -> MemoryGovernanceReport:
        """Build a read-only governance report.

        This method must remain side-effect free: it does not approve records,
        write drafts, commit git changes, or apply defrag plans.
        """
        pending_records = self._pending_records()
        defrag_plan = MemFSDefragger(self.memfs_store).analyze()
        drafts = ReflectionCandidateBuilder().build_from_events(reflection_events or [])

        actions: list[GovernanceAction] = []
        actions.extend(self._candidate_actions(pending_records))
        actions.extend(self._reflection_actions(drafts))
        actions.extend(self._defrag_actions(defrag_plan))

        return MemoryGovernanceReport.build(
            dry_run=True,
            pending_candidates=len(pending_records),
            reflection_drafts=len(drafts),
            defrag_plan=defrag_plan,
            actions=actions,
        )

    def execute(
        self,
        *,
        reflection_events: list[dict] | None = None,
        approve_pending: bool = False,
        write_reflection_drafts: bool = False,
        apply_defrag: bool = False,
        allow_system_defrag: bool = False,
        reviewer: str = "governance",
    ) -> MemoryGovernanceReport:
        """Execute an explicit governance pass.

        This remains conservative by default: callers must opt in to each side
        effect individually. The returned report still includes the dry-run
        analysis plus the effects that were actually performed.
        """
        report = self.analyze(reflection_events=reflection_events)
        if not approve_pending and not write_reflection_drafts and not apply_defrag:
            return report

        pending_records = self._pending_records()
        actions = list(report.actions)

        if approve_pending and self.event_store is not None and hasattr(self.event_store, "review_candidate"):
            for record in pending_records:
                record_id = int(record.get("record_id") or 0)
                if not record_id:
                    continue
                self.event_store.review_candidate(
                    record_id,
                    decision="approved",
                    reviewer=reviewer,
                    decision_reason="approved by governance execute",
                )
                actions.append(
                    GovernanceAction(
                        kind="approve_memory_candidate",
                        target=str(record.get("target_file") or ""),
                        reason="explicit governance execution approved a pending memory candidate",
                        requires_human_review=False,
                        metadata={"record_id": record_id, "candidate_id": record.get("candidate_id")},
                    )
                )

        if write_reflection_drafts and reflection_events:
            writer = ReflectionWriter(self.memfs_store)
            drafts = ReflectionCandidateBuilder().build_from_events(reflection_events)
            for draft in drafts:
                if writer.write_draft(draft, allow_system=False):
                    actions.append(
                        GovernanceAction(
                            kind="write_reflection_draft",
                            target=draft.relative_path,
                            reason="explicit governance execution wrote a reflection draft",
                            requires_human_review=False,
                            metadata={"target_layer": draft.target_layer, "provenance": draft.provenance},
                        )
                    )

        if apply_defrag:
            safe_plan = DefragPlan(
                actions=[
                    action
                    for action in report.defrag_plan.actions
                    if allow_system_defrag or not action.source_path.startswith("system/")
                ],
                total_files_analyzed=report.defrag_plan.total_files_analyzed,
                duplicate_count=report.defrag_plan.duplicate_count,
                oversized_count=report.defrag_plan.oversized_count,
            )
            blocked_actions = [
                action
                for action in report.defrag_plan.actions
                if action.source_path.startswith("system/") and not allow_system_defrag
            ]
            MemFSDefragger(self.memfs_store).apply(safe_plan)
            actions.extend(
                GovernanceAction(
                    kind="blocked_system_defrag",
                    target=action.source_path,
                    reason="system-layer MemFS changes require explicit allow_system_defrag",
                    requires_human_review=True,
                    metadata={"target_path": action.target_path, "action_type": action.action_type},
                )
                for action in blocked_actions
            )
            actions.extend(
                GovernanceAction(
                    kind=f"apply_defrag_{action.action_type}",
                    target=action.source_path,
                    reason=action.reason,
                    requires_human_review=False,
                    metadata={"target_path": action.target_path},
                )
                for action in safe_plan.actions
            )

        return MemoryGovernanceReport.build(
            dry_run=False,
            pending_candidates=report.pending_candidates,
            reflection_drafts=report.reflection_drafts,
            defrag_plan=report.defrag_plan,
            actions=actions,
        )

    def _pending_records(self) -> list[dict[str, Any]]:
        if self.event_store is None or not hasattr(self.event_store, "list_memory_records"):
            return []
        try:
            records = self.event_store.list_memory_records(status="pending")
        except TypeError:
            records = [row for row in self.event_store.list_memory_records() if row.get("status") == "pending"]
        return [dict(row) for row in records]

    def _candidate_actions(self, records: list[dict[str, Any]]) -> list[GovernanceAction]:
        actions: list[GovernanceAction] = []
        for record in records:
            record_id = record.get("record_id")
            target_file = str(record.get("target_file") or "")
            candidate_id = str(record.get("candidate_id") or record_id or "")
            actions.append(
                GovernanceAction(
                    kind="review_memory_candidate",
                    target=target_file,
                    reason="pending memory record requires explicit review before becoming active memory",
                    requires_human_review=True,
                    metadata={
                        "record_id": record_id,
                        "candidate_id": candidate_id,
                        "status": record.get("status"),
                    },
                )
            )
        return actions

    def _reflection_actions(self, drafts: list[MemoryDraft]) -> list[GovernanceAction]:
        actions: list[GovernanceAction] = []
        for draft in drafts:
            actions.append(
                GovernanceAction(
                    kind="write_reflection_draft",
                    target=draft.relative_path,
                    reason="reflection draft should be written only after governance approval",
                    requires_human_review=True,
                    metadata={
                        "target_layer": draft.target_layer,
                        "authority": draft.frontmatter.authority,
                        "provenance": draft.provenance,
                    },
                )
            )
        return actions

    def _defrag_actions(self, plan: DefragPlan) -> list[GovernanceAction]:
        actions: list[GovernanceAction] = []
        for action in plan.actions:
            actions.append(
                GovernanceAction(
                    kind=f"defrag_{action.action_type}",
                    target=action.source_path,
                    reason=action.reason or "MemFS defrag action proposed by dry-run analysis",
                    requires_human_review=True,
                    metadata={
                        "source_path": action.source_path,
                        "target_path": action.target_path,
                        "action_type": action.action_type,
                    },
                )
            )
        return actions


__all__ = [
    "GovernanceAction",
    "MemoryGovernanceReport",
    "MemoryGovernanceOrchestrator",
    "MemoryLifecycleLedger",
    "MemoryLifecycleTransition",
]
