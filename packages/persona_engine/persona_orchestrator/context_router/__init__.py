"""Context-aware multi-turn router for persona state orchestration.

This module provides belief-based routing with:
- Multi-signal extraction (abstract tags only)
- Belief score computation (probabilistic, not hard classification)
- Policy gate (anti-flap, boundary priority, desire gate)
- Transition management (hold/switch/downgrade/reference_only)

Integration modes:
- shadow_only: observe and log, no override
- record_advice: write shadow advice to state packet
- candidate_write: may set secondary_candidate
- guard_override: may veto obvious misclassifications
- direct_switch: full authority (requires explicit activation)
"""

from .types import (
    ContextSignals,
    BeliefScores,
    ContextRouteResult,
    ContextRouterConfig,
    IntegrationLevel,
)
from .router import ContextRouter
from .adapter import AbstractStateAdapter

__all__ = [
    "ContextSignals",
    "BeliefScores",
    "ContextRouteResult",
    "ContextRouterConfig",
    "IntegrationLevel",
    "ContextRouter",
    "AbstractStateAdapter",
]
