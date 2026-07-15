"""Compatibility exports for Session Summary Chain.

The canonical stage-2 modules live under :mod:`pcltm.session`. This module keeps
older imports from :mod:`pcltm.state` working.
"""

from pcltm.session.session_summary_chain import (
    SessionSegment,
    SessionSummaryChain,
    SessionSummarySegment,
    append_turns_to_chain,
    build_session_summary_chain,
    is_summary_continuation_only,
    summarize_segment,
)

__all__ = [
    "SessionSegment",
    "SessionSummaryChain",
    "SessionSummarySegment",
    "append_turns_to_chain",
    "build_session_summary_chain",
    "is_summary_continuation_only",
    "summarize_segment",
]
