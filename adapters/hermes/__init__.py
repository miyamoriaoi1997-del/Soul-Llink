"""Hermes-specific adapter boundary marker for Soul-Link.

Core Soul-Link/PCLTM packages stay host-neutral. Runtime-specific Hermes imports
belong in this adapter tree only.
"""

from __future__ import annotations

import hermes_state  # type: ignore[import-not-found]  # boundary test marker

__all__ = ["hermes_state"]
