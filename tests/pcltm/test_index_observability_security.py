from __future__ import annotations

import sqlite3

import pytest

from pcltm.index_observability import _count


def test_index_count_accepts_only_internal_observability_tables() -> None:
    con = sqlite3.connect(":memory:")
    try:
        con.execute("CREATE TABLE events (event_id INTEGER PRIMARY KEY)")
        con.execute("INSERT INTO events DEFAULT VALUES")

        assert _count(con, "events") == 1
        with pytest.raises(ValueError, match="unsupported observability table"):
            _count(con, "events WHERE 1=1")
    finally:
        con.close()
