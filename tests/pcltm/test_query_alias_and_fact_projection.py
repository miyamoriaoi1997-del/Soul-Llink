from __future__ import annotations

import json
import sqlite3
from pathlib import Path

from pcltm import memory_adapter
from pcltm.memfs_store import MemFSStore


def _rows(records):
    con = sqlite3.connect(":memory:")
    con.row_factory = sqlite3.Row
    con.execute("CREATE TABLE rows (record_id INTEGER, content TEXT, metadata TEXT)")
    con.executemany("INSERT INTO rows VALUES (?, ?, ?)", records)
    return con, con.execute("SELECT * FROM rows").fetchall()


def test_query_aliases_are_data_driven_for_db_recall():
    policy = memory_adapter.ViewPolicy(query_alias_groups=(("revise", "edit", "polish"),))
    con, rows = _rows([
        (1, "General preference.", json.dumps({"gov_score": 0.95})),
        (2, "Prefers careful editing and revision.", json.dumps({"gov_score": 0.40})),
    ])
    try:
        ranked = memory_adapter._rank_rows("USER.md", rows, "work", query="polish", policy=policy)
    finally:
        con.close()
    assert ranked[0]["record_id"] == 2


def test_structured_fact_projection_ranks_without_mutating_body():
    con, rows = _rows([
        (1, "General preference.", json.dumps({"gov_score": 0.90})),
        (2, "Specific preference.", json.dumps({"gov_score": 0.35, "facts": {"workflow": "scene continuity"}})),
    ])
    try:
        ranked = memory_adapter._rank_rows("USER.md", rows, "work", query="scene continuity")
    finally:
        con.close()
    assert ranked[0]["record_id"] == 2
    assert ranked[0]["content"] == "Specific preference."


def test_memfs_alias_and_fact_projection_match_db_contract(tmp_path: Path):
    path = tmp_path / "pinned" / "workflow.md"
    path.parent.mkdir(parents=True)
    path.write_text(
        "---\ndescription: workflow\nauthority: pinned\nmode_scope: [work]\n"
        "buckets: [project]\nfacts: {workflow: scene continuity}\n---\nSpecific preference.\n",
        encoding="utf-8",
    )
    store = MemFSStore(tmp_path, query_alias_groups=(("revise", "polish"),))
    results = store.search("scene continuity", layers=("pinned",), mode="work")
    assert results[0]["memory_id"] == "pinned/workflow.md"
