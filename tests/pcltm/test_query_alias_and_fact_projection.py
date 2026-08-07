from __future__ import annotations

import json
import sqlite3

from pcltm import memory_adapter
from pcltm.memfs_store import MemFSStore


def _rows(records):
    con = sqlite3.connect(":memory:")
    con.row_factory = sqlite3.Row
    con.execute("CREATE TABLE rows (record_id INTEGER, content TEXT, metadata TEXT)")
    con.executemany("INSERT INTO rows VALUES (?, ?, ?)", records)
    return con, con.execute("SELECT * FROM rows").fetchall()


def test_query_aliases_are_data_driven_and_improve_work_preference_recall():
    policy = memory_adapter.ViewPolicy(query_alias_groups=(("改稿", "修订", "精修"), ("写作", "小说", "创作")))
    con, rows = _rows([
        (1, "用户偏好普通简洁回复。", json.dumps({"gov_score": 0.95})),
        (2, "用户在小说创作中偏好专项修订并保持人物一致性。", json.dumps({"gov_score": 0.40})),
    ])
    try:
        ranked = memory_adapter._rank_rows("USER.md", rows, "work", query="写作时怎么改稿", policy=policy)
    finally:
        con.close()

    assert ranked[0]["record_id"] == 2


def test_structured_fact_projection_participates_in_recall_without_mutating_body():
    con, rows = _rows([
        (1, "用户偏好普通简洁回复。", json.dumps({"gov_score": 0.90})),
        (2, "用户有一条具体工作偏好。", json.dumps({
            "gov_score": 0.35,
            "facts": {"workflow": "角色一致性 场景连续性 专项修订"},
        }, ensure_ascii=False)),
    ])
    try:
        ranked = memory_adapter._rank_rows("USER.md", rows, "work", query="场景连续性", policy=memory_adapter.ViewPolicy())
    finally:
        con.close()

    assert ranked[0]["record_id"] == 2
    assert ranked[0]["content"] == "用户有一条具体工作偏好。"


def test_memfs_query_aliases_and_fact_projection_match_db_recall_semantics(tmp_path):
    root = tmp_path / "memfs"
    for name, description, body, facts in (
        ("generic.md", "普通偏好", "用户偏好简洁回复。", {}),
        ("work.md", "具体工作偏好", "用户有一条具体工作偏好。", {"workflow": "角色一致性 场景连续性 专项修订"}),
    ):
        path = root / "pinned" / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(
            "---\n"
            f"description: {description}\n"
            "authority: pinned\n"
            "mode_scope: [work]\n"
            "buckets: [user_preference]\n"
            "memory_type: UserPreference\n"
            "lifecycle_state: active\n"
            f"metadata: {json.dumps({'facts': facts}, ensure_ascii=False)}\n"
            "---\n\n"
            f"{body}\n",
            encoding="utf-8",
        )

    store = MemFSStore(root, query_alias_groups=(("改稿", "修订", "精修"),))
    alias_view = store.load_layer("pinned", mode="work", query="怎么改稿", budget_chars=1000)
    facts_view = store.load_layer("pinned", mode="work", query="场景连续性", budget_chars=1000)

    assert alias_view.items[0].path == "pinned/work.md"
    assert facts_view.items[0].path == "pinned/work.md"
    assert facts_view.items[0].body.strip() == "用户有一条具体工作偏好。"
