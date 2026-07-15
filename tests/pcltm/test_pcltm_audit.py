from __future__ import annotations

import sqlite3

import pcltm.pcltm_audit as pcltm_audit


def _make_db(path):
    con = sqlite3.connect(path)
    con.execute(
        """
        create table memory_records (
            record_id integer primary key,
            candidate_id text not null,
            kind text not null,
            target_file text not null,
            content text not null,
            confidence real not null,
            sensitivity text not null,
            source_event_ids text not null,
            source_node_ids text not null,
            status text not null,
            reviewer text,
            reviewed_at text,
            decision_reason text,
            patch_suggestion text,
            metadata text not null,
            created_at text not null
        )
        """
    )
    rows = [
        (1, "c1", "UserPreference", "USER.md", "approved memory", 0.9, "low", "[]", "[]", "approved", "reviewer", None, "ok", None, '{"buckets":["user_preference"]}', "2026-01-01T00:00:00"),
        (2, "c2", "UserPreference", "MEMORY.md", "pending memory one", 0.8, "low", "[]", "[]", "pending", None, None, None, None, '{"buckets":["generic"]}', "2026-01-01T00:00:01"),
        (3, "c3", "UserPreference", "MEMORY.md", "pending memory two", 0.8, "low", "[]", "[]", "pending", None, None, None, None, '{"buckets":["generic"]}', "2026-01-01T00:00:02"),
        (4, "c4", "UserPreference", "MEMORY.md", "pending memory two", 0.8, "low", "[]", "[]", "pending", None, None, None, None, '{"buckets":["generic"]}', "2026-01-01T00:00:03"),
        (5, "c5", "UserPreference", "MEMORY.md", "old memory", 0.8, "low", "[]", "[]", "superseded", None, None, None, None, "{}", "2026-01-01T00:00:04"),
    ]
    con.executemany(
        "insert into memory_records values (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
        rows,
    )
    con.commit()
    con.close()


def test_audit_summary_mode_is_bounded_and_reports_health(tmp_path, monkeypatch, capsys):
    db_path = tmp_path / "pcltm.db"
    _make_db(db_path)
    monkeypatch.setenv("HERMES_PCLTM_DB", str(db_path))

    assert pcltm_audit.audit(samples=1, pending_warn=2, pending_critical=3) == 0

    output = capsys.readouterr().out
    assert "=== GOVERNANCE HEALTH ===" in output
    assert "status: critical" in output
    assert "pending_records: 3" in output
    assert "duplicate_groups_exact: 1" in output
    assert "pending memory one" in output
    assert "pending memory two" not in output
    assert "Hint: use --full" in output


def test_audit_full_mode_preserves_complete_dump(tmp_path, monkeypatch, capsys):
    db_path = tmp_path / "pcltm.db"
    _make_db(db_path)
    monkeypatch.setenv("HERMES_PCLTM_DB", str(db_path))

    assert pcltm_audit.audit(full=True) == 0

    output = capsys.readouterr().out
    assert "=== APPROVED RECORDS ===" in output
    assert "=== PENDING RECORDS ===" in output
    assert "pending memory one" in output
    assert "pending memory two" in output
