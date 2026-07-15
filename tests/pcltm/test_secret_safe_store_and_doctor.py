from __future__ import annotations

from pcltm.doctor import PersonaLCMDoctor
from pcltm.store import EventStore


def test_store_add_memory_record_rejects_raw_secret(tmp_path) -> None:
    store = EventStore(tmp_path / "pcltm.db")
    fake_secret = "PASSWORD=hunter2"

    record_id, created = store.add_memory_record(
        candidate_id="raw-secret",
        kind="memory_note",
        target_file="MEMORY.md",
        content=f"remember {fake_secret}",
        confidence=1.0,
        sensitivity="normal",
        status="approved",
    )

    assert created is True
    record = store.list_candidate_queue()[0]
    assert record["record_id"] == record_id
    assert record["status"] == "rejected"
    assert record["sensitivity"] == "secret"
    assert fake_secret not in record["content"]
    assert record["metadata"]["rejected_raw_secret"] is True


def test_store_add_memory_record_sanitizes_mixed_connection_secret(tmp_path) -> None:
    store = EventStore(tmp_path / "pcltm.db")

    record_id, created = store.add_memory_record(
        candidate_id="mixed-secret",
        kind="memory_note",
        target_file="MEMORY.md",
        content="SSH host=203.0.113.10 user=ubuntu password=hunter2 path=/srv/app",
        confidence=1.0,
        sensitivity="normal",
        status="approved",
    )

    assert created is True
    record = [item for item in store.list_candidate_queue() if item["record_id"] == record_id][0]
    assert record["status"] == "approved"
    assert "203.0.113.10" in record["content"]
    assert "ubuntu" in record["content"]
    assert "/srv/app" in record["content"]
    assert "hunter2" not in record["content"]
    assert record["metadata"]["sanitized_from_secret"] is True


def test_doctor_reports_approved_memory_secret_without_value(tmp_path) -> None:
    store = EventStore(tmp_path / "pcltm.db")
    # Seed a legacy dirty row directly to simulate pre-policy data.
    store._conn.execute(
        """
        INSERT INTO memory_records (
            candidate_id, kind, target_file, content, confidence, sensitivity,
            source_event_ids, source_node_ids, status, metadata
        ) VALUES ('legacy-dirty', 'memory_note', 'MEMORY.md', 'Legacy PASSWORD=hunter2 value', 1.0, 'normal', '[]', '[]', 'approved', '{}')
        """
    )
    store._conn.commit()

    report = PersonaLCMDoctor(store).run_checks()

    matching = [issue for issue in report["issues"] if issue["code"] == "approved_memory_contains_secret"]
    assert matching
    assert matching[0]["record_id"]
    assert "PASSWORD=hunter2" not in str(matching)
    assert "secret_assignment" in matching[0]["categories"]


def test_doctor_reports_pending_memory_secret_without_value(tmp_path) -> None:
    store = EventStore(tmp_path / "pcltm.db")
    store._conn.execute(
        """
        INSERT INTO memory_records (
            candidate_id, kind, target_file, content, confidence, sensitivity,
            source_event_ids, source_node_ids, status, metadata
        ) VALUES ('pending-dirty', 'memory_note', 'MEMORY.md', 'Pending PASSWORD=hunter2 value', 1.0, 'normal', '[]', '[]', 'pending', '{}')
        """
    )
    store._conn.commit()

    report = PersonaLCMDoctor(store).run_checks()

    matching = [issue for issue in report["issues"] if issue["code"] == "pending_memory_contains_secret"]
    assert matching
    assert matching[0]["record_id"]
    assert "PASSWORD=hunter2" not in str(matching)
    assert "secret_assignment" in matching[0]["categories"]
