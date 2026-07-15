from datetime import UTC, datetime, timedelta

from pcltm.memory import (
    SemanticNamespace,
    SemanticStore,
    SemanticWriter,
    Stability,
    TemporalFact,
    WriteDecision,
    make_request,
)


def test_semantic_writer_saves_long_term_preference_with_provenance(tmp_path):
    writer = SemanticWriter(SemanticStore(tmp_path / "semantic.sqlite3"))

    result = writer.add(
        make_request(
            subject="user",
            predicate="prefers_response_style",
            object="concise, evidence-backed engineering updates",
            namespace=SemanticNamespace.USER_PREFERENCE,
            source_refs=("session:42:msg:7",),
            write_reason="user explicitly requested stable response style",
            confidence=0.92,
            stability=Stability.HIGH,
            explicit=True,
        )
    )

    assert result.decision == WriteDecision.ACCEPTED
    assert result.fact is not None
    assert result.fact.namespace == "user_preference"
    assert result.fact.source_refs == ("session:42:msg:7",)
    assert "explicitly requested" in result.fact.write_reason

    found = writer.search(subject="user", predicate="prefers_response_style")
    assert [fact.object for fact in found] == ["concise, evidence-backed engineering updates"]


def test_semantic_memory_supersedes_past_truth_without_erasing_history(tmp_path):
    writer = SemanticWriter(SemanticStore(tmp_path / "semantic.sqlite3"))
    old_time = datetime(2026, 1, 1, tzinfo=UTC)
    new_time = datetime(2026, 6, 1, tzinfo=UTC)

    old = writer.add(
        make_request(
            subject="user",
            predicate="preferred_language",
            object="English",
            namespace=SemanticNamespace.USER_PREFERENCE,
            source_refs=("session:1:msg:1",),
            write_reason="user explicitly stated a language preference",
            confidence=0.8,
            stability=Stability.HIGH,
            valid_from=old_time,
            explicit=True,
        )
    )
    assert old.fact is not None

    new = writer.add(
        make_request(
            subject="user",
            predicate="preferred_language",
            object="Chinese",
            namespace=SemanticNamespace.USER_PREFERENCE,
            source_refs=("session:9:msg:3",),
            write_reason="user explicitly changed the previously true language preference",
            confidence=0.95,
            stability=Stability.HIGH,
            valid_from=new_time,
            explicit=True,
        )
    )

    assert new.decision == WriteDecision.ACCEPTED
    assert new.fact is not None
    assert new.resolution is not None
    assert new.resolution.superseded == (old.fact.memory_id,)

    historical_old = writer.store.get(old.fact.memory_id)
    assert historical_old is not None
    assert historical_old.valid_until is not None
    assert historical_old.superseded_by == new.fact.memory_id
    assert new.fact.supersedes == (old.fact.memory_id,)

    active = writer.search(subject="user", predicate="preferred_language")
    assert [fact.object for fact in active] == ["Chinese"]

    all_versions = writer.store.search(
        subject="user",
        predicate="preferred_language",
        active_only=False,
    )
    assert {fact.object for fact in all_versions} == {"English", "Chinese"}


def test_semantic_writer_keeps_unresolved_conflict_when_authority_is_lower(tmp_path):
    writer = SemanticWriter(SemanticStore(tmp_path / "semantic.sqlite3"))
    old = writer.add(
        make_request(
            subject="project",
            predicate="default_test_runner",
            object="pytest",
            namespace=SemanticNamespace.PROJECT_FACT,
            source_refs=("docs:test-policy",),
            write_reason="verified project testing policy",
            confidence=0.95,
            stability=Stability.VERIFIED,
            verified=True,
        )
    )
    assert old.fact is not None

    lower = writer.add(
        make_request(
            subject="project",
            predicate="default_test_runner",
            object="nose",
            namespace=SemanticNamespace.PROJECT_FACT,
            source_refs=("session:rumor",),
            write_reason="single weaker observation conflicts with verified policy",
            confidence=0.55,
            stability=Stability.LOW,
            repeated_observation=True,
        )
    )

    assert lower.decision == WriteDecision.ACCEPTED
    assert lower.resolution is not None
    assert lower.resolution.superseded == ()
    assert lower.resolution.reason == "candidate_kept_in_conflict_group_without_supersession"

    old_after = writer.store.get(old.fact.memory_id)
    assert old_after is not None
    assert old_after.valid_until is None
    assert old_after.superseded_by is None
    group = old.fact.conflict_group
    assert group is not None
    assert len(writer.store.list_conflicts(group)) == 2


def test_semantic_writer_rejects_temporary_task_state_pollution(tmp_path):
    writer = SemanticWriter(SemanticStore(tmp_path / "semantic.sqlite3"))

    result = writer.add(
        make_request(
            subject="current task",
            predicate="status",
            object="PR #123 is almost done today",
            namespace=SemanticNamespace.PROJECT_FACT,
            source_refs=("session:now:msg:4",),
            write_reason="temporary current task update from this week",
            confidence=0.9,
            stability=Stability.HIGH,
            explicit=True,
        )
    )

    assert result.decision == WriteDecision.REJECTED
    assert result.reason == "semantic_memory_rejects_temporary_or_task_state"
    assert writer.search(active_only=False) == []


def test_semantic_writer_requires_verified_environment_facts(tmp_path):
    writer = SemanticWriter(SemanticStore(tmp_path / "semantic.sqlite3"))

    rejected = writer.add(
        make_request(
            subject="host",
            predicate="has_binary",
            object="docker is installed",
            namespace=SemanticNamespace.ENVIRONMENT_FACT,
            source_refs=("user:claim",),
            write_reason="unverified environment claim",
            confidence=0.8,
            stability=Stability.HIGH,
            explicit=True,
        )
    )
    assert rejected.decision == WriteDecision.REJECTED
    assert rejected.reason == "environment_fact_requires_verification"

    accepted = writer.add(
        make_request(
            subject="host",
            predicate="has_binary",
            object="python is installed",
            namespace=SemanticNamespace.ENVIRONMENT_FACT,
            source_refs=("terminal:python --version",),
            write_reason="verified by command output",
            confidence=0.99,
            stability=Stability.VERIFIED,
            verified=True,
        )
    )
    assert accepted.decision == WriteDecision.ACCEPTED
    assert accepted.fact is not None
    assert accepted.fact.last_verified_at is not None


def test_temporal_fact_round_trip_and_delete(tmp_path):
    store = SemanticStore(tmp_path / "semantic.sqlite3")
    fact = TemporalFact(
        memory_id="sem_test",
        subject="persona",
        predicate="identity_anchor",
        object="stable identity is preserved across modes",
        confidence=0.9,
        valid_from=datetime.now(UTC) - timedelta(days=1),
        source_refs=("doc:core",),
        stability=Stability.HIGH,
        namespace=SemanticNamespace.PERSONA_FACT,
        conflict_group="persona::identity_anchor",
        write_reason="project governance requires identity anchors to be explicit",
    )

    store.add(fact)
    loaded = store.get("sem_test")
    assert loaded == fact
    assert store.delete("sem_test") is True
    assert store.get("sem_test") is None
