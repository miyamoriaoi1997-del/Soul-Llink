import copy
import json
import math

from pcltm.continuity_baseline import build_continuity_baseline
from pcltm.persona import default_persona_anchor
from pcltm.session.session_summary_chain import SessionSummaryChain
from pcltm.state.active_dialogue_state import ActiveDialogueState
from soul_link.continuity import ConversationContinuitySnapshot


def _inputs():
    identity = default_persona_anchor()
    conversation = ConversationContinuitySnapshot(
        previous_session_id="session-before",
        current_session_id="session-now",
        session_key="cli:thread-1",
        should_resume=True,
        last_real_user_message="继续 SoulLink 连续性保护。",
        recent_user_messages=("建立基线。", "继续 SoulLink 连续性保护。"),
        recent_message_ids=("u1", "a1", "u2"),
        recent_tool_evidence=(
            {"tool_name": "pytest", "content": "331 passed", "authority": "evidence_only"},
        ),
    )
    active = ActiveDialogueState(
        conversation_goal="保护完全跨会话、跨时间连续性。",
        current_task="建立 Continuity Baseline V1。",
        last_user_intent="继续。",
        last_assistant_commitment="先做只读 Baseline Producer。",
        open_threads=("真实历史回放集",),
        local_constraints=("production-read-only", "no-git-commit"),
        response_mode="work",
    )
    chain = SessionSummaryChain.from_dict(
        {
            "session_id": "session-before",
            "segments": [
                {
                    "segment_id": 0,
                    "time_range": [0, 2],
                    "raw_message_refs": ["u1", "a1"],
                    "local_summary": "建立连续性保护门禁。",
                    "current_task": "建立 Continuity Baseline V1。",
                    "unresolved_items": ["真实历史回放集"],
                    "verification_notes": ["331 passed"],
                }
            ],
            "current_spine": "建立 Continuity Baseline V1。",
            "unresolved_index": ["真实历史回放集"],
            "decision_index": ["生产链路保持不变"],
            "commitment_index": ["先做只读基线"],
            "source_turn_count": 2,
        }
    )
    return identity, conversation, active, chain


def test_baseline_producer_normalizes_existing_runtime_artifacts_without_reinference():
    identity, conversation, active, chain = _inputs()

    artifact = build_continuity_baseline(
        baseline_id="continuity-baseline-v1",
        case_id="cross-session-active-task",
        identity_anchor=identity,
        conversation_snapshot=conversation,
        active_dialogue_state=active,
        session_summary_chain=chain,
        evidence_refs=("session:session-before", "test:331-passed"),
    )
    payload = artifact.to_dict()

    assert payload["schema_version"] == 1
    assert payload["object_type"] == "soul_link_continuity_baseline"
    assert payload["authority_boundary"] == "read_only_artifact_normalization"
    assert payload["baseline_id"] == "continuity-baseline-v1"
    assert payload["case_id"] == "cross-session-active-task"
    assert payload["identity"]["identity"] == "the configured persona"
    assert payload["identity"]["read_only"] is True
    assert payload["conversation"]["should_resume"] is True
    assert payload["conversation"]["last_real_user_message"] == "继续 SoulLink 连续性保护。"
    assert payload["active_dialogue"]["current_task"] == "建立 Continuity Baseline V1。"
    assert payload["active_dialogue"]["local_constraints"] == [
        "production-read-only",
        "no-git-commit",
    ]
    assert payload["summary_chain"]["segments"][0]["raw_message_refs"] == ["u1", "a1"]
    assert payload["summary_chain"]["decision_index"] == ["生产链路保持不变"]
    assert payload["evidence_refs"] == ["session:session-before", "test:331-passed"]
    json.dumps(payload, ensure_ascii=False, sort_keys=True)


def test_baseline_producer_is_deterministic_and_does_not_mutate_mapping_inputs():
    identity, conversation, active, chain = _inputs()
    identity_mapping = dict(identity.as_prompt_anchor())
    conversation_mapping = conversation.to_dict()
    active_mapping = active.to_dict()
    chain_mapping = chain.to_dict()
    before = copy.deepcopy((identity_mapping, conversation_mapping, active_mapping, chain_mapping))

    first = build_continuity_baseline(
        baseline_id="v1",
        case_id="case-1",
        identity_anchor=identity_mapping,
        conversation_snapshot=conversation_mapping,
        active_dialogue_state=active_mapping,
        session_summary_chain=chain_mapping,
        evidence_refs=("source:1", "source:1", "source:2"),
    ).to_json()
    second = build_continuity_baseline(
        baseline_id="v1",
        case_id="case-1",
        identity_anchor=identity_mapping,
        conversation_snapshot=conversation_mapping,
        active_dialogue_state=active_mapping,
        session_summary_chain=chain_mapping,
        evidence_refs=("source:1", "source:1", "source:2"),
    ).to_json()

    assert first == second
    assert (identity_mapping, conversation_mapping, active_mapping, chain_mapping) == before
    assert json.loads(first)["evidence_refs"] == ["source:1", "source:2"]


def test_baseline_producer_rejects_missing_identity_or_evidence():
    _, conversation, active, chain = _inputs()

    for identity, evidence in (({}, ("source:1",)), ({"identity": "rin"}, ())):
        try:
            build_continuity_baseline(
                baseline_id="v1",
                case_id="case-1",
                identity_anchor=identity,
                conversation_snapshot=conversation,
                active_dialogue_state=active,
                session_summary_chain=chain,
                evidence_refs=evidence,
            )
        except ValueError:
            pass
        else:
            raise AssertionError("invalid baseline input must fail closed")


def test_baseline_producer_rejects_empty_required_artifacts():
    identity, conversation, active, chain = _inputs()
    valid = {
        "identity_anchor": identity,
        "conversation_snapshot": conversation,
        "active_dialogue_state": active,
        "session_summary_chain": chain,
    }

    for field_name in (
        "conversation_snapshot",
        "active_dialogue_state",
        "session_summary_chain",
    ):
        inputs = dict(valid)
        inputs[field_name] = {}
        try:
            build_continuity_baseline(
                baseline_id="v1",
                case_id="case-1",
                evidence_refs=("source:1",),
                **inputs,
            )
        except ValueError:
            pass
        else:
            raise AssertionError(f"empty {field_name} must fail closed")


def test_baseline_producer_rejects_key_collisions_and_non_finite_floats():
    identity, conversation, _, chain = _inputs()

    for active_mapping in (
        {1: "first", "1": "second"},
        {"current_task": math.nan},
        {"current_task": math.inf},
    ):
        try:
            build_continuity_baseline(
                baseline_id="v1",
                case_id="case-1",
                identity_anchor=identity,
                conversation_snapshot=conversation,
                active_dialogue_state=active_mapping,
                session_summary_chain=chain,
                evidence_refs=("source:1",),
            )
        except ValueError:
            pass
        else:
            raise AssertionError("lossy or non-standard JSON data must fail closed")


def test_baseline_producer_rejects_non_string_ids_and_evidence_refs():
    identity, conversation, active, chain = _inputs()

    for baseline_id, case_id, evidence_refs in (
        (1, "case-1", ("source:1",)),
        ("v1", True, ("source:1",)),
        ("v1", "case-1", (1,)),
        ("v1", "case-1", "source:1"),
    ):
        try:
            build_continuity_baseline(
                baseline_id=baseline_id,
                case_id=case_id,
                identity_anchor=identity,
                conversation_snapshot=conversation,
                active_dialogue_state=active,
                session_summary_chain=chain,
                evidence_refs=evidence_refs,
            )
        except ValueError:
            pass
        else:
            raise AssertionError("non-string identifiers must fail closed")


def test_baseline_producer_requires_string_identity_and_true_read_only_anchor():
    _, conversation, active, chain = _inputs()

    for identity in (
        {"identity": True, "read_only": True},
        {"identity": 1, "read_only": True},
        {"identity": "rin", "read_only": False},
    ):
        try:
            build_continuity_baseline(
                baseline_id="v1",
                case_id="case-1",
                identity_anchor=identity,
                conversation_snapshot=conversation,
                active_dialogue_state=active,
                session_summary_chain=chain,
                evidence_refs=("source:1",),
            )
        except ValueError:
            pass
        else:
            raise AssertionError("identity authority fields must fail closed")
