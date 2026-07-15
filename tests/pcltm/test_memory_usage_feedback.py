from __future__ import annotations

from pcltm.memory_feedback import (
    MemoryFeedbackSignal,
    MemoryUsageFeedbackReport,
    MemoryUsageFeedbackRecorder,
)
from pcltm.memfs_types import MemoryLayerItem, MemoryLayerView, PromptMemoryView


def test_usage_feedback_report_marks_cited_and_uncited_selected_records() -> None:
    view = PromptMemoryView(
        pinned=MemoryLayerView(
            layer="pinned",
            items=[
                MemoryLayerItem(
                    path="pinned/pref.md",
                    id="pref-1",
                    body="用户偏好 Soul-Link 统一 WebUI",
                    metadata={"record_id": 101, "memory_type": "UserPreference"},
                ),
                MemoryLayerItem(
                    path="pinned/stale.md",
                    id="pref-2",
                    body="旧任务状态：继续追查昨天的临时日志",
                    metadata={"record_id": 102, "memory_type": "TemporaryTaskState"},
                ),
            ],
        )
    )

    report = MemoryUsageFeedbackRecorder().analyze_response(
        memory_view=view,
        response_text="我会把 Soul-Link WebUI 做成统一控制台，不单独做记忆页面。",
        user_message="可以",
        mode="work",
    )

    assert report.schema_version == 1
    assert report.authority_boundary == "read_only_usage_feedback"
    assert report.selected_record_ids == (101, 102)
    assert report.used_record_ids == (101,)
    assert report.unused_record_ids == (102,)
    assert report.signals[0].record_id == 101
    assert report.signals[0].signal == "used_in_response"
    assert report.signals[0].suggested_adjustment == "stabilize"
    assert report.signals[1].record_id == 102
    assert report.signals[1].signal == "selected_but_unused"
    assert report.signals[1].suggested_adjustment == "observe_decay"
    serialized = report.to_dict()
    assert serialized["used_record_ids"] == [101]
    assert serialized["signals"][0]["memory_type"] == "UserPreference"


def test_usage_feedback_detects_user_correction_and_mode_mismatch() -> None:
    view = PromptMemoryView(
        pinned=MemoryLayerView(
            layer="pinned",
            items=[
                MemoryLayerItem(
                    path="pinned/intimacy.md",
                    body="用户亲密偏好",
                    buckets=("relationship",),
                    mode_scope=("daily", "sex"),
                    metadata={"record_id": 201, "memory_type": "RelationshipAnchor"},
                )
            ],
        )
    )

    report = MemoryUsageFeedbackRecorder().analyze_response(
        memory_view=view,
        response_text="按这条记忆处理。",
        user_message="不是这样，以后别这样",
        mode="work",
    )

    signals = {(signal.signal, signal.suggested_adjustment) for signal in report.signals}
    assert ("user_corrected", "review_or_supersede") in signals
    assert ("mode_mismatch", "restrict_mode_affinity") in signals
    correction = next(signal for signal in report.signals if signal.signal == "user_corrected")
    assert correction.requires_human_review is True
    assert correction.evidence_refs[0] == {"type": "memfs_path", "path": "pinned/intimacy.md"}


def test_usage_feedback_report_can_be_built_from_explicit_signals() -> None:
    signal = MemoryFeedbackSignal(
        record_id=7,
        memory_id="mem-7",
        signal="helped",
        mode="work",
        memory_type="RuntimeInvariant",
        suggested_adjustment="promote",
        confidence=0.8,
    )

    report = MemoryUsageFeedbackReport.build(selected_record_ids=[7], signals=[signal])

    assert report.selected_record_ids == (7,)
    assert report.used_record_ids == ()
    assert report.signals == (signal,)
    assert report.to_dict()["signals"][0]["confidence"] == 0.8
