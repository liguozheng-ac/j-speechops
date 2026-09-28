from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path

import pytest

from j_speech_ops.generated_audio_qa import QADecision, QAExecutionStatus
from j_speech_ops.human_review_release import (
    HUMAN_REVIEW_POLICY_VERSION,
    RELEASE_POLICY_VERSION,
    HumanReviewDecision,
    HumanReviewPolicy,
    HumanReviewQueueItem,
    HardBlockerCode,
    MachineReviewIssue,
    MachineReviewIssueSource,
    ReleaseGateDecision,
    ReleaseGateReason,
    ReviewFreshness,
    ReviewPriority,
    ReviewReasonCode,
    ReworkTarget,
    build_release_manifest,
    evaluate_release_gate,
    human_review_policy_fingerprint,
    iter_human_review_records,
    iter_release_gate_results,
    iter_released_audio,
    record_human_review,
)
from j_speech_ops.human_review_release_cli import build_parser
from j_speech_ops.pronunciation_regression import (
    PronunciationEvidenceDecision,
    RegressionDecision,
)
from j_speech_ops.tts_synthesis import AudioPostProcessingProvenance


HASH_A = "a" * 64
HASH_B = "b" * 64
HASH_C = "c" * 64
HASH_D = "d" * 64
HASH_E = "e" * 64
HASH_F = "f" * 64
REVIEWED_AT = datetime(2026, 9, 21, 12, 0, tzinfo=timezone.utc)
RELEASED_AT = datetime(2026, 9, 21, 13, 0, tzinfo=timezone.utc)


def make_queue(**updates) -> HumanReviewQueueItem:
    payload = {
        "queue_item_id": HASH_A,
        "logical_case_id": "CASE_JR",
        "sample_id": "CASE_JR",
        "priorities": [ReviewPriority.MANDATORY_FINAL_REVIEW],
        "artifact_run_id": HASH_B,
        "artifact_sha256": HASH_C,
        "audio_path": "D:/audio/jr.wav",
        "audio_post_processing": AudioPostProcessingProvenance(
            duration_ms=300,
            policy_fingerprint_sha256=HASH_F,
        ),
        "artifact_integrity_valid": True,
        "hard_blocker_codes": [],
        "machine_review_issues": [],
        "source_text": "JR新宿駅までご案内いたします。",
        "normalized_text": "JR新宿駅までご案内いたします。",
        "synthesis_input": "ジェイアール新宿駅までご案内いたします。",
        "expected_reading": "ジェイアールシンジュクエキマデゴアンナイイタシマス。",
        "expected_reference_ids": [HASH_D],
        "pronunciation_risk_ids": [HASH_E],
        "asr_transcript": "JR新宿駅までご案内いたします",
        "observed_canonical_reading": "ジェイアールシンジュクエキマデゴアンナイイタシマス",
        "stage9_qa_id": HASH_D,
        "stage9_status": QAExecutionStatus.COMPLETED,
        "stage9_decision": QADecision.PASS,
        "stage9_issue_codes": [],
        "stage9_policy_fingerprint_sha256": HASH_E,
        "stage10a_evidence_id": HASH_E,
        "stage10a_decision": PronunciationEvidenceDecision.NO_REVIEW_NEEDED,
        "stage10a_policy_fingerprint_sha256": HASH_F,
        "regression_id": HASH_F,
        "regression_decision": RegressionDecision.NO_REGRESSION_DETECTED,
        "regression_baseline_id": HASH_A,
        "regression_policy_fingerprint_sha256": HASH_F,
        "tts_backend": "qwen3-tts",
        "tts_model_name": "Qwen3-TTS-12Hz-1.7B-CustomVoice",
        "speaker": "Ono_Anna",
        "synthesis_config_fingerprint_sha256": HASH_A,
        "override_fingerprint_sha256": HASH_B,
        "review_policy_version": HUMAN_REVIEW_POLICY_VERSION,
        "review_policy_fingerprint_sha256": human_review_policy_fingerprint(
            HumanReviewPolicy()
        ),
        "review_context_fingerprint_sha256": HASH_C,
    }
    payload.update(updates)
    return HumanReviewQueueItem.model_validate(payload)


def write_models(path: Path, items) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "".join(item.model_dump_json() + "\n" for item in items), encoding="utf-8"
    )


def make_issue(
    issue_id: str,
    source: MachineReviewIssueSource,
    code: str,
) -> MachineReviewIssue:
    return MachineReviewIssue(
        issue_id=issue_id,
        source=source,
        code=code,
        evidence_reference_id=f"fixture:{code}",
        summary=f"Fixture machine review issue: {code}",
    )


def setup_queue(tmp_path: Path, item: HumanReviewQueueItem | None = None):
    queue = tmp_path / "human_review_queue.jsonl"
    records = tmp_path / "human_review_records.jsonl"
    write_models(queue, [item or make_queue()])
    records.write_text("", encoding="utf-8")
    return queue, records


def add_review(
    queue: Path,
    records: Path,
    decision: HumanReviewDecision,
    *,
    reasons=(),
    resolved=(),
    rework_target=None,
):
    return record_human_review(
        queue_manifest=queue,
        records_manifest=records,
        artifact_run_id=HASH_B,
        reviewer_id="fixture-reviewer",
        decision=decision,
        reason_codes=reasons,
        resolved_issue_ids=resolved,
        notes="test fixture decision; not a real review",
        rework_target=rework_target,
        reviewed_at=REVIEWED_AT,
    )


def evaluate(tmp_path: Path, queue: Path, records: Path):
    outputs = evaluate_release_gate(
        queue_manifest=queue,
        records_manifest=records,
        output_root=tmp_path / "release",
    )
    result = list(iter_release_gate_results(outputs.results_manifest))[0]
    return outputs, result


def test_valid_human_approve_is_eligible_for_release(tmp_path: Path) -> None:
    queue, records = setup_queue(tmp_path)
    review = add_review(queue, records, HumanReviewDecision.APPROVE)
    outputs, result = evaluate(tmp_path, queue, records)
    assert result.decision is ReleaseGateDecision.RELEASED
    assert result.review_freshness is ReviewFreshness.CURRENT
    assert result.review_id == review.review_id
    assert result.reason_codes == ()
    assert outputs.summary.released == 1


@pytest.mark.parametrize(
    ("decision", "reason", "gate_reason"),
    [
        (
            HumanReviewDecision.REWORK,
            ReviewReasonCode.PRONUNCIATION,
            ReleaseGateReason.HUMAN_REWORK,
        ),
        (
            HumanReviewDecision.REJECT,
            ReviewReasonCode.NATURALNESS,
            ReleaseGateReason.HUMAN_REJECT,
        ),
    ],
)
def test_rework_and_reject_deny_release(
    tmp_path: Path,
    decision: HumanReviewDecision,
    reason: ReviewReasonCode,
    gate_reason: ReleaseGateReason,
) -> None:
    issue = make_issue(
        HASH_A, MachineReviewIssueSource.STAGE9_CONTENT_QA, "content_mismatch"
    )
    item = make_queue(
        stage9_decision=QADecision.REVIEW,
        stage9_issue_codes=("content_mismatch",),
        machine_review_issues=(issue,),
        priorities=(
            ReviewPriority.MANDATORY_FINAL_REVIEW,
            ReviewPriority.CONTENT_REVIEW,
        ),
    )
    queue, records = setup_queue(tmp_path, item)
    add_review(
        queue,
        records,
        decision,
        reasons=(reason,),
        rework_target=(
            ReworkTarget.PRONUNCIATION_OVERRIDE
            if decision is HumanReviewDecision.REWORK
            else None
        ),
    )
    outputs, result = evaluate(tmp_path, queue, records)
    assert result.decision is ReleaseGateDecision.NOT_RELEASED
    assert gate_reason in result.reason_codes
    assert outputs.summary.released == 0


def test_machine_pass_without_human_review_is_not_released(tmp_path: Path) -> None:
    queue, records = setup_queue(tmp_path)
    outputs, result = evaluate(tmp_path, queue, records)
    assert result.review_freshness is ReviewFreshness.MISSING
    assert result.decision is ReleaseGateDecision.NOT_RELEASED
    assert result.reason_codes == (ReleaseGateReason.MISSING_HUMAN_REVIEW,)
    assert outputs.summary.missing_reviews == 1


@pytest.mark.parametrize(
    "changed_field",
    [
        "artifact_sha256",
        "artifact_run_id",
        "source_text",
        "synthesis_input",
        "override_fingerprint_sha256",
        "stage9_qa_id",
        "stage10a_evidence_id",
        "regression_id",
        "review_policy_fingerprint_sha256",
    ],
)
def test_artifact_or_evidence_change_makes_previous_review_stale(
    tmp_path: Path, changed_field: str
) -> None:
    original = make_queue()
    queue, records = setup_queue(tmp_path, original)
    add_review(queue, records, HumanReviewDecision.APPROVE)
    payload = original.model_dump(mode="json")
    payload[changed_field] = HASH_D if payload[changed_field] != HASH_D else HASH_F
    payload["queue_item_id"] = HASH_F
    payload["review_context_fingerprint_sha256"] = HASH_E
    changed = HumanReviewQueueItem.model_validate(payload)
    write_models(queue, [changed])
    outputs, result = evaluate(tmp_path, queue, records)
    assert result.review_freshness is ReviewFreshness.STALE
    assert result.decision is ReleaseGateDecision.NOT_RELEASED
    assert ReleaseGateReason.STALE_HUMAN_REVIEW in result.reason_codes
    assert outputs.summary.stale_reviews == 1


def test_artifact_integrity_hard_blocker_cannot_be_approved(tmp_path: Path) -> None:
    item = make_queue(
        artifact_integrity_valid=False,
        hard_blocker_codes=(HardBlockerCode.ARTIFACT_INTEGRITY_INVALID,),
    )
    queue, records = setup_queue(tmp_path, item)
    add_review(queue, records, HumanReviewDecision.APPROVE)
    _, result = evaluate(tmp_path, queue, records)
    assert result.decision is ReleaseGateDecision.NOT_RELEASED
    assert ReleaseGateReason.ARTIFACT_INTEGRITY_INVALID in result.reason_codes


def test_stage9_review_with_explicit_human_resolution_can_release(tmp_path: Path) -> None:
    issue = make_issue(
        HASH_A, MachineReviewIssueSource.STAGE9_CONTENT_QA, "content_mismatch"
    )
    item = make_queue(
        stage9_decision=QADecision.REVIEW,
        stage9_issue_codes=("content_mismatch",),
        machine_review_issues=(issue,),
        priorities=(
            ReviewPriority.MANDATORY_FINAL_REVIEW,
            ReviewPriority.CONTENT_REVIEW,
        ),
    )
    queue, records = setup_queue(tmp_path, item)
    add_review(
        queue,
        records,
        HumanReviewDecision.APPROVE,
        resolved=(issue.issue_id,),
    )
    _, result = evaluate(tmp_path, queue, records)
    assert result.decision is ReleaseGateDecision.RELEASED
    assert result.resolved_issue_ids == (issue.issue_id,)
    assert result.unresolved_issue_ids == ()


def test_stage9_review_without_issue_resolution_is_not_released(tmp_path: Path) -> None:
    issue = make_issue(
        HASH_A, MachineReviewIssueSource.STAGE9_CONTENT_QA, "content_mismatch"
    )
    item = make_queue(
        stage9_decision=QADecision.REVIEW,
        stage9_issue_codes=("content_mismatch",),
        machine_review_issues=(issue,),
    )
    queue, records = setup_queue(tmp_path, item)
    add_review(queue, records, HumanReviewDecision.APPROVE)
    _, result = evaluate(tmp_path, queue, records)
    assert result.decision is ReleaseGateDecision.NOT_RELEASED
    assert result.unresolved_issue_ids == (issue.issue_id,)
    assert ReleaseGateReason.UNRESOLVED_MACHINE_REVIEW_ISSUES in result.reason_codes


@pytest.mark.parametrize(
    ("source", "decision_field", "priority"),
    [
        (
            MachineReviewIssueSource.STAGE10A_PRONUNCIATION,
            "stage10a_decision",
            ReviewPriority.PRONUNCIATION_REVIEW,
        ),
        (
            MachineReviewIssueSource.STAGE10A_REGRESSION,
            "regression_decision",
            ReviewPriority.REGRESSION_REVIEW,
        ),
    ],
)
def test_stage10a_or_regression_review_can_be_resolved_by_human_approve(
    tmp_path: Path,
    source: MachineReviewIssueSource,
    decision_field: str,
    priority: ReviewPriority,
) -> None:
    issue = make_issue(HASH_A, source, "machine_review")
    item = make_queue(
        **{
            decision_field: (
                PronunciationEvidenceDecision.REVIEW
                if decision_field == "stage10a_decision"
                else RegressionDecision.REVIEW
            ),
            "machine_review_issues": (issue,),
            "priorities": (ReviewPriority.MANDATORY_FINAL_REVIEW, priority),
        }
    )
    queue, records = setup_queue(tmp_path, item)
    add_review(
        queue,
        records,
        HumanReviewDecision.APPROVE,
        resolved=(issue.issue_id,),
    )
    _, result = evaluate(tmp_path, queue, records)
    assert result.decision is ReleaseGateDecision.RELEASED


def test_stage9_hard_drop_cannot_be_overridden_by_human_approve(tmp_path: Path) -> None:
    item = make_queue(
        stage9_decision=QADecision.DROP,
        hard_blocker_codes=(HardBlockerCode.STAGE9_HARD_DROP,),
    )
    queue, records = setup_queue(tmp_path, item)
    add_review(queue, records, HumanReviewDecision.APPROVE)
    _, result = evaluate(tmp_path, queue, records)
    assert result.decision is ReleaseGateDecision.NOT_RELEASED
    assert ReleaseGateReason.STAGE9_HARD_DROP in result.reason_codes


def test_partial_machine_issue_resolution_identifies_remaining_issue(
    tmp_path: Path,
) -> None:
    issue_a = make_issue(
        HASH_A, MachineReviewIssueSource.STAGE9_CONTENT_QA, "issue_a"
    )
    issue_b = make_issue(
        HASH_B, MachineReviewIssueSource.STAGE10A_PRONUNCIATION, "issue_b"
    )
    item = make_queue(
        stage9_decision=QADecision.REVIEW,
        stage10a_decision=PronunciationEvidenceDecision.REVIEW,
        machine_review_issues=(issue_a, issue_b),
    )
    queue, records = setup_queue(tmp_path, item)
    add_review(
        queue,
        records,
        HumanReviewDecision.APPROVE,
        resolved=(issue_a.issue_id,),
    )
    _, result = evaluate(tmp_path, queue, records)
    assert result.decision is ReleaseGateDecision.NOT_RELEASED
    assert result.resolved_issue_ids == (issue_a.issue_id,)
    assert result.unresolved_issue_ids == (issue_b.issue_id,)
    assert ReleaseGateReason.UNRESOLVED_MACHINE_REVIEW_ISSUES in result.reason_codes


def test_review_record_requires_structured_reason_for_negative_decision(
    tmp_path: Path,
) -> None:
    queue, records = setup_queue(tmp_path)
    with pytest.raises(ValueError, match="structured reason"):
        add_review(queue, records, HumanReviewDecision.REJECT)


def test_release_manifest_retains_stage8_through_gate_provenance(tmp_path: Path) -> None:
    queue, records = setup_queue(tmp_path)
    review = add_review(queue, records, HumanReviewDecision.APPROVE)
    outputs, result = evaluate(tmp_path, queue, records)
    manifest = tmp_path / "released_audio_manifest.jsonl"
    assert build_release_manifest(
        queue_manifest=queue,
        records_manifest=records,
        gate_results_manifest=outputs.results_manifest,
        output_path=manifest,
        released_at=RELEASED_AT,
    ) == 1
    released = list(iter_released_audio(manifest))[0]
    assert released.artifact_run_id == HASH_B
    assert released.synthesis_run_id == HASH_B
    assert released.source_text == "JR新宿駅までご案内いたします。"
    assert released.audio_post_processing == make_queue().audio_post_processing
    assert released.stage9_qa_id == HASH_D
    assert released.stage10a_evidence_id == HASH_E
    assert released.regression_id == HASH_F
    assert released.human_review_id == review.review_id
    assert released.release_gate_id == result.gate_id
    assert released.release_policy_version == RELEASE_POLICY_VERSION


def test_empty_gate_writes_empty_release_manifest(tmp_path: Path) -> None:
    queue, records = setup_queue(tmp_path)
    outputs, _ = evaluate(tmp_path, queue, records)
    manifest = tmp_path / "released_audio_manifest.jsonl"
    assert build_release_manifest(
        queue_manifest=queue,
        records_manifest=records,
        gate_results_manifest=outputs.results_manifest,
        output_path=manifest,
    ) == 0
    assert manifest.read_bytes() == b""


def test_gate_outputs_are_deterministic_for_same_inputs(tmp_path: Path) -> None:
    queue, records = setup_queue(tmp_path)
    add_review(queue, records, HumanReviewDecision.APPROVE)
    first, _ = evaluate(tmp_path / "first", queue, records)
    second, _ = evaluate(tmp_path / "second", queue, records)
    assert first.results_manifest.read_bytes() == second.results_manifest.read_bytes()
    assert first.summary_file.read_bytes() == second.summary_file.read_bytes()


def test_records_keep_real_timestamp_and_are_append_only_history(tmp_path: Path) -> None:
    queue, records = setup_queue(tmp_path)
    first = add_review(queue, records, HumanReviewDecision.APPROVE)
    second = record_human_review(
        queue_manifest=queue,
        records_manifest=records,
        artifact_run_id=HASH_B,
        reviewer_id="fixture-reviewer-2",
        decision=HumanReviewDecision.REJECT,
        reason_codes=(ReviewReasonCode.VOICE_STYLE,),
        reviewed_at=datetime(2026, 9, 21, 12, 1, tzinfo=timezone.utc),
    )
    history = list(iter_human_review_records(records))
    assert {item.review_id for item in history} == {first.review_id, second.review_id}
    assert first.reviewed_at == REVIEWED_AT
    _, result = evaluate(tmp_path, queue, records)
    assert result.review_id == second.review_id
    assert result.decision is ReleaseGateDecision.NOT_RELEASED


def test_contracts_have_no_fake_quality_scores() -> None:
    fields = set(HumanReviewQueueItem.model_fields)
    assert "human_quality_score" not in fields
    assert "pronunciation_score" not in fields
    assert "naturalness_score" not in fields
    assert "release_confidence" not in fields
    assert "mos" not in fields
    assert "pitch_accent_accuracy" not in fields


def test_cli_has_no_force_release_and_supports_required_commands() -> None:
    parser = build_parser()
    prepare = parser.parse_args(
        [
            "prepare-review",
            "--artifact-manifest",
            "artifacts.jsonl",
            "--synthesis-runs",
            "runs.jsonl",
            "--qa-results",
            "qa.jsonl",
            "--pronunciation-evidence",
            "evidence.jsonl",
            "--regression-results",
            "regression.jsonl",
        ]
    )
    record = parser.parse_args(
        [
            "record-review",
            "--queue",
            "queue.jsonl",
            "--records",
            "records.jsonl",
            "--artifact-id",
            HASH_B,
            "--reviewer-id",
            "fixture-reviewer",
            "--decision",
            "approve",
        ]
    )
    evaluate_args = parser.parse_args(
        [
            "evaluate-release",
            "--queue",
            "queue.jsonl",
            "--records",
            "records.jsonl",
        ]
    )
    manifest = parser.parse_args(
        [
            "build-release-manifest",
            "--queue",
            "queue.jsonl",
            "--records",
            "records.jsonl",
            "--gate-results",
            "gate.jsonl",
            "--output",
            "released.jsonl",
        ]
    )
    assert {
        prepare.command,
        record.command,
        evaluate_args.command,
        manifest.command,
    } == {
        "prepare-review",
        "record-review",
        "evaluate-release",
        "build-release-manifest",
    }
    with pytest.raises(SystemExit):
        parser.parse_args(["build-release-manifest", "--force-release"])
