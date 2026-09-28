"""Stage 10B human-review and release-gate contracts and orchestration."""

from __future__ import annotations

import hashlib
import json
import re
from collections.abc import Iterable, Iterator
from dataclasses import dataclass
from datetime import datetime, timezone
from enum import Enum
from os import PathLike
from pathlib import Path
from typing import Annotated, Literal, TypeVar

from pydantic import BaseModel, ConfigDict, Field, JsonValue, StringConstraints, model_validator

from .generated_audio_qa import (
    GeneratedAudioQAResult,
    QADecision,
    QAExecutionStatus,
    iter_qa_results,
)
from .pronunciation_regression import (
    PronunciationEvidenceDecision,
    PronunciationEvidenceResult,
    PronunciationRegressionResult,
    RegressionDecision,
    RiskAlignmentStatus,
    iter_pronunciation_evidence,
    iter_pronunciation_regressions,
)
from .tts_synthesis import (
    AudioPostProcessingProvenance,
    GeneratedAudioArtifact,
    SynthesisRun,
    SynthesisRunStatus,
    iter_synthesis_runs,
)


HUMAN_REVIEW_RELEASE_SCHEMA_VERSION = "1.0"
HUMAN_REVIEW_POLICY_VERSION = "mandatory-human-final-review-v1"
RELEASE_POLICY_VERSION = "human-approved-release-gate-v1"
NonEmptyStr = Annotated[str, StringConstraints(strip_whitespace=True, min_length=1)]
ModelT = TypeVar("ModelT", bound=BaseModel)


class ReviewReleaseModel(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)


class ReviewQueueStatus(str, Enum):
    PENDING = "pending"


class ReviewPriority(str, Enum):
    MANDATORY_FINAL_REVIEW = "mandatory_final_review"
    PRONUNCIATION_REVIEW = "pronunciation_review"
    REGRESSION_REVIEW = "regression_review"
    CONTENT_REVIEW = "content_review"


class HumanReviewDecision(str, Enum):
    APPROVE = "approve"
    REWORK = "rework"
    REJECT = "reject"


class ReviewReasonCode(str, Enum):
    CONTENT = "content"
    PRONUNCIATION = "pronunciation"
    PROSODY = "prosody"
    NATURALNESS = "naturalness"
    AUDIO_ARTIFACT = "audio_artifact"
    VOICE_STYLE = "voice_style"
    OTHER = "other"


class ReworkTarget(str, Enum):
    TEXT_NORMALIZATION = "text_normalization"
    READING = "reading"
    PRONUNCIATION_OVERRIDE = "pronunciation_override"
    SYNTHESIS = "synthesis"
    OTHER = "other"


class ReviewFreshness(str, Enum):
    MISSING = "missing"
    CURRENT = "current"
    STALE = "stale"


class MachineReviewIssueSource(str, Enum):
    STAGE9_CONTENT_QA = "stage9_content_qa"
    STAGE10A_PRONUNCIATION = "stage10a_pronunciation"
    STAGE10A_REGRESSION = "stage10a_regression"


class HardBlockerCode(str, Enum):
    ARTIFACT_INTEGRITY_INVALID = "artifact_integrity_invalid"
    STAGE9_HARD_DROP = "stage9_hard_drop"
    STAGE9_EXECUTION_FAILED = "stage9_execution_failed"


class ReleaseGateDecision(str, Enum):
    RELEASED = "released"
    NOT_RELEASED = "not_released"


class ReleaseGateReason(str, Enum):
    ARTIFACT_INTEGRITY_INVALID = "artifact_integrity_invalid"
    STAGE9_HARD_DROP = "stage9_hard_drop"
    STAGE9_EXECUTION_FAILED = "stage9_execution_failed"
    UNRESOLVED_MACHINE_REVIEW_ISSUES = "unresolved_machine_review_issues"
    MISSING_HUMAN_REVIEW = "missing_human_review"
    STALE_HUMAN_REVIEW = "stale_human_review"
    HUMAN_REWORK = "human_rework"
    HUMAN_REJECT = "human_reject"


class HumanReviewPolicy(ReviewReleaseModel):
    policy_version: Literal[HUMAN_REVIEW_POLICY_VERSION] = HUMAN_REVIEW_POLICY_VERSION
    all_release_candidates_require_review: Literal[True] = True
    approval_bound_to_review_context: Literal[True] = True
    reviewer_identity_is_provenance_only: Literal[True] = True


class ReleasePolicy(ReviewReleaseModel):
    policy_version: Literal[RELEASE_POLICY_VERSION] = RELEASE_POLICY_VERSION
    require_artifact_integrity: Literal[True] = True
    stage9_review_is_human_resolvable: Literal[True] = True
    stage10a_review_is_human_resolvable: Literal[True] = True
    regression_review_is_human_resolvable: Literal[True] = True
    require_explicit_resolution_of_machine_review_issues: Literal[True] = True
    require_current_human_approval: Literal[True] = True
    force_release_supported: Literal[False] = False


class MachineReviewIssue(ReviewReleaseModel):
    issue_id: str = Field(pattern=r"^[0-9a-f]{64}$")
    source: MachineReviewIssueSource
    code: NonEmptyStr
    evidence_reference_id: NonEmptyStr
    summary: NonEmptyStr


class HumanReviewQueueItem(ReviewReleaseModel):
    schema_version: Literal[HUMAN_REVIEW_RELEASE_SCHEMA_VERSION] = (
        HUMAN_REVIEW_RELEASE_SCHEMA_VERSION
    )
    queue_item_id: str = Field(pattern=r"^[0-9a-f]{64}$")
    logical_case_id: NonEmptyStr
    sample_id: NonEmptyStr
    status: Literal[ReviewQueueStatus.PENDING] = ReviewQueueStatus.PENDING
    priorities: tuple[ReviewPriority, ...]
    artifact_run_id: str = Field(pattern=r"^[0-9a-f]{64}$")
    artifact_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    audio_path: NonEmptyStr
    audio_post_processing: AudioPostProcessingProvenance | None = None
    artifact_integrity_valid: bool
    hard_blocker_codes: tuple[HardBlockerCode, ...] = ()
    machine_review_issues: tuple[MachineReviewIssue, ...] = ()
    source_text: NonEmptyStr
    normalized_text: NonEmptyStr
    synthesis_input: NonEmptyStr
    expected_reading: NonEmptyStr
    expected_reference_ids: tuple[str, ...]
    pronunciation_risk_ids: tuple[str, ...]
    asr_transcript: str
    observed_canonical_reading: str
    stage9_qa_id: str = Field(pattern=r"^[0-9a-f]{64}$")
    stage9_status: QAExecutionStatus
    stage9_decision: QADecision
    stage9_issue_codes: tuple[str, ...]
    stage9_policy_fingerprint_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    stage10a_evidence_id: str = Field(pattern=r"^[0-9a-f]{64}$")
    stage10a_decision: PronunciationEvidenceDecision
    stage10a_policy_fingerprint_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    regression_id: str = Field(pattern=r"^[0-9a-f]{64}$")
    regression_decision: RegressionDecision
    regression_baseline_id: str | None = Field(
        default=None, pattern=r"^[0-9a-f]{64}$"
    )
    regression_policy_fingerprint_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    tts_backend: NonEmptyStr
    tts_model_name: NonEmptyStr
    speaker: NonEmptyStr
    synthesis_config_fingerprint_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    override_fingerprint_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    review_policy_version: Literal[HUMAN_REVIEW_POLICY_VERSION]
    review_policy_fingerprint_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    review_context_fingerprint_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")

    @model_validator(mode="after")
    def mandatory_priority_is_present(self) -> "HumanReviewQueueItem":
        if ReviewPriority.MANDATORY_FINAL_REVIEW not in self.priorities:
            raise ValueError("every release candidate requires mandatory final review")
        if len(set(self.expected_reference_ids)) != len(self.expected_reference_ids):
            raise ValueError("expected reference identifiers must be unique")
        if len(set(self.pronunciation_risk_ids)) != len(self.pronunciation_risk_ids):
            raise ValueError("pronunciation risk identifiers must be unique")
        issue_ids = [issue.issue_id for issue in self.machine_review_issues]
        if len(set(issue_ids)) != len(issue_ids):
            raise ValueError("machine review issue identifiers must be unique")
        if len(set(self.hard_blocker_codes)) != len(self.hard_blocker_codes):
            raise ValueError("hard blocker codes must be unique")
        return self


class HumanReviewRecord(ReviewReleaseModel):
    schema_version: Literal[HUMAN_REVIEW_RELEASE_SCHEMA_VERSION] = (
        HUMAN_REVIEW_RELEASE_SCHEMA_VERSION
    )
    review_id: str = Field(pattern=r"^[0-9a-f]{64}$")
    queue_item_id: str = Field(pattern=r"^[0-9a-f]{64}$")
    logical_case_id: NonEmptyStr
    sample_id: NonEmptyStr
    reviewer_id: NonEmptyStr
    reviewed_at: datetime
    decision: HumanReviewDecision
    reason_codes: tuple[ReviewReasonCode, ...] = ()
    resolved_issue_ids: tuple[str, ...] = Field(
        default=(),
    )
    notes: str | None = None
    rework_target: ReworkTarget | None = None
    artifact_run_id: str = Field(pattern=r"^[0-9a-f]{64}$")
    artifact_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    stage9_qa_id: str = Field(pattern=r"^[0-9a-f]{64}$")
    stage10a_evidence_id: str = Field(pattern=r"^[0-9a-f]{64}$")
    regression_id: str = Field(pattern=r"^[0-9a-f]{64}$")
    review_context_fingerprint_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    review_policy_version: Literal[HUMAN_REVIEW_POLICY_VERSION]
    review_policy_fingerprint_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")

    @model_validator(mode="after")
    def decision_fields_agree(self) -> "HumanReviewRecord":
        if self.reviewed_at.utcoffset() is None:
            raise ValueError("review timestamp must include a timezone")
        if len(set(self.reason_codes)) != len(self.reason_codes):
            raise ValueError("review reason codes must be unique")
        if len(set(self.resolved_issue_ids)) != len(self.resolved_issue_ids):
            raise ValueError("resolved machine issue identifiers must be unique")
        if any(
            not re.fullmatch(r"[0-9a-f]{64}", issue_id)
            for issue_id in self.resolved_issue_ids
        ):
            raise ValueError("resolved machine issue identifiers must be SHA-256 values")
        if self.decision in {HumanReviewDecision.REWORK, HumanReviewDecision.REJECT}:
            if not self.reason_codes:
                raise ValueError("REWORK and REJECT require a structured reason code")
        if self.decision is not HumanReviewDecision.REWORK and self.rework_target:
            raise ValueError("only REWORK can include an upstream return target")
        return self


class HumanReviewBatchSummary(ReviewReleaseModel):
    schema_version: Literal[HUMAN_REVIEW_RELEASE_SCHEMA_VERSION] = (
        HUMAN_REVIEW_RELEASE_SCHEMA_VERSION
    )
    review_policy_version: Literal[HUMAN_REVIEW_POLICY_VERSION]
    review_policy_fingerprint_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    total_items: int = Field(ge=0)
    pending_items: int = Field(ge=0)
    mandatory_final_review_items: int = Field(ge=0)
    pronunciation_review_items: int = Field(ge=0)
    regression_review_items: int = Field(ge=0)
    content_review_items: int = Field(ge=0)


class ReleaseGateResult(ReviewReleaseModel):
    schema_version: Literal[HUMAN_REVIEW_RELEASE_SCHEMA_VERSION] = (
        HUMAN_REVIEW_RELEASE_SCHEMA_VERSION
    )
    gate_id: str = Field(pattern=r"^[0-9a-f]{64}$")
    queue_item_id: str = Field(pattern=r"^[0-9a-f]{64}$")
    logical_case_id: NonEmptyStr
    sample_id: NonEmptyStr
    artifact_run_id: str = Field(pattern=r"^[0-9a-f]{64}$")
    artifact_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    review_id: str | None = Field(default=None, pattern=r"^[0-9a-f]{64}$")
    review_freshness: ReviewFreshness
    human_decision: HumanReviewDecision | None = None
    decision: ReleaseGateDecision
    reason_codes: tuple[ReleaseGateReason, ...]
    hard_blocker_codes: tuple[HardBlockerCode, ...] = ()
    resolved_issue_ids: tuple[str, ...] = ()
    unresolved_issue_ids: tuple[str, ...] = ()
    stage9_qa_id: str = Field(pattern=r"^[0-9a-f]{64}$")
    stage10a_evidence_id: str = Field(pattern=r"^[0-9a-f]{64}$")
    regression_id: str = Field(pattern=r"^[0-9a-f]{64}$")
    release_policy_version: Literal[RELEASE_POLICY_VERSION]
    release_policy_fingerprint_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")

    @model_validator(mode="after")
    def decision_fields_agree(self) -> "ReleaseGateResult":
        if self.review_freshness is ReviewFreshness.MISSING:
            if self.review_id is not None or self.human_decision is not None:
                raise ValueError("missing review cannot reference a decision")
        elif self.review_id is None or self.human_decision is None:
            raise ValueError("current or stale review must reference the review record")
        if self.decision is ReleaseGateDecision.RELEASED:
            if self.reason_codes:
                raise ValueError("released result cannot contain denial reasons")
            if self.review_freshness is not ReviewFreshness.CURRENT:
                raise ValueError("release requires a current review")
            if self.human_decision is not HumanReviewDecision.APPROVE:
                raise ValueError("release requires human APPROVE")
            if self.hard_blocker_codes or self.unresolved_issue_ids:
                raise ValueError("release cannot contain hard or unresolved blockers")
        elif not self.reason_codes:
            raise ValueError("not-released result requires at least one reason")
        return self


class ReleasedAudioArtifact(ReviewReleaseModel):
    schema_version: Literal[HUMAN_REVIEW_RELEASE_SCHEMA_VERSION] = (
        HUMAN_REVIEW_RELEASE_SCHEMA_VERSION
    )
    release_id: str = Field(pattern=r"^[0-9a-f]{64}$")
    logical_case_id: NonEmptyStr
    sample_id: NonEmptyStr
    artifact_run_id: str = Field(pattern=r"^[0-9a-f]{64}$")
    synthesis_run_id: str = Field(pattern=r"^[0-9a-f]{64}$")
    artifact_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    audio_path: NonEmptyStr
    source_text: NonEmptyStr | None = None
    audio_post_processing: AudioPostProcessingProvenance | None = None
    stage9_qa_id: str = Field(pattern=r"^[0-9a-f]{64}$")
    stage10a_evidence_id: str = Field(pattern=r"^[0-9a-f]{64}$")
    regression_id: str = Field(pattern=r"^[0-9a-f]{64}$")
    human_review_id: str = Field(pattern=r"^[0-9a-f]{64}$")
    release_gate_id: str = Field(pattern=r"^[0-9a-f]{64}$")
    reviewer_id: NonEmptyStr
    released_at: datetime
    tts_backend: NonEmptyStr
    tts_model_name: NonEmptyStr
    speaker: NonEmptyStr
    release_policy_version: Literal[RELEASE_POLICY_VERSION]
    release_policy_fingerprint_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")

    @model_validator(mode="after")
    def release_timestamp_has_timezone(self) -> "ReleasedAudioArtifact":
        if self.released_at.utcoffset() is None:
            raise ValueError("release timestamp must include a timezone")
        return self


class ReleaseBatchSummary(ReviewReleaseModel):
    schema_version: Literal[HUMAN_REVIEW_RELEASE_SCHEMA_VERSION] = (
        HUMAN_REVIEW_RELEASE_SCHEMA_VERSION
    )
    release_policy_version: Literal[RELEASE_POLICY_VERSION]
    release_policy_fingerprint_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    total_candidates: int = Field(ge=0)
    released: int = Field(ge=0)
    not_released: int = Field(ge=0)
    missing_reviews: int = Field(ge=0)
    stale_reviews: int = Field(ge=0)


@dataclass(frozen=True)
class HumanReviewQueueOutputs:
    queue_manifest: Path
    records_manifest: Path
    summary_file: Path
    summary: HumanReviewBatchSummary


@dataclass(frozen=True)
class ReleaseGateOutputs:
    results_manifest: Path
    summary_file: Path
    summary: ReleaseBatchSummary


def _canonical_sha256(payload: object) -> str:
    return hashlib.sha256(
        json.dumps(
            payload, ensure_ascii=False, sort_keys=True, separators=(",", ":")
        ).encode("utf-8")
    ).hexdigest()


def human_review_policy_fingerprint(policy: HumanReviewPolicy) -> str:
    return _canonical_sha256(policy.model_dump(mode="json"))


def release_policy_fingerprint(policy: ReleasePolicy) -> str:
    return _canonical_sha256(policy.model_dump(mode="json"))


def _read_models(path: Path, model: type[ModelT]) -> Iterator[ModelT]:
    if not path.exists():
        return
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            try:
                yield model.model_validate_json(line)
            except ValueError as exc:
                raise ValueError(
                    f"{path}:{line_number}: invalid {model.__name__}: {exc}"
                ) from exc


def _write_models(path: Path, items: Iterable[ReviewReleaseModel], key: str) -> int:
    values = sorted(items, key=lambda item: str(getattr(item, key)))
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="\n") as handle:
        for item in values:
            handle.write(item.model_dump_json())
            handle.write("\n")
    return len(values)


def _unique_index(items: Iterable[ModelT], field: str, kind: str) -> dict[str, ModelT]:
    indexed: dict[str, ModelT] = {}
    for item in items:
        value = getattr(item, field)
        if value is None:
            continue
        key = str(value)
        if key in indexed:
            raise ValueError(f"duplicate {kind} {field}: {key}")
        indexed[key] = item
    return indexed


def iter_human_review_queue(
    path: str | PathLike[str],
) -> Iterator[HumanReviewQueueItem]:
    yield from _read_models(Path(path), HumanReviewQueueItem)


def iter_human_review_records(
    path: str | PathLike[str],
) -> Iterator[HumanReviewRecord]:
    yield from _read_models(Path(path), HumanReviewRecord)


def iter_release_gate_results(
    path: str | PathLike[str],
) -> Iterator[ReleaseGateResult]:
    yield from _read_models(Path(path), ReleaseGateResult)


def iter_released_audio(
    path: str | PathLike[str],
) -> Iterator[ReleasedAudioArtifact]:
    yield from _read_models(Path(path), ReleasedAudioArtifact)


def _audio_integrity_valid(qa: GeneratedAudioQAResult) -> bool:
    integrity = qa.audio_integrity
    return bool(
        qa.status is QAExecutionStatus.COMPLETED
        and integrity.observed_audio_sha256 == integrity.expected_audio_sha256
        and qa.observed_audio_sha256 == qa.expected_audio_sha256
        and integrity.observed_sample_rate_hz == integrity.expected_sample_rate_hz
        and integrity.observed_channels == integrity.expected_channels
        and integrity.observed_duration_sec is not None
    )


def _hard_blocker_codes(qa: GeneratedAudioQAResult) -> tuple[HardBlockerCode, ...]:
    blockers: list[HardBlockerCode] = []
    if not _audio_integrity_valid(qa):
        blockers.append(HardBlockerCode.ARTIFACT_INTEGRITY_INVALID)
    if qa.status is QAExecutionStatus.FAILED or qa.decision is QADecision.UNDECIDED:
        blockers.append(HardBlockerCode.STAGE9_EXECUTION_FAILED)
    if qa.decision is QADecision.DROP:
        blockers.append(HardBlockerCode.STAGE9_HARD_DROP)
    return tuple(dict.fromkeys(blockers))


def _machine_issue(
    *,
    source: MachineReviewIssueSource,
    code: str,
    evidence_reference_id: str,
    summary: str,
) -> MachineReviewIssue:
    payload = {
        "source": source.value,
        "code": code,
        "evidence_reference_id": evidence_reference_id,
        "summary": summary,
    }
    return MachineReviewIssue(issue_id=_canonical_sha256(payload), **payload)


def _machine_review_issues(
    qa: GeneratedAudioQAResult,
    evidence: PronunciationEvidenceResult,
    regression: PronunciationRegressionResult,
) -> tuple[MachineReviewIssue, ...]:
    issues: list[MachineReviewIssue] = []
    if qa.decision is QADecision.REVIEW:
        issues.extend(
            _machine_issue(
                source=MachineReviewIssueSource.STAGE9_CONTENT_QA,
                code=issue.code,
                evidence_reference_id=qa.qa_id,
                summary=issue.message,
            )
            for issue in qa.issues
        )
    if evidence.decision is PronunciationEvidenceDecision.REVIEW:
        issues.extend(
            _machine_issue(
                source=MachineReviewIssueSource.STAGE10A_PRONUNCIATION,
                code=f"pronunciation_{risk.alignment_status.value}",
                evidence_reference_id=risk.risk_id,
                summary=risk.evidence_statement,
            )
            for risk in evidence.risk_evidence
            if risk.alignment_status
            in {
                RiskAlignmentStatus.EXPECTED_SEQUENCE_NOT_OBSERVED,
                RiskAlignmentStatus.ALIGNMENT_UNRESOLVED,
            }
        )
    if regression.decision is RegressionDecision.REVIEW:
        issues.extend(
            _machine_issue(
                source=MachineReviewIssueSource.STAGE10A_REGRESSION,
                code=f"regression_{change.kind.value}",
                evidence_reference_id=regression.regression_id,
                summary=change.message,
            )
            for change in regression.changes
            if change.material_pronunciation_change
        )
    return tuple(sorted(issues, key=lambda item: item.issue_id))


def _review_context_payload(
    *,
    artifact: GeneratedAudioArtifact,
    run: SynthesisRun,
    qa: GeneratedAudioQAResult,
    evidence: PronunciationEvidenceResult,
    regression: PronunciationRegressionResult,
    hard_blocker_codes: tuple[HardBlockerCode, ...],
    machine_review_issues: tuple[MachineReviewIssue, ...],
    review_policy_fingerprint_sha256: str,
) -> dict[str, JsonValue]:
    return {
        "logical_case_id": evidence.logical_case_id,
        "sample_id": artifact.sample_id,
        "artifact_run_id": artifact.run_id,
        "artifact_sha256": artifact.audio_sha256,
        "audio_post_processing": (
            artifact.audio_post_processing.model_dump(mode="json")
            if artifact.audio_post_processing is not None
            else None
        ),
        "source_text": evidence.source_text,
        "normalized_text": evidence.normalized_text,
        "synthesis_input": evidence.synthesis_input,
        "expected_reference_ids": sorted(
            reference.reference_id for reference in evidence.expected_references
        ),
        "pronunciation_risk_ids": sorted(
            risk.risk_id for risk in evidence.risk_evidence
        ),
        "synthesis_config_fingerprint_sha256": run.config_fingerprint_sha256,
        "override_fingerprint_sha256": evidence.override_fingerprint_sha256,
        "stage9_qa_id": qa.qa_id,
        "stage9_policy_fingerprint_sha256": qa.policy_fingerprint_sha256,
        "stage10a_evidence_id": evidence.evidence_id,
        "stage10a_policy_fingerprint_sha256": evidence.policy_fingerprint_sha256,
        "regression_id": regression.regression_id,
        "regression_policy_fingerprint_sha256": regression.policy_fingerprint_sha256,
        "hard_blocker_codes": [item.value for item in hard_blocker_codes],
        "machine_review_issues": [
            item.model_dump(mode="json") for item in machine_review_issues
        ],
        "review_policy_fingerprint_sha256": review_policy_fingerprint_sha256,
    }


def _validate_review_chain(
    artifact: GeneratedAudioArtifact,
    run: SynthesisRun,
    qa: GeneratedAudioQAResult,
    evidence: PronunciationEvidenceResult,
    regression: PronunciationRegressionResult,
) -> None:
    if run.status is not SynthesisRunStatus.SUCCESS:
        raise ValueError("human review queue requires a successful synthesis run")
    identifiers = {artifact.sample_id, run.sample_id, qa.sample_id, evidence.sample_id}
    if len(identifiers) != 1 or evidence.logical_case_id != artifact.sample_id:
        raise ValueError("sample identifiers disagree across Stage 8-10B provenance")
    run_ids = {artifact.run_id, run.run_id, qa.run_id, evidence.artifact_run_id}
    if len(run_ids) != 1:
        raise ValueError("run identifiers disagree across Stage 8-10B provenance")
    if evidence.synthesis_run_id != run.run_id:
        raise ValueError("Stage 10A synthesis reference does not match Stage 8")
    if evidence.stage9_qa_id != qa.qa_id:
        raise ValueError("Stage 10A QA reference does not match Stage 9")
    if evidence.observed_evidence.stage9_decision is not qa.decision:
        raise ValueError("Stage 10A decision snapshot does not match Stage 9")
    if evidence.observed_evidence.stage9_issue_codes != qa.decision_reasons:
        raise ValueError("Stage 10A issue snapshot does not match Stage 9")
    if regression.current_evidence_id != evidence.evidence_id:
        raise ValueError("regression result does not reference current Stage 10A evidence")
    if regression.logical_case_id != evidence.logical_case_id:
        raise ValueError("regression logical case does not match Stage 10A evidence")
    hashes = {
        artifact.audio_sha256,
        run.audio_sha256,
        qa.expected_audio_sha256,
        qa.observed_audio_sha256,
        evidence.audio_sha256,
    }
    if len(hashes) != 1 or None in hashes:
        raise ValueError("audio hashes disagree across Stage 8-10B provenance")
    if artifact.input_text != evidence.normalized_text:
        raise ValueError("normalized text changed across Stage 8-10A provenance")
    if artifact.synthesis_text != evidence.synthesis_input:
        raise ValueError("synthesis input changed across Stage 8-10A provenance")
    if qa.actual_synthesis_text != artifact.synthesis_text:
        raise ValueError("Stage 9 did not review the current synthesis input")
    if artifact.backend != evidence.tts_backend:
        raise ValueError("TTS backend provenance changed")
    if artifact.model_name != evidence.tts_model_name or artifact.speaker != evidence.speaker:
        raise ValueError("TTS model or speaker provenance changed")


def prepare_human_review_queue(
    *,
    artifact_manifest: Path,
    synthesis_runs_manifest: Path,
    qa_results_manifest: Path,
    pronunciation_evidence_manifest: Path,
    regression_results_manifest: Path,
    output_root: Path,
    policy: HumanReviewPolicy | None = None,
) -> HumanReviewQueueOutputs:
    active_policy = policy or HumanReviewPolicy()
    policy_fingerprint = human_review_policy_fingerprint(active_policy)
    artifacts = _unique_index(
        _read_models(Path(artifact_manifest), GeneratedAudioArtifact),
        "run_id",
        "generated audio artifact",
    )
    runs = _unique_index(
        iter_synthesis_runs(synthesis_runs_manifest), "run_id", "synthesis run"
    )
    qa_results = _unique_index(
        iter_qa_results(qa_results_manifest), "qa_id", "Stage 9 QA result"
    )
    evidence = _unique_index(
        iter_pronunciation_evidence(pronunciation_evidence_manifest),
        "evidence_id",
        "Stage 10A evidence",
    )
    regressions = _unique_index(
        iter_pronunciation_regressions(regression_results_manifest),
        "current_evidence_id",
        "Stage 10A regression result",
    )

    queue: list[HumanReviewQueueItem] = []
    for evidence_item in sorted(evidence.values(), key=lambda item: item.logical_case_id):
        artifact = artifacts.get(evidence_item.artifact_run_id)
        run = runs.get(evidence_item.synthesis_run_id)
        qa = qa_results.get(evidence_item.stage9_qa_id)
        regression = regressions.get(evidence_item.evidence_id)
        if artifact is None or run is None or qa is None or regression is None:
            raise ValueError(
                f"incomplete human-review provenance for {evidence_item.logical_case_id}"
            )
        _validate_review_chain(artifact, run, qa, evidence_item, regression)
        hard_blockers = _hard_blocker_codes(qa)
        machine_issues = _machine_review_issues(qa, evidence_item, regression)
        priorities = [ReviewPriority.MANDATORY_FINAL_REVIEW]
        if any(
            issue.source is MachineReviewIssueSource.STAGE9_CONTENT_QA
            for issue in machine_issues
        ) or hard_blockers:
            priorities.append(ReviewPriority.CONTENT_REVIEW)
        if any(
            issue.source is MachineReviewIssueSource.STAGE10A_PRONUNCIATION
            for issue in machine_issues
        ):
            priorities.append(ReviewPriority.PRONUNCIATION_REVIEW)
        if any(
            issue.source is MachineReviewIssueSource.STAGE10A_REGRESSION
            for issue in machine_issues
        ):
            priorities.append(ReviewPriority.REGRESSION_REVIEW)
        context = _canonical_sha256(
            _review_context_payload(
                artifact=artifact,
                run=run,
                qa=qa,
                evidence=evidence_item,
                regression=regression,
                hard_blocker_codes=hard_blockers,
                machine_review_issues=machine_issues,
                review_policy_fingerprint_sha256=policy_fingerprint,
            )
        )
        queue_id = _canonical_sha256(
            {
                "review_context_fingerprint_sha256": context,
                "priorities": [priority.value for priority in priorities],
            }
        )
        queue.append(
            HumanReviewQueueItem(
                queue_item_id=queue_id,
                logical_case_id=evidence_item.logical_case_id,
                sample_id=evidence_item.sample_id,
                priorities=tuple(priorities),
                artifact_run_id=artifact.run_id,
                artifact_sha256=artifact.audio_sha256,
                audio_path=artifact.audio_path,
                audio_post_processing=artifact.audio_post_processing,
                artifact_integrity_valid=_audio_integrity_valid(qa),
                hard_blocker_codes=hard_blockers,
                machine_review_issues=machine_issues,
                source_text=evidence_item.source_text,
                normalized_text=evidence_item.normalized_text,
                synthesis_input=evidence_item.synthesis_input,
                expected_reading=artifact.expected_reading_kana,
                expected_reference_ids=tuple(
                    sorted(item.reference_id for item in evidence_item.expected_references)
                ),
                pronunciation_risk_ids=tuple(
                    sorted(item.risk_id for item in evidence_item.risk_evidence)
                ),
                asr_transcript=evidence_item.observed_evidence.asr_transcript,
                observed_canonical_reading=(
                    evidence_item.observed_evidence.canonical_observed_reading
                ),
                stage9_qa_id=qa.qa_id,
                stage9_status=qa.status,
                stage9_decision=qa.decision,
                stage9_issue_codes=qa.decision_reasons,
                stage9_policy_fingerprint_sha256=qa.policy_fingerprint_sha256,
                stage10a_evidence_id=evidence_item.evidence_id,
                stage10a_decision=evidence_item.decision,
                stage10a_policy_fingerprint_sha256=(
                    evidence_item.policy_fingerprint_sha256
                ),
                regression_id=regression.regression_id,
                regression_decision=regression.decision,
                regression_baseline_id=regression.baseline_id,
                regression_policy_fingerprint_sha256=(
                    regression.policy_fingerprint_sha256
                ),
                tts_backend=artifact.backend,
                tts_model_name=artifact.model_name,
                speaker=artifact.speaker,
                synthesis_config_fingerprint_sha256=run.config_fingerprint_sha256,
                override_fingerprint_sha256=evidence_item.override_fingerprint_sha256,
                review_policy_version=active_policy.policy_version,
                review_policy_fingerprint_sha256=policy_fingerprint,
                review_context_fingerprint_sha256=context,
            )
        )

    destination = Path(output_root).resolve()
    queue_path = destination / "human_review_queue.jsonl"
    records_path = destination / "human_review_records.jsonl"
    summary_path = destination / "human_review_summary.json"
    _write_models(queue_path, queue, "queue_item_id")
    records_path.parent.mkdir(parents=True, exist_ok=True)
    records_path.touch(exist_ok=True)
    summary = HumanReviewBatchSummary(
        review_policy_version=active_policy.policy_version,
        review_policy_fingerprint_sha256=policy_fingerprint,
        total_items=len(queue),
        pending_items=len(queue),
        mandatory_final_review_items=len(queue),
        pronunciation_review_items=sum(
            ReviewPriority.PRONUNCIATION_REVIEW in item.priorities for item in queue
        ),
        regression_review_items=sum(
            ReviewPriority.REGRESSION_REVIEW in item.priorities for item in queue
        ),
        content_review_items=sum(
            ReviewPriority.CONTENT_REVIEW in item.priorities for item in queue
        ),
    )
    summary_path.write_text(summary.model_dump_json(indent=2) + "\n", encoding="utf-8")
    return HumanReviewQueueOutputs(
        queue_manifest=queue_path,
        records_manifest=records_path,
        summary_file=summary_path,
        summary=summary,
    )


def _utc_now() -> datetime:
    return datetime.now(timezone.utc)


def _normalize_timestamp(value: datetime | None) -> datetime:
    result = value or _utc_now()
    if result.utcoffset() is None:
        raise ValueError("timestamp must include a timezone")
    return result.astimezone(timezone.utc)


def record_human_review(
    *,
    queue_manifest: Path,
    records_manifest: Path,
    artifact_run_id: str,
    reviewer_id: str,
    decision: HumanReviewDecision,
    reason_codes: Iterable[ReviewReasonCode] = (),
    resolved_issue_ids: Iterable[str] = (),
    notes: str | None = None,
    rework_target: ReworkTarget | None = None,
    reviewed_at: datetime | None = None,
    policy: HumanReviewPolicy | None = None,
) -> HumanReviewRecord:
    active_policy = policy or HumanReviewPolicy()
    policy_fingerprint = human_review_policy_fingerprint(active_policy)
    matches = [
        item
        for item in iter_human_review_queue(queue_manifest)
        if item.artifact_run_id == artifact_run_id
    ]
    if len(matches) != 1:
        raise ValueError(
            f"artifact id must identify exactly one review item: {artifact_run_id}"
        )
    item = matches[0]
    if item.review_policy_fingerprint_sha256 != policy_fingerprint:
        raise ValueError("review queue uses a different human-review policy")
    timestamp = _normalize_timestamp(reviewed_at)
    reasons = tuple(dict.fromkeys(reason_codes))
    resolved = tuple(dict.fromkeys(resolved_issue_ids))
    known_issue_ids = {issue.issue_id for issue in item.machine_review_issues}
    unknown_issue_ids = sorted(set(resolved) - known_issue_ids)
    if unknown_issue_ids:
        raise ValueError(
            "resolved issue identifiers are not present in the current queue item: "
            + ", ".join(unknown_issue_ids)
        )
    normalized_notes = notes.strip() if notes and notes.strip() else None
    payload = {
        "queue_item_id": item.queue_item_id,
        "review_context_fingerprint_sha256": item.review_context_fingerprint_sha256,
        "reviewer_id": reviewer_id.strip(),
        "reviewed_at": timestamp.isoformat(),
        "decision": decision.value,
        "reason_codes": [reason.value for reason in reasons],
        "resolved_issue_ids": sorted(resolved),
        "notes": normalized_notes,
        "rework_target": rework_target.value if rework_target else None,
        "review_policy_fingerprint_sha256": policy_fingerprint,
    }
    record = HumanReviewRecord(
        review_id=_canonical_sha256(payload),
        queue_item_id=item.queue_item_id,
        logical_case_id=item.logical_case_id,
        sample_id=item.sample_id,
        reviewer_id=reviewer_id,
        reviewed_at=timestamp,
        decision=decision,
        reason_codes=reasons,
        resolved_issue_ids=tuple(sorted(resolved)),
        notes=normalized_notes,
        rework_target=rework_target,
        artifact_run_id=item.artifact_run_id,
        artifact_sha256=item.artifact_sha256,
        stage9_qa_id=item.stage9_qa_id,
        stage10a_evidence_id=item.stage10a_evidence_id,
        regression_id=item.regression_id,
        review_context_fingerprint_sha256=item.review_context_fingerprint_sha256,
        review_policy_version=active_policy.policy_version,
        review_policy_fingerprint_sha256=policy_fingerprint,
    )
    records_path = Path(records_manifest)
    existing = list(iter_human_review_records(records_path))
    if any(item.review_id == record.review_id for item in existing):
        raise ValueError(f"duplicate human review record: {record.review_id}")
    _write_models(records_path, [*existing, record], "review_id")
    return record


def _latest_reviews_by_case(
    records: Iterable[HumanReviewRecord],
) -> dict[str, HumanReviewRecord]:
    latest: dict[str, HumanReviewRecord] = {}
    for record in records:
        previous = latest.get(record.logical_case_id)
        if previous is None or (record.reviewed_at, record.review_id) > (
            previous.reviewed_at,
            previous.review_id,
        ):
            latest[record.logical_case_id] = record
    return latest


def _review_is_current(item: HumanReviewQueueItem, record: HumanReviewRecord) -> bool:
    known_issue_ids = {issue.issue_id for issue in item.machine_review_issues}
    return bool(
        record.queue_item_id == item.queue_item_id
        and record.artifact_run_id == item.artifact_run_id
        and record.artifact_sha256 == item.artifact_sha256
        and record.stage9_qa_id == item.stage9_qa_id
        and record.stage10a_evidence_id == item.stage10a_evidence_id
        and record.regression_id == item.regression_id
        and record.review_context_fingerprint_sha256
        == item.review_context_fingerprint_sha256
        and record.review_policy_fingerprint_sha256
        == item.review_policy_fingerprint_sha256
        and set(record.resolved_issue_ids) <= known_issue_ids
    )


def evaluate_release_gate(
    *,
    queue_manifest: Path,
    records_manifest: Path,
    output_root: Path,
    policy: ReleasePolicy | None = None,
) -> ReleaseGateOutputs:
    active_policy = policy or ReleasePolicy()
    policy_fingerprint = release_policy_fingerprint(active_policy)
    queue = list(iter_human_review_queue(queue_manifest))
    latest = _latest_reviews_by_case(iter_human_review_records(records_manifest))
    results: list[ReleaseGateResult] = []
    for item in queue:
        record = latest.get(item.logical_case_id)
        if record is None:
            freshness = ReviewFreshness.MISSING
        elif _review_is_current(item, record):
            freshness = ReviewFreshness.CURRENT
        else:
            freshness = ReviewFreshness.STALE

        hard_blockers = list(item.hard_blocker_codes)
        if not item.artifact_integrity_valid:
            hard_blockers.append(HardBlockerCode.ARTIFACT_INTEGRITY_INVALID)
        if (
            item.stage9_status is QAExecutionStatus.FAILED
            or item.stage9_decision is QADecision.UNDECIDED
        ):
            hard_blockers.append(HardBlockerCode.STAGE9_EXECUTION_FAILED)
        if item.stage9_decision is QADecision.DROP:
            hard_blockers.append(HardBlockerCode.STAGE9_HARD_DROP)
        hard_blockers = list(dict.fromkeys(hard_blockers))

        known_issue_ids = {issue.issue_id for issue in item.machine_review_issues}
        resolved_issue_ids = (
            set(record.resolved_issue_ids)
            if record is not None and freshness is ReviewFreshness.CURRENT
            else set()
        )
        unresolved_issue_ids = sorted(known_issue_ids - resolved_issue_ids)

        reasons: list[ReleaseGateReason] = []
        if HardBlockerCode.ARTIFACT_INTEGRITY_INVALID in hard_blockers:
            reasons.append(ReleaseGateReason.ARTIFACT_INTEGRITY_INVALID)
        if HardBlockerCode.STAGE9_HARD_DROP in hard_blockers:
            reasons.append(ReleaseGateReason.STAGE9_HARD_DROP)
        if HardBlockerCode.STAGE9_EXECUTION_FAILED in hard_blockers:
            reasons.append(ReleaseGateReason.STAGE9_EXECUTION_FAILED)
        if unresolved_issue_ids:
            reasons.append(ReleaseGateReason.UNRESOLVED_MACHINE_REVIEW_ISSUES)
        if freshness is ReviewFreshness.MISSING:
            reasons.append(ReleaseGateReason.MISSING_HUMAN_REVIEW)
        elif freshness is ReviewFreshness.STALE:
            reasons.append(ReleaseGateReason.STALE_HUMAN_REVIEW)
        elif record is not None and record.decision is HumanReviewDecision.REWORK:
            reasons.append(ReleaseGateReason.HUMAN_REWORK)
        elif record is not None and record.decision is HumanReviewDecision.REJECT:
            reasons.append(ReleaseGateReason.HUMAN_REJECT)

        decision = (
            ReleaseGateDecision.NOT_RELEASED
            if reasons
            else ReleaseGateDecision.RELEASED
        )
        payload = {
            "queue_item_id": item.queue_item_id,
            "review_id": record.review_id if record else None,
            "review_freshness": freshness.value,
            "decision": decision.value,
            "reason_codes": [reason.value for reason in reasons],
            "hard_blocker_codes": [item.value for item in hard_blockers],
            "resolved_issue_ids": sorted(resolved_issue_ids),
            "unresolved_issue_ids": unresolved_issue_ids,
            "release_policy_fingerprint_sha256": policy_fingerprint,
        }
        results.append(
            ReleaseGateResult(
                gate_id=_canonical_sha256(payload),
                queue_item_id=item.queue_item_id,
                logical_case_id=item.logical_case_id,
                sample_id=item.sample_id,
                artifact_run_id=item.artifact_run_id,
                artifact_sha256=item.artifact_sha256,
                review_id=record.review_id if record else None,
                review_freshness=freshness,
                human_decision=record.decision if record else None,
                decision=decision,
                reason_codes=tuple(reasons),
                hard_blocker_codes=tuple(hard_blockers),
                resolved_issue_ids=tuple(sorted(resolved_issue_ids)),
                unresolved_issue_ids=tuple(unresolved_issue_ids),
                stage9_qa_id=item.stage9_qa_id,
                stage10a_evidence_id=item.stage10a_evidence_id,
                regression_id=item.regression_id,
                release_policy_version=active_policy.policy_version,
                release_policy_fingerprint_sha256=policy_fingerprint,
            )
        )

    destination = Path(output_root).resolve()
    results_path = destination / "release_gate_results.jsonl"
    summary_path = destination / "release_summary.json"
    _write_models(results_path, results, "gate_id")
    summary = ReleaseBatchSummary(
        release_policy_version=active_policy.policy_version,
        release_policy_fingerprint_sha256=policy_fingerprint,
        total_candidates=len(results),
        released=sum(item.decision is ReleaseGateDecision.RELEASED for item in results),
        not_released=sum(
            item.decision is ReleaseGateDecision.NOT_RELEASED for item in results
        ),
        missing_reviews=sum(
            item.review_freshness is ReviewFreshness.MISSING for item in results
        ),
        stale_reviews=sum(
            item.review_freshness is ReviewFreshness.STALE for item in results
        ),
    )
    destination.mkdir(parents=True, exist_ok=True)
    summary_path.write_text(summary.model_dump_json(indent=2) + "\n", encoding="utf-8")
    return ReleaseGateOutputs(
        results_manifest=results_path,
        summary_file=summary_path,
        summary=summary,
    )


def build_release_manifest(
    *,
    queue_manifest: Path,
    records_manifest: Path,
    gate_results_manifest: Path,
    output_path: Path,
    released_at: datetime | None = None,
    policy: ReleasePolicy | None = None,
) -> int:
    active_policy = policy or ReleasePolicy()
    policy_fingerprint = release_policy_fingerprint(active_policy)
    queue = _unique_index(
        iter_human_review_queue(queue_manifest), "queue_item_id", "review queue item"
    )
    reviews = list(iter_human_review_records(records_manifest))
    reviews_by_id = _unique_index(reviews, "review_id", "human review record")
    latest = _latest_reviews_by_case(reviews)
    released: list[ReleasedAudioArtifact] = []
    release_time: datetime | None = None
    for gate in iter_release_gate_results(gate_results_manifest):
        if gate.decision is not ReleaseGateDecision.RELEASED:
            continue
        item = queue.get(gate.queue_item_id)
        record = reviews_by_id.get(gate.review_id or "")
        if item is None or record is None:
            raise ValueError("released gate result has incomplete queue/review provenance")
        if gate.release_policy_fingerprint_sha256 != policy_fingerprint:
            raise ValueError("release gate result uses a different active policy")
        if (
            gate.logical_case_id != item.logical_case_id
            or gate.sample_id != item.sample_id
            or gate.artifact_run_id != item.artifact_run_id
            or gate.artifact_sha256 != item.artifact_sha256
            or gate.stage9_qa_id != item.stage9_qa_id
            or gate.stage10a_evidence_id != item.stage10a_evidence_id
            or gate.regression_id != item.regression_id
            or gate.review_id != record.review_id
        ):
            raise ValueError("release gate identities do not match queue/review provenance")
        if latest.get(item.logical_case_id) != record:
            raise ValueError("release gate result does not use the latest human review")
        if not _review_is_current(item, record):
            raise ValueError("release gate result references a stale human review")
        if record.decision is not HumanReviewDecision.APPROVE:
            raise ValueError("release manifest requires human APPROVE")
        required_issue_ids = {issue.issue_id for issue in item.machine_review_issues}
        if (
            not item.artifact_integrity_valid
            or item.hard_blocker_codes
            or item.stage9_status is QAExecutionStatus.FAILED
            or item.stage9_decision
            in {QADecision.DROP, QADecision.UNDECIDED}
            or set(record.resolved_issue_ids) != required_issue_ids
            or gate.hard_blocker_codes
            or gate.unresolved_issue_ids
            or set(gate.resolved_issue_ids) != required_issue_ids
        ):
            raise ValueError("release gate result conflicts with current machine evidence")
        expected_gate_id = _canonical_sha256(
            {
                "queue_item_id": item.queue_item_id,
                "review_id": record.review_id,
                "review_freshness": ReviewFreshness.CURRENT.value,
                "decision": ReleaseGateDecision.RELEASED.value,
                "reason_codes": [],
                "hard_blocker_codes": [],
                "resolved_issue_ids": sorted(required_issue_ids),
                "unresolved_issue_ids": [],
                "release_policy_fingerprint_sha256": policy_fingerprint,
            }
        )
        if gate.gate_id != expected_gate_id:
            raise ValueError("release gate identifier does not match its decision evidence")
        if release_time is None:
            release_time = _normalize_timestamp(released_at)
        payload = {
            "queue_item_id": item.queue_item_id,
            "review_id": record.review_id,
            "gate_id": gate.gate_id,
            "artifact_sha256": item.artifact_sha256,
            "source_text": item.source_text,
            "audio_post_processing": (
                item.audio_post_processing.model_dump(mode="json")
                if item.audio_post_processing is not None
                else None
            ),
            "released_at": release_time.isoformat(),
            "release_policy_fingerprint_sha256": policy_fingerprint,
        }
        released.append(
            ReleasedAudioArtifact(
                release_id=_canonical_sha256(payload),
                logical_case_id=item.logical_case_id,
                sample_id=item.sample_id,
                artifact_run_id=item.artifact_run_id,
                synthesis_run_id=item.artifact_run_id,
                artifact_sha256=item.artifact_sha256,
                audio_path=item.audio_path,
                source_text=item.source_text,
                audio_post_processing=item.audio_post_processing,
                stage9_qa_id=item.stage9_qa_id,
                stage10a_evidence_id=item.stage10a_evidence_id,
                regression_id=item.regression_id,
                human_review_id=record.review_id,
                release_gate_id=gate.gate_id,
                reviewer_id=record.reviewer_id,
                released_at=release_time,
                tts_backend=item.tts_backend,
                tts_model_name=item.tts_model_name,
                speaker=item.speaker,
                release_policy_version=active_policy.policy_version,
                release_policy_fingerprint_sha256=policy_fingerprint,
            )
        )
    return _write_models(Path(output_path), released, "release_id")
