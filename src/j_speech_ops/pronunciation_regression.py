"""Stage 10A pronunciation evidence and regression contracts/orchestration."""

from __future__ import annotations

import hashlib
import json
from collections.abc import Iterable, Iterator
from dataclasses import dataclass
from enum import Enum
from os import PathLike
from pathlib import Path
from typing import Annotated, Literal, TypeVar

from pydantic import BaseModel, ConfigDict, Field, JsonValue, StringConstraints, model_validator

from .generated_audio_qa import (
    GeneratedAudioQAResult,
    QADecision,
    QAExecutionStatus,
    QAPassedAudioArtifact,
    canonicalize_japanese_reading,
    iter_qa_results,
)
from .schemas import CurationDecision
from .tts_synthesis import (
    GeneratedAudioArtifact,
    SynthesisRun,
    SynthesisRunStatus,
    iter_synthesis_runs,
)
from .tts_text_curation import (
    TTSTextCurationReport,
    TTSTextCurationStatus,
    iter_tts_text_curation_reports,
)


PRONUNCIATION_EVIDENCE_SCHEMA_VERSION = "1.0"
PRONUNCIATION_EVIDENCE_POLICY_VERSION = "pronunciation-evidence-regression-v1"
NonEmptyStr = Annotated[str, StringConstraints(strip_whitespace=True, min_length=1)]
ModelT = TypeVar("ModelT", bound=BaseModel)


class PronunciationModel(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)


class ExpectationSource(str, Enum):
    NORMALIZATION_RULE = "normalization_rule"
    READING_PROVIDER = "reading_provider"
    CONFIRMED_OVERRIDE = "confirmed_override"
    HUMAN_CONFIRMED = "human_confirmed"


class ConfirmationStatus(str, Enum):
    DERIVED = "derived"
    CONFIG_CONFIRMED = "config_confirmed"
    HUMAN_CONFIRMED = "human_confirmed"
    UNCONFIRMED = "unconfirmed"


class ExpectationScope(str, Enum):
    FULL_TEXT = "full_text"
    RISK_SPAN = "risk_span"


class RiskEvidenceSource(str, Enum):
    STAGE7_WATCHLIST = "stage7_watchlist"
    CONFIRMED_OVERRIDE_PROVENANCE = "confirmed_override_provenance"


class RiskAlignmentStatus(str, Enum):
    EXPECTED_SEQUENCE_OBSERVED = "expected_sequence_observed"
    EXPECTED_SEQUENCE_NOT_OBSERVED = "expected_sequence_not_observed"
    ALIGNMENT_UNRESOLVED = "alignment_unresolved"


class PronunciationEvidenceDecision(str, Enum):
    NO_REVIEW_NEEDED = "no_review_needed"
    REVIEW = "review"


class RegressionDecision(str, Enum):
    NO_REGRESSION_DETECTED = "no_regression_detected"
    REVIEW = "review"


class RegressionChangeKind(str, Enum):
    SOURCE_TEXT_CHANGED = "source_text_changed"
    EXPECTED_READING_CHANGED = "expected_reading_changed"
    OBSERVED_READING_CHANGED = "observed_reading_changed"
    RISK_STATUS_CHANGED = "risk_status_changed"
    PROVENANCE_CONFIG_CHANGED = "provenance_config_changed"
    ARTIFACT_IDENTITY_CHANGED = "artifact_identity_changed"
    BASELINE_CASE_MISSING = "baseline_case_missing"
    CURRENT_CASE_MISSING = "current_case_missing"


class PronunciationEvidencePolicy(PronunciationModel):
    policy_version: Literal[PRONUNCIATION_EVIDENCE_POLICY_VERSION] = (
        PRONUNCIATION_EVIDENCE_POLICY_VERSION
    )
    expected_sequence_matching: Literal["canonical-substring-v1"] = (
        "canonical-substring-v1"
    )
    unresolved_alignment_routes_review: Literal[True] = True
    changed_expected_or_observed_routes_review: Literal[True] = True
    artifact_hash_change_is_pronunciation_regression: Literal[False] = False


class TextSpan(PronunciationModel):
    text_form: Literal["source", "normalized", "synthesis"]
    start: int = Field(ge=0)
    end: int = Field(gt=0)
    text: NonEmptyStr

    @model_validator(mode="after")
    def interval_matches_text(self) -> "TextSpan":
        if self.end - self.start != len(self.text):
            raise ValueError("span offsets must match text length")
        return self


class ExpectedPronunciationReference(PronunciationModel):
    reference_id: str = Field(pattern=r"^[0-9a-f]{64}$")
    scope: ExpectationScope
    surface: str | None = None
    expected_reading: NonEmptyStr
    canonical_expected_reading: NonEmptyStr
    expectation_source: ExpectationSource
    confirmation_status: ConfirmationStatus
    provenance_reference: NonEmptyStr

    @model_validator(mode="after")
    def source_and_confirmation_agree(self) -> "ExpectedPronunciationReference":
        if self.scope is ExpectationScope.RISK_SPAN and not self.surface:
            raise ValueError("risk-span expectation requires a surface")
        if self.expectation_source is ExpectationSource.HUMAN_CONFIRMED:
            if self.confirmation_status is not ConfirmationStatus.HUMAN_CONFIRMED:
                raise ValueError("human-confirmed source requires human confirmation")
        if self.expectation_source is ExpectationSource.READING_PROVIDER:
            if self.confirmation_status is not ConfirmationStatus.DERIVED:
                raise ValueError("reading-provider evidence must remain derived")
        return self


class PronunciationRiskEvidence(PronunciationModel):
    risk_id: str = Field(pattern=r"^[0-9a-f]{64}$")
    surface: NonEmptyStr
    risk_type: NonEmptyStr
    reason: NonEmptyStr
    occurrence_count: int = Field(gt=0)
    evidence_source: RiskEvidenceSource
    spans: tuple[TextSpan, ...]
    expected_reference_id: str | None = Field(
        default=None, pattern=r"^[0-9a-f]{64}$"
    )
    alignment_status: RiskAlignmentStatus
    evidence_statement: NonEmptyStr

    @model_validator(mode="after")
    def alignment_and_reference_agree(self) -> "PronunciationRiskEvidence":
        if self.alignment_status is RiskAlignmentStatus.ALIGNMENT_UNRESOLVED:
            if self.expected_reference_id is not None:
                raise ValueError("unresolved alignment must not claim an expectation")
        elif self.expected_reference_id is None:
            raise ValueError("sequence observation requires an expected reference")
        return self


class ObservedPronunciationEvidence(PronunciationModel):
    stage9_qa_id: str = Field(pattern=r"^[0-9a-f]{64}$")
    asr_transcript: str
    observed_reading_kana: str
    canonical_observed_reading: str
    stage9_decision: QADecision
    stage9_issue_codes: tuple[str, ...] = ()
    stage9_edit_distance: int | None = Field(default=None, ge=0)
    stage9_cer: float | None = Field(default=None, ge=0.0)


class PronunciationEvidenceResult(PronunciationModel):
    schema_version: Literal[PRONUNCIATION_EVIDENCE_SCHEMA_VERSION] = (
        PRONUNCIATION_EVIDENCE_SCHEMA_VERSION
    )
    evidence_id: str = Field(pattern=r"^[0-9a-f]{64}$")
    logical_case_id: NonEmptyStr
    sample_id: NonEmptyStr
    artifact_run_id: str = Field(pattern=r"^[0-9a-f]{64}$")
    synthesis_run_id: str = Field(pattern=r"^[0-9a-f]{64}$")
    stage9_qa_id: str = Field(pattern=r"^[0-9a-f]{64}$")
    policy_version: Literal[PRONUNCIATION_EVIDENCE_POLICY_VERSION]
    policy_fingerprint_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    decision: PronunciationEvidenceDecision
    review_reasons: tuple[str, ...] = ()
    source_text: NonEmptyStr
    normalized_text: NonEmptyStr
    synthesis_input: NonEmptyStr
    normalization_transformations: tuple[str, ...] = ()
    expected_references: tuple[ExpectedPronunciationReference, ...]
    risk_evidence: tuple[PronunciationRiskEvidence, ...] = ()
    observed_evidence: ObservedPronunciationEvidence
    audio_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    tts_backend: NonEmptyStr
    tts_model_name: NonEmptyStr
    speaker: NonEmptyStr
    synthesis_strategy: NonEmptyStr
    synthesis_config_fingerprint_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    stage7_policy_version: NonEmptyStr
    stage7_policy_fingerprint_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    preparation_policy_version: NonEmptyStr
    watchlist_version: NonEmptyStr
    watchlist_fingerprint_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    override_fingerprint_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    stage9_policy_version: NonEmptyStr
    stage9_policy_fingerprint_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    stage9_asr_model_identity: NonEmptyStr
    stage9_asr_config_fingerprint_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    stage9_reading_provider_fingerprint_sha256: str = Field(
        pattern=r"^[0-9a-f]{64}$"
    )

    @model_validator(mode="after")
    def decision_matches_risks(self) -> "PronunciationEvidenceResult":
        needs_review = any(
            risk.alignment_status
            in {
                RiskAlignmentStatus.EXPECTED_SEQUENCE_NOT_OBSERVED,
                RiskAlignmentStatus.ALIGNMENT_UNRESOLVED,
            }
            for risk in self.risk_evidence
        )
        if needs_review != (self.decision is PronunciationEvidenceDecision.REVIEW):
            raise ValueError("pronunciation evidence decision does not match risk evidence")
        if self.decision is PronunciationEvidenceDecision.REVIEW and not self.review_reasons:
            raise ValueError("review evidence requires reasons")
        if self.decision is PronunciationEvidenceDecision.NO_REVIEW_NEEDED and self.review_reasons:
            raise ValueError("no-review evidence cannot contain review reasons")
        return self


class Stage9ComparisonSnapshot(PronunciationModel):
    qa_id: str = Field(pattern=r"^[0-9a-f]{64}$")
    decision: QADecision
    canonical_expected_reading: NonEmptyStr
    canonical_observed_reading: NonEmptyStr
    edit_distance: int = Field(ge=0)
    cer: float = Field(ge=0.0)
    issue_codes: tuple[str, ...] = ()


class PronunciationRegressionBaseline(PronunciationModel):
    schema_version: Literal[PRONUNCIATION_EVIDENCE_SCHEMA_VERSION] = (
        PRONUNCIATION_EVIDENCE_SCHEMA_VERSION
    )
    baseline_id: str = Field(pattern=r"^[0-9a-f]{64}$")
    logical_case_id: NonEmptyStr
    source_evidence_id: str = Field(pattern=r"^[0-9a-f]{64}$")
    source_text: NonEmptyStr
    expected_references: tuple[ExpectedPronunciationReference, ...]
    risk_evidence: tuple[PronunciationRiskEvidence, ...]
    observed_canonical_reading: str
    stage9_comparison: Stage9ComparisonSnapshot
    audio_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    tts_backend: NonEmptyStr
    tts_model_name: NonEmptyStr
    speaker: NonEmptyStr
    synthesis_strategy: NonEmptyStr
    synthesis_config_fingerprint_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    stage7_policy_version: NonEmptyStr
    stage7_policy_fingerprint_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    preparation_policy_version: NonEmptyStr
    watchlist_version: NonEmptyStr
    watchlist_fingerprint_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    override_fingerprint_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    stage9_policy_version: NonEmptyStr
    stage9_policy_fingerprint_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    stage9_asr_model_identity: NonEmptyStr
    stage9_asr_config_fingerprint_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    stage9_reading_provider_fingerprint_sha256: str = Field(
        pattern=r"^[0-9a-f]{64}$"
    )
    stage10a_policy_version: NonEmptyStr
    stage10a_policy_fingerprint_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")


class RegressionChange(PronunciationModel):
    kind: RegressionChangeKind
    field: NonEmptyStr
    baseline_value: JsonValue
    current_value: JsonValue
    material_pronunciation_change: bool
    message: NonEmptyStr


class PronunciationRegressionResult(PronunciationModel):
    schema_version: Literal[PRONUNCIATION_EVIDENCE_SCHEMA_VERSION] = (
        PRONUNCIATION_EVIDENCE_SCHEMA_VERSION
    )
    regression_id: str = Field(pattern=r"^[0-9a-f]{64}$")
    logical_case_id: NonEmptyStr
    baseline_id: str | None = Field(default=None, pattern=r"^[0-9a-f]{64}$")
    current_evidence_id: str | None = Field(default=None, pattern=r"^[0-9a-f]{64}$")
    decision: RegressionDecision
    changes: tuple[RegressionChange, ...]
    policy_version: Literal[PRONUNCIATION_EVIDENCE_POLICY_VERSION]
    policy_fingerprint_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")

    @model_validator(mode="after")
    def decision_matches_material_changes(self) -> "PronunciationRegressionResult":
        material = any(change.material_pronunciation_change for change in self.changes)
        if material != (self.decision is RegressionDecision.REVIEW):
            raise ValueError("regression decision must follow material evidence")
        return self


class PronunciationRegressionSummary(PronunciationModel):
    schema_version: Literal[PRONUNCIATION_EVIDENCE_SCHEMA_VERSION] = (
        PRONUNCIATION_EVIDENCE_SCHEMA_VERSION
    )
    policy_version: Literal[PRONUNCIATION_EVIDENCE_POLICY_VERSION]
    policy_fingerprint_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    baseline_cases: int = Field(ge=0)
    current_cases: int = Field(ge=0)
    compared_cases: int = Field(ge=0)
    no_regression_detected: int = Field(ge=0)
    review_candidates: int = Field(ge=0)
    provenance_only_changes: int = Field(ge=0)


@dataclass(frozen=True)
class PronunciationEvidenceOutputs:
    evidence_manifest: Path
    review_candidates: Path
    total: int
    review_count: int


@dataclass(frozen=True)
class PronunciationRegressionOutputs:
    results_manifest: Path
    review_candidates: Path
    summary_file: Path
    summary: PronunciationRegressionSummary


def _canonical_sha256(payload: object) -> str:
    return hashlib.sha256(
        json.dumps(
            payload, ensure_ascii=False, sort_keys=True, separators=(",", ":")
        ).encode("utf-8")
    ).hexdigest()


def pronunciation_policy_fingerprint(policy: PronunciationEvidencePolicy) -> str:
    return _canonical_sha256(policy.model_dump(mode="json"))


def _read_models(path: Path, model: type[ModelT]) -> Iterator[ModelT]:
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            try:
                yield model.model_validate_json(line)
            except ValueError as exc:
                raise ValueError(f"{path}:{line_number}: invalid {model.__name__}: {exc}") from exc


def _unique_index(items: Iterable[BaseModel], field: str, kind: str) -> dict[str, BaseModel]:
    indexed: dict[str, BaseModel] = {}
    for item in items:
        key = str(getattr(item, field))
        if key in indexed:
            raise ValueError(f"duplicate {kind} {field}: {key}")
        indexed[key] = item
    return indexed


def _write_models(path: Path, items: Iterable[PronunciationModel], key: str) -> int:
    values = sorted(items, key=lambda item: str(getattr(item, key)))
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="\n") as handle:
        for item in values:
            handle.write(item.model_dump_json())
            handle.write("\n")
    return len(values)


def iter_pronunciation_evidence(
    path: str | PathLike[str],
) -> Iterator[PronunciationEvidenceResult]:
    yield from _read_models(Path(path), PronunciationEvidenceResult)


def iter_pronunciation_baselines(
    path: str | PathLike[str],
) -> Iterator[PronunciationRegressionBaseline]:
    yield from _read_models(Path(path), PronunciationRegressionBaseline)


def iter_pronunciation_regressions(
    path: str | PathLike[str],
) -> Iterator[PronunciationRegressionResult]:
    yield from _read_models(Path(path), PronunciationRegressionResult)


def _find_spans(text: str, surface: str, text_form: str) -> tuple[TextSpan, ...]:
    spans: list[TextSpan] = []
    start = 0
    while True:
        index = text.find(surface, start)
        if index < 0:
            break
        spans.append(
            TextSpan(
                text_form=text_form,
                start=index,
                end=index + len(surface),
                text=surface,
            )
        )
        start = index + len(surface)
    return tuple(spans)


def _expectation_reference(
    *,
    sample_id: str,
    scope: ExpectationScope,
    surface: str | None,
    expected_reading: str,
    source: ExpectationSource,
    confirmation: ConfirmationStatus,
    provenance: str,
) -> ExpectedPronunciationReference:
    canonical = canonicalize_japanese_reading(expected_reading)
    if not canonical:
        raise ValueError("expected pronunciation canonicalized to empty text")
    reference_id = _canonical_sha256(
        {
            "sample_id": sample_id,
            "scope": scope.value,
            "surface": surface,
            "expected_reading": expected_reading,
            "expectation_source": source.value,
            "confirmation_status": confirmation.value,
            "provenance_reference": provenance,
        }
    )
    return ExpectedPronunciationReference(
        reference_id=reference_id,
        scope=scope,
        surface=surface,
        expected_reading=expected_reading,
        canonical_expected_reading=canonical,
        expectation_source=source,
        confirmation_status=confirmation,
        provenance_reference=provenance,
    )


def evaluate_expected_sequence(
    expected: ExpectedPronunciationReference | None,
    observed_canonical_reading: str,
) -> RiskAlignmentStatus:
    if expected is None:
        return RiskAlignmentStatus.ALIGNMENT_UNRESOLVED
    if expected.canonical_expected_reading in observed_canonical_reading:
        return RiskAlignmentStatus.EXPECTED_SEQUENCE_OBSERVED
    return RiskAlignmentStatus.EXPECTED_SEQUENCE_NOT_OBSERVED


def pronunciation_decision_for_risks(
    risks: Iterable[PronunciationRiskEvidence],
) -> PronunciationEvidenceDecision:
    if any(
        risk.alignment_status
        in {
            RiskAlignmentStatus.EXPECTED_SEQUENCE_NOT_OBSERVED,
            RiskAlignmentStatus.ALIGNMENT_UNRESOLVED,
        }
        for risk in risks
    ):
        return PronunciationEvidenceDecision.REVIEW
    return PronunciationEvidenceDecision.NO_REVIEW_NEEDED


class PronunciationEvidencePipeline:
    def __init__(
        self,
        *,
        output_root: Path,
        policy: PronunciationEvidencePolicy | None = None,
    ) -> None:
        self.output_root = Path(output_root).resolve()
        self.policy = policy or PronunciationEvidencePolicy()
        self.policy_fingerprint = pronunciation_policy_fingerprint(self.policy)

    def build(
        self,
        *,
        artifact_manifest: Path,
        synthesis_runs_manifest: Path,
        qa_results_manifest: Path,
        qa_pass_manifest: Path,
        curation_report: Path,
    ) -> PronunciationEvidenceOutputs:
        artifacts = _unique_index(
            _read_models(Path(artifact_manifest), GeneratedAudioArtifact),
            "run_id",
            "generated artifact",
        )
        runs = _unique_index(
            iter_synthesis_runs(synthesis_runs_manifest), "run_id", "synthesis run"
        )
        qa_results = _unique_index(
            iter_qa_results(qa_results_manifest), "qa_id", "Stage 9 QA result"
        )
        qa_pass = _unique_index(
            _read_models(Path(qa_pass_manifest), QAPassedAudioArtifact),
            "qa_id",
            "Stage 9 QA-pass artifact",
        )
        curations = _unique_index(
            iter_tts_text_curation_reports(curation_report),
            "sample_id",
            "Stage 7 curation report",
        )

        evidence: list[PronunciationEvidenceResult] = []
        for qa_pass_item in sorted(
            qa_pass.values(), key=lambda item: (item.sample_id, item.run_id)
        ):
            artifact = artifacts.get(qa_pass_item.run_id)
            run = runs.get(qa_pass_item.run_id)
            qa_result = qa_results.get(qa_pass_item.qa_id)
            curation = curations.get(qa_pass_item.sample_id)
            if artifact is None or run is None or qa_result is None or curation is None:
                raise ValueError(
                    f"incomplete pronunciation provenance for {qa_pass_item.sample_id}"
                )
            evidence.append(
                self._build_one(
                    qa_pass_item=qa_pass_item,
                    artifact=artifact,
                    run=run,
                    qa_result=qa_result,
                    curation=curation,
                )
            )

        evidence_path = self.output_root / "pronunciation_evidence.jsonl"
        review_path = self.output_root / "pronunciation_review_candidates.jsonl"
        _write_models(evidence_path, evidence, "evidence_id")
        review = [
            item
            for item in evidence
            if item.decision is PronunciationEvidenceDecision.REVIEW
        ]
        _write_models(review_path, review, "evidence_id")
        return PronunciationEvidenceOutputs(
            evidence_manifest=evidence_path,
            review_candidates=review_path,
            total=len(evidence),
            review_count=len(review),
        )

    def _build_one(
        self,
        *,
        qa_pass_item: QAPassedAudioArtifact,
        artifact: GeneratedAudioArtifact,
        run: SynthesisRun,
        qa_result: GeneratedAudioQAResult,
        curation: TTSTextCurationReport,
    ) -> PronunciationEvidenceResult:
        self._validate_chain(qa_pass_item, artifact, run, qa_result, curation)
        assert qa_result.canonical_observed_reading is not None
        assert qa_result.asr_transcript is not None
        assert qa_result.observed_reading_kana is not None
        assert qa_result.edit_distance is not None
        assert qa_result.cer is not None
        assert curation.policy_fingerprint_sha256 is not None
        assert curation.preparation_policy_version is not None
        assert curation.override_fingerprint_sha256 is not None

        provider = curation.reading_provider
        provider_identity = (
            f"{provider.provider}:{provider.provider_version}"
            if provider is not None
            else "unknown-reading-provider"
        )
        references: list[ExpectedPronunciationReference] = [
            _expectation_reference(
                sample_id=artifact.sample_id,
                scope=ExpectationScope.FULL_TEXT,
                surface=None,
                expected_reading=artifact.expected_reading_kana,
                source=ExpectationSource.READING_PROVIDER,
                confirmation=ConfirmationStatus.DERIVED,
                provenance=(
                    f"stage7:{curation.sample_id}:reading-provider:{provider_identity}"
                ),
            )
        ]
        override_references: dict[str, ExpectedPronunciationReference] = {}
        for override in artifact.applied_pronunciation_overrides:
            reference = _expectation_reference(
                sample_id=artifact.sample_id,
                scope=ExpectationScope.RISK_SPAN,
                surface=override.surface,
                expected_reading=override.reading_kana,
                source=ExpectationSource.CONFIRMED_OVERRIDE,
                confirmation=ConfirmationStatus.CONFIG_CONFIRMED,
                provenance=override.source,
            )
            references.append(reference)
            override_references[override.surface] = reference

        risk_evidence = self._build_risks(
            artifact=artifact,
            curation=curation,
            observed_canonical=qa_result.canonical_observed_reading,
            override_references=override_references,
        )
        review_reasons = tuple(
            f"{risk.surface}:{risk.alignment_status.value}"
            for risk in risk_evidence
            if risk.alignment_status
            in {
                RiskAlignmentStatus.EXPECTED_SEQUENCE_NOT_OBSERVED,
                RiskAlignmentStatus.ALIGNMENT_UNRESOLVED,
            }
        )
        decision = pronunciation_decision_for_risks(risk_evidence)
        observed = ObservedPronunciationEvidence(
            stage9_qa_id=qa_result.qa_id,
            asr_transcript=qa_result.asr_transcript,
            observed_reading_kana=qa_result.observed_reading_kana,
            canonical_observed_reading=qa_result.canonical_observed_reading,
            stage9_decision=qa_result.decision,
            stage9_issue_codes=qa_result.decision_reasons,
            stage9_edit_distance=qa_result.edit_distance,
            stage9_cer=qa_result.cer,
        )
        payload = {
            "sample_id": artifact.sample_id,
            "run_id": artifact.run_id,
            "qa_id": qa_result.qa_id,
            "audio_sha256": artifact.audio_sha256,
            "policy_fingerprint": self.policy_fingerprint,
            "references": [item.model_dump(mode="json") for item in references],
            "risks": [item.model_dump(mode="json") for item in risk_evidence],
        }
        return PronunciationEvidenceResult(
            evidence_id=_canonical_sha256(payload),
            logical_case_id=artifact.sample_id,
            sample_id=artifact.sample_id,
            artifact_run_id=artifact.run_id,
            synthesis_run_id=run.run_id,
            stage9_qa_id=qa_result.qa_id,
            policy_version=self.policy.policy_version,
            policy_fingerprint_sha256=self.policy_fingerprint,
            decision=decision,
            review_reasons=review_reasons,
            source_text=curation.raw_text,
            normalized_text=artifact.input_text,
            synthesis_input=artifact.synthesis_text,
            normalization_transformations=(
                curation.normalization.transformations if curation.normalization else ()
            ),
            expected_references=tuple(references),
            risk_evidence=risk_evidence,
            observed_evidence=observed,
            audio_sha256=artifact.audio_sha256,
            tts_backend=artifact.backend,
            tts_model_name=artifact.model_name,
            speaker=artifact.speaker,
            synthesis_strategy=artifact.synthesis_text_strategy,
            synthesis_config_fingerprint_sha256=run.config_fingerprint_sha256,
            stage7_policy_version=curation.policy_version,
            stage7_policy_fingerprint_sha256=curation.policy_fingerprint_sha256,
            preparation_policy_version=curation.preparation_policy_version,
            watchlist_version=curation.watchlist_version,
            watchlist_fingerprint_sha256=curation.watchlist_fingerprint_sha256,
            override_fingerprint_sha256=curation.override_fingerprint_sha256,
            stage9_policy_version=qa_result.policy_version,
            stage9_policy_fingerprint_sha256=qa_result.policy_fingerprint_sha256,
            stage9_asr_model_identity=qa_result.asr_model_identity,
            stage9_asr_config_fingerprint_sha256=(
                qa_result.asr_config_fingerprint_sha256
            ),
            stage9_reading_provider_fingerprint_sha256=(
                qa_result.reading_provider_fingerprint_sha256
            ),
        )

    def _build_risks(
        self,
        *,
        artifact: GeneratedAudioArtifact,
        curation: TTSTextCurationReport,
        observed_canonical: str,
        override_references: dict[str, ExpectedPronunciationReference],
    ) -> tuple[PronunciationRiskEvidence, ...]:
        risks: list[PronunciationRiskEvidence] = []
        for match in curation.risk_terms:
            spans = self._all_spans(artifact, curation, match.surface)
            risks.append(
                self._risk(
                    artifact=artifact,
                    surface=match.surface,
                    risk_type=match.category,
                    reason=match.reason,
                    occurrence_count=match.occurrence_count,
                    evidence_source=RiskEvidenceSource.STAGE7_WATCHLIST,
                    spans=spans,
                    expected=None,
                    observed_canonical=observed_canonical,
                )
            )

        resolved_by_surface = {
            match.surface: match for match in curation.override_resolved_risk_terms
        }
        for override in artifact.applied_pronunciation_overrides:
            match = resolved_by_surface.get(override.surface)
            risks.append(
                self._risk(
                    artifact=artifact,
                    surface=override.surface,
                    risk_type=(match.category if match else "confirmed_override_surface"),
                    reason=(match.reason if match else "confirmed_override_applied"),
                    occurrence_count=override.occurrences,
                    evidence_source=(
                        RiskEvidenceSource.STAGE7_WATCHLIST
                        if match
                        else RiskEvidenceSource.CONFIRMED_OVERRIDE_PROVENANCE
                    ),
                    spans=self._all_spans(artifact, curation, override.surface),
                    expected=override_references[override.surface],
                    observed_canonical=observed_canonical,
                )
            )
        return tuple(sorted(risks, key=lambda item: (item.surface, item.risk_id)))

    @staticmethod
    def _all_spans(
        artifact: GeneratedAudioArtifact,
        curation: TTSTextCurationReport,
        surface: str,
    ) -> tuple[TextSpan, ...]:
        return (
            *_find_spans(curation.raw_text, surface, "source"),
            *_find_spans(artifact.input_text, surface, "normalized"),
            *_find_spans(artifact.synthesis_text, surface, "synthesis"),
        )

    @staticmethod
    def _risk(
        *,
        artifact: GeneratedAudioArtifact,
        surface: str,
        risk_type: str,
        reason: str,
        occurrence_count: int,
        evidence_source: RiskEvidenceSource,
        spans: tuple[TextSpan, ...],
        expected: ExpectedPronunciationReference | None,
        observed_canonical: str,
    ) -> PronunciationRiskEvidence:
        status = evaluate_expected_sequence(expected, observed_canonical)
        if status is RiskAlignmentStatus.ALIGNMENT_UNRESOLVED:
            statement = (
                "No confirmed span reading is available; round-trip alignment is unresolved."
            )
        elif status is RiskAlignmentStatus.EXPECTED_SEQUENCE_OBSERVED:
            statement = (
                "The expected canonical sequence occurs in the full observed round-trip "
                "reading; this is supporting evidence, not proof of pronunciation."
            )
        else:
            statement = (
                "The expected canonical sequence was not found in the full observed "
                "round-trip reading; human review is required."
            )
        risk_id = _canonical_sha256(
            {
                "run_id": artifact.run_id,
                "surface": surface,
                "risk_type": risk_type,
                "reason": reason,
                "expected_reference_id": expected.reference_id if expected else None,
                "alignment_status": status.value,
            }
        )
        return PronunciationRiskEvidence(
            risk_id=risk_id,
            surface=surface,
            risk_type=risk_type,
            reason=reason,
            occurrence_count=occurrence_count,
            evidence_source=evidence_source,
            spans=spans,
            expected_reference_id=expected.reference_id if expected else None,
            alignment_status=status,
            evidence_statement=statement,
        )

    @staticmethod
    def _validate_chain(
        qa_pass_item: QAPassedAudioArtifact,
        artifact: GeneratedAudioArtifact,
        run: SynthesisRun,
        qa_result: GeneratedAudioQAResult,
        curation: TTSTextCurationReport,
    ) -> None:
        if qa_result.status is not QAExecutionStatus.COMPLETED:
            raise ValueError("Stage 10A requires completed Stage 9 evidence")
        if qa_result.decision is not QADecision.PASS:
            raise ValueError("Stage 10A build input must be Stage 9 QA-pass audio")
        if run.status is not SynthesisRunStatus.SUCCESS:
            raise ValueError("Stage 10A requires a successful synthesis run")
        if curation.status not in {
            TTSTextCurationStatus.CURATED,
            TTSTextCurationStatus.SKIPPED,
        } or curation.decision is not CurationDecision.PASS:
            raise ValueError("Stage 10A requires Stage 7 CURATED/SKIPPED + PASS provenance")
        identifiers = {
            artifact.sample_id,
            run.sample_id,
            qa_result.sample_id,
            qa_pass_item.sample_id,
            curation.sample_id,
        }
        if len(identifiers) != 1:
            raise ValueError("sample identifiers disagree across Stage 7-10A provenance")
        run_ids = {artifact.run_id, run.run_id, qa_result.run_id, qa_pass_item.run_id}
        if len(run_ids) != 1:
            raise ValueError("run identifiers disagree across Stage 8-10A provenance")
        if qa_pass_item.qa_id != qa_result.qa_id:
            raise ValueError("Stage 9 QA identifiers disagree")
        hashes = {
            artifact.audio_sha256,
            run.audio_sha256,
            qa_result.observed_audio_sha256,
            qa_pass_item.audio_sha256,
        }
        if len(hashes) != 1 or None in hashes:
            raise ValueError("audio hashes disagree across Stage 8-10A provenance")
        if artifact.expected_reading_kana != qa_pass_item.expected_reading_kana:
            raise ValueError("Stage 8 and Stage 9 expected readings disagree")
        if artifact.synthesis_text != run.synthesis_text:
            raise ValueError("artifact and synthesis-run input text disagree")


def capture_pronunciation_baseline(
    *, evidence_manifest: Path, output_path: Path
) -> int:
    evidence = list(iter_pronunciation_evidence(evidence_manifest))
    baselines = [_baseline_from_evidence(item) for item in evidence]
    return _write_models(Path(output_path), baselines, "logical_case_id")


def _baseline_from_evidence(
    evidence: PronunciationEvidenceResult,
) -> PronunciationRegressionBaseline:
    full_reference = next(
        item
        for item in evidence.expected_references
        if item.scope is ExpectationScope.FULL_TEXT
    )
    snapshot = Stage9ComparisonSnapshot(
        qa_id=evidence.stage9_qa_id,
        decision=evidence.observed_evidence.stage9_decision,
        canonical_expected_reading=full_reference.canonical_expected_reading,
        canonical_observed_reading=evidence.observed_evidence.canonical_observed_reading,
        edit_distance=evidence.observed_evidence.stage9_edit_distance or 0,
        cer=evidence.observed_evidence.stage9_cer or 0.0,
        issue_codes=evidence.observed_evidence.stage9_issue_codes,
    )
    payload = {
        "logical_case_id": evidence.logical_case_id,
        "source_evidence_id": evidence.evidence_id,
        "expected_references": [
            item.model_dump(mode="json") for item in evidence.expected_references
        ],
        "risk_evidence": [item.model_dump(mode="json") for item in evidence.risk_evidence],
        "observed": evidence.observed_evidence.canonical_observed_reading,
        "provenance": {
            "tts_backend": evidence.tts_backend,
            "tts_model_name": evidence.tts_model_name,
            "speaker": evidence.speaker,
            "synthesis_strategy": evidence.synthesis_strategy,
            "synthesis_config": evidence.synthesis_config_fingerprint_sha256,
            "stage7_policy": evidence.stage7_policy_fingerprint_sha256,
            "watchlist": evidence.watchlist_fingerprint_sha256,
            "overrides": evidence.override_fingerprint_sha256,
            "stage9_policy": evidence.stage9_policy_fingerprint_sha256,
            "stage9_asr_model": evidence.stage9_asr_model_identity,
            "stage9_asr_config": evidence.stage9_asr_config_fingerprint_sha256,
            "stage9_reading_provider": (
                evidence.stage9_reading_provider_fingerprint_sha256
            ),
            "stage10a_policy": evidence.policy_fingerprint_sha256,
        },
    }
    return PronunciationRegressionBaseline(
        baseline_id=_canonical_sha256(payload),
        logical_case_id=evidence.logical_case_id,
        source_evidence_id=evidence.evidence_id,
        source_text=evidence.source_text,
        expected_references=evidence.expected_references,
        risk_evidence=evidence.risk_evidence,
        observed_canonical_reading=evidence.observed_evidence.canonical_observed_reading,
        stage9_comparison=snapshot,
        audio_sha256=evidence.audio_sha256,
        tts_backend=evidence.tts_backend,
        tts_model_name=evidence.tts_model_name,
        speaker=evidence.speaker,
        synthesis_strategy=evidence.synthesis_strategy,
        synthesis_config_fingerprint_sha256=evidence.synthesis_config_fingerprint_sha256,
        stage7_policy_version=evidence.stage7_policy_version,
        stage7_policy_fingerprint_sha256=evidence.stage7_policy_fingerprint_sha256,
        preparation_policy_version=evidence.preparation_policy_version,
        watchlist_version=evidence.watchlist_version,
        watchlist_fingerprint_sha256=evidence.watchlist_fingerprint_sha256,
        override_fingerprint_sha256=evidence.override_fingerprint_sha256,
        stage9_policy_version=evidence.stage9_policy_version,
        stage9_policy_fingerprint_sha256=evidence.stage9_policy_fingerprint_sha256,
        stage9_asr_model_identity=evidence.stage9_asr_model_identity,
        stage9_asr_config_fingerprint_sha256=(
            evidence.stage9_asr_config_fingerprint_sha256
        ),
        stage9_reading_provider_fingerprint_sha256=(
            evidence.stage9_reading_provider_fingerprint_sha256
        ),
        stage10a_policy_version=evidence.policy_version,
        stage10a_policy_fingerprint_sha256=evidence.policy_fingerprint_sha256,
    )


def compare_pronunciation_regression(
    *,
    evidence_manifest: Path,
    baseline_manifest: Path,
    output_root: Path,
    policy: PronunciationEvidencePolicy | None = None,
) -> PronunciationRegressionOutputs:
    active_policy = policy or PronunciationEvidencePolicy()
    fingerprint = pronunciation_policy_fingerprint(active_policy)
    current = _unique_index(
        iter_pronunciation_evidence(evidence_manifest),
        "logical_case_id",
        "current pronunciation case",
    )
    baseline = _unique_index(
        iter_pronunciation_baselines(baseline_manifest),
        "logical_case_id",
        "pronunciation baseline case",
    )
    results: list[PronunciationRegressionResult] = []
    for case_id in sorted(set(current) | set(baseline)):
        current_item = current.get(case_id)
        baseline_item = baseline.get(case_id)
        changes: list[RegressionChange]
        if baseline_item is None:
            changes = [
                RegressionChange(
                    kind=RegressionChangeKind.BASELINE_CASE_MISSING,
                    field="logical_case_id",
                    baseline_value=None,
                    current_value=case_id,
                    material_pronunciation_change=True,
                    message="Current evidence has no regression baseline.",
                )
            ]
        elif current_item is None:
            changes = [
                RegressionChange(
                    kind=RegressionChangeKind.CURRENT_CASE_MISSING,
                    field="logical_case_id",
                    baseline_value=case_id,
                    current_value=None,
                    material_pronunciation_change=True,
                    message="A baseline case is absent from current evidence.",
                )
            ]
        else:
            changes = _compare_case(baseline_item, current_item)
        material = any(change.material_pronunciation_change for change in changes)
        regression_id = _canonical_sha256(
            {
                "logical_case_id": case_id,
                "baseline_id": baseline_item.baseline_id if baseline_item else None,
                "current_evidence_id": current_item.evidence_id if current_item else None,
                "policy_fingerprint": fingerprint,
                "changes": [change.model_dump(mode="json") for change in changes],
            }
        )
        results.append(
            PronunciationRegressionResult(
                regression_id=regression_id,
                logical_case_id=case_id,
                baseline_id=baseline_item.baseline_id if baseline_item else None,
                current_evidence_id=current_item.evidence_id if current_item else None,
                decision=(
                    RegressionDecision.REVIEW
                    if material
                    else RegressionDecision.NO_REGRESSION_DETECTED
                ),
                changes=tuple(changes),
                policy_version=active_policy.policy_version,
                policy_fingerprint_sha256=fingerprint,
            )
        )

    destination = Path(output_root).resolve()
    results_path = destination / "pronunciation_regression_results.jsonl"
    review_path = destination / "pronunciation_regression_review_candidates.jsonl"
    summary_path = destination / "pronunciation_regression_summary.json"
    _write_models(results_path, results, "regression_id")
    review = [item for item in results if item.decision is RegressionDecision.REVIEW]
    _write_models(review_path, review, "regression_id")
    summary = PronunciationRegressionSummary(
        policy_version=active_policy.policy_version,
        policy_fingerprint_sha256=fingerprint,
        baseline_cases=len(baseline),
        current_cases=len(current),
        compared_cases=len(results),
        no_regression_detected=sum(
            item.decision is RegressionDecision.NO_REGRESSION_DETECTED
            for item in results
        ),
        review_candidates=len(review),
        provenance_only_changes=sum(
            bool(item.changes)
            and not any(change.material_pronunciation_change for change in item.changes)
            for item in results
        ),
    )
    destination.mkdir(parents=True, exist_ok=True)
    summary_path.write_text(summary.model_dump_json(indent=2) + "\n", encoding="utf-8")
    return PronunciationRegressionOutputs(
        results_manifest=results_path,
        review_candidates=review_path,
        summary_file=summary_path,
        summary=summary,
    )


def _expectation_signature(
    references: tuple[ExpectedPronunciationReference, ...],
) -> list[dict[str, str]]:
    return [
        {
            "scope": reference.scope.value,
            "surface": reference.surface or "",
            "canonical_expected_reading": reference.canonical_expected_reading,
            "expectation_source": reference.expectation_source.value,
            "confirmation_status": reference.confirmation_status.value,
        }
        for reference in references
    ]


def _risk_signature(risks: tuple[PronunciationRiskEvidence, ...]) -> list[dict[str, str]]:
    return [
        {
            "surface": risk.surface,
            "risk_type": risk.risk_type,
            "alignment_status": risk.alignment_status.value,
        }
        for risk in risks
    ]


def _change(
    kind: RegressionChangeKind,
    field: str,
    baseline_value: JsonValue,
    current_value: JsonValue,
    *,
    material: bool,
    message: str,
) -> RegressionChange:
    return RegressionChange(
        kind=kind,
        field=field,
        baseline_value=baseline_value,
        current_value=current_value,
        material_pronunciation_change=material,
        message=message,
    )


def _compare_case(
    baseline: PronunciationRegressionBaseline,
    current: PronunciationEvidenceResult,
) -> list[RegressionChange]:
    changes: list[RegressionChange] = []
    if baseline.source_text != current.source_text:
        changes.append(
            _change(
                RegressionChangeKind.SOURCE_TEXT_CHANGED,
                "source_text",
                baseline.source_text,
                current.source_text,
                material=True,
                message="The logical case source text changed.",
            )
        )
    baseline_expected = _expectation_signature(baseline.expected_references)
    current_expected = _expectation_signature(current.expected_references)
    if baseline_expected != current_expected:
        changes.append(
            _change(
                RegressionChangeKind.EXPECTED_READING_CHANGED,
                "expected_references",
                baseline_expected,
                current_expected,
                material=True,
                message="Expected canonical pronunciation evidence changed.",
            )
        )
    current_observed = current.observed_evidence.canonical_observed_reading
    if baseline.observed_canonical_reading != current_observed:
        changes.append(
            _change(
                RegressionChangeKind.OBSERVED_READING_CHANGED,
                "canonical_observed_reading",
                baseline.observed_canonical_reading,
                current_observed,
                material=True,
                message="Observed round-trip canonical reading changed.",
            )
        )
    baseline_risks = _risk_signature(baseline.risk_evidence)
    current_risks = _risk_signature(current.risk_evidence)
    if baseline_risks != current_risks:
        changes.append(
            _change(
                RegressionChangeKind.RISK_STATUS_CHANGED,
                "risk_evidence",
                baseline_risks,
                current_risks,
                material=True,
                message="Pronunciation risk spans or alignment status changed.",
            )
        )

    provenance_pairs = {
        "tts_backend": (baseline.tts_backend, current.tts_backend),
        "tts_model_name": (baseline.tts_model_name, current.tts_model_name),
        "speaker": (baseline.speaker, current.speaker),
        "synthesis_strategy": (baseline.synthesis_strategy, current.synthesis_strategy),
        "synthesis_config_fingerprint_sha256": (
            baseline.synthesis_config_fingerprint_sha256,
            current.synthesis_config_fingerprint_sha256,
        ),
        "stage7_policy_fingerprint_sha256": (
            baseline.stage7_policy_fingerprint_sha256,
            current.stage7_policy_fingerprint_sha256,
        ),
        "watchlist_fingerprint_sha256": (
            baseline.watchlist_fingerprint_sha256,
            current.watchlist_fingerprint_sha256,
        ),
        "override_fingerprint_sha256": (
            baseline.override_fingerprint_sha256,
            current.override_fingerprint_sha256,
        ),
        "stage9_policy_fingerprint_sha256": (
            baseline.stage9_policy_fingerprint_sha256,
            current.stage9_policy_fingerprint_sha256,
        ),
        "stage9_asr_model_identity": (
            baseline.stage9_asr_model_identity,
            current.stage9_asr_model_identity,
        ),
        "stage9_asr_config_fingerprint_sha256": (
            baseline.stage9_asr_config_fingerprint_sha256,
            current.stage9_asr_config_fingerprint_sha256,
        ),
        "stage9_reading_provider_fingerprint_sha256": (
            baseline.stage9_reading_provider_fingerprint_sha256,
            current.stage9_reading_provider_fingerprint_sha256,
        ),
        "stage10a_policy_fingerprint_sha256": (
            baseline.stage10a_policy_fingerprint_sha256,
            current.policy_fingerprint_sha256,
        ),
    }
    for field, (old, new) in provenance_pairs.items():
        if old != new:
            changes.append(
                _change(
                    RegressionChangeKind.PROVENANCE_CONFIG_CHANGED,
                    field,
                    old,
                    new,
                    material=False,
                    message=(
                        "Configuration provenance changed; this alone is not a "
                        "pronunciation regression."
                    ),
                )
            )
    if baseline.audio_sha256 != current.audio_sha256:
        changes.append(
            _change(
                RegressionChangeKind.ARTIFACT_IDENTITY_CHANGED,
                "audio_sha256",
                baseline.audio_sha256,
                current.audio_sha256,
                material=False,
                message=(
                    "Artifact identity changed; WAV byte differences alone are not a "
                    "pronunciation regression."
                ),
            )
        )
    return changes
