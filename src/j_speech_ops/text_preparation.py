"""Stage 6 Japanese normalization and reading-preparation batch pipeline."""

from __future__ import annotations

import importlib.metadata
from collections.abc import Iterator
from dataclasses import dataclass
from enum import Enum
from os import PathLike
from pathlib import Path

from pydantic import BaseModel, ConfigDict, Field

from .japanese_normalization import (
    JAPANESE_TEXT_PREP_POLICY_VERSION,
    JapaneseNormalizer,
    NormalizationResult,
    NormalizationSampleError,
)
from .japanese_reading import (
    ReadingOverrides,
    ReadingProvider,
    ReadingProviderInfo,
    ReadingResult,
    ReadingSampleError,
)
from .schemas import CurationDecision
from .text_manifest import iter_text_manifest, write_text_manifest
from .tts_text import (
    TTSTextContent,
    TTSTextSample,
    TTSTextStage,
    transition_text_stage,
)


class TextPreparationStatus(str, Enum):
    READING_PREPARED = "reading_prepared"
    NORMALIZED_ONLY = "normalized_only"
    SKIPPED = "skipped"
    FAILED = "failed"


class TextPreparationIssue(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    code: str = Field(min_length=1)
    message: str = Field(min_length=1)


class TextPreparationDependencies(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    jaconv_version: str
    reading_provider: ReadingProviderInfo


class TextPreparationReport(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    sample_id: str
    policy_version: str
    status: TextPreparationStatus
    input_stage: TTSTextStage
    output_stage: TTSTextStage
    normalization: NormalizationResult | None = None
    reading: ReadingResult | None = None
    dependencies: TextPreparationDependencies
    warnings: tuple[TextPreparationIssue, ...] = ()
    errors: tuple[TextPreparationIssue, ...] = ()
    skipped_existing: bool = False


class TextPreparationBatchSummary(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    total_samples: int = Field(ge=0)
    reading_prepared: int = Field(ge=0)
    normalized_only: int = Field(ge=0)
    skipped: int = Field(ge=0)
    failed: int = Field(ge=0)


@dataclass(frozen=True)
class TextPreparationBatchOutputs:
    normalized_manifest: Path
    reading_prepared_manifest: Path
    report_file: Path
    summary: TextPreparationBatchSummary


def write_text_preparation_reports(
    path: str | PathLike[str], reports: list[TextPreparationReport]
) -> int:
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    with destination.open("w", encoding="utf-8", newline="\n") as handle:
        for report in reports:
            handle.write(report.model_dump_json())
            handle.write("\n")
    return len(reports)


def iter_text_preparation_reports(
    path: str | PathLike[str],
) -> Iterator[TextPreparationReport]:
    report_path = Path(path)
    with report_path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            try:
                yield TextPreparationReport.model_validate_json(line)
            except ValueError as exc:
                raise ValueError(
                    f"{report_path}:{line_number}: invalid text preparation report: {exc}"
                ) from exc


class TextPreparationPipeline:
    def __init__(
        self,
        *,
        output_root: Path,
        normalizer: JapaneseNormalizer,
        reading_provider: ReadingProvider,
        overrides: ReadingOverrides | None = None,
    ) -> None:
        self.output_root = Path(output_root).resolve()
        self.normalizer = normalizer
        self.reading_provider = reading_provider
        self.overrides = overrides or ReadingOverrides.empty()
        self.manifest_dir = self.output_root / "manifests"
        self.report_dir = self.output_root / "reports"
        self.dependencies = TextPreparationDependencies(
            jaconv_version=importlib.metadata.version("jaconv"),
            reading_provider=reading_provider.info,
        )

    @staticmethod
    def _replace_text(
        sample: TTSTextSample,
        *,
        normalized_text: str,
        reading_kana: str | None,
    ) -> TTSTextSample:
        content = TTSTextContent(
            raw_text=sample.text.raw_text,
            normalized_text=normalized_text,
            reading_kana=reading_kana,
            phonemes=None,
        )
        return sample.model_copy(update={"text": content}, deep=True)

    def _normalize(
        self, sample: TTSTextSample
    ) -> tuple[TTSTextSample, NormalizationResult]:
        result = self.normalizer.normalize(sample.text.raw_text)
        if sample.state.stage is TTSTextStage.RAW:
            transitioned = transition_text_stage(sample, TTSTextStage.NORMALIZED)
        else:
            transitioned = transition_text_stage(sample, sample.state.stage)
        return (
            self._replace_text(
                transitioned,
                normalized_text=result.normalized_text,
                reading_kana=None,
            ),
            result,
        )

    def _prepare_reading(
        self, sample: TTSTextSample
    ) -> tuple[TTSTextSample, ReadingResult]:
        normalized_text = sample.text.normalized_text
        if normalized_text is None or not normalized_text.strip():
            raise ReadingSampleError("normalized_text is unavailable")
        result = self.reading_provider.prepare(
            sample.sample_id, normalized_text, self.overrides
        )
        if sample.state.stage is TTSTextStage.NORMALIZED:
            transitioned = transition_text_stage(
                sample, TTSTextStage.READING_PREPARED
            )
        else:
            transitioned = transition_text_stage(sample, sample.state.stage)
        return (
            self._replace_text(
                transitioned,
                normalized_text=normalized_text,
                reading_kana=result.reading_kana,
            ),
            result,
        )

    def run(
        self, input_manifest: Path, *, force: bool = False
    ) -> TextPreparationBatchOutputs:
        samples = list(iter_text_manifest(input_manifest))
        normalized_samples: list[TTSTextSample] = []
        reading_samples: list[TTSTextSample] = []
        reports: list[TextPreparationReport] = []

        for sample in samples:
            input_stage = sample.state.stage
            if input_stage is TTSTextStage.CURATED:
                raise ValueError(
                    f"sample {sample.sample_id} is already beyond Stage 6"
                )
            if sample.text.phonemes is not None:
                reports.append(
                    self._failed_report(
                        sample,
                        "phonemes_already_set",
                        "Stage 6 requires phonemes to remain unset",
                    )
                )
                continue
            if sample.state.curation_decision is not CurationDecision.UNDECIDED:
                reports.append(
                    self._failed_report(
                        sample,
                        "curation_already_decided",
                        "Stage 6 does not process text with a curation decision",
                    )
                )
                continue
            if input_stage is TTSTextStage.READING_PREPARED and not force:
                if (
                    sample.text.reading_kana is None
                    or not sample.text.reading_kana.strip()
                ):
                    reports.append(
                        self._failed_report(
                            sample,
                            "reading_kana_missing",
                            "READING_PREPARED sample has no reading_kana",
                        )
                    )
                    continue
                reading_samples.append(sample)
                reports.append(
                    TextPreparationReport(
                        sample_id=sample.sample_id,
                        policy_version=JAPANESE_TEXT_PREP_POLICY_VERSION,
                        status=TextPreparationStatus.SKIPPED,
                        input_stage=input_stage,
                        output_stage=input_stage,
                        dependencies=self.dependencies,
                        warnings=(
                            TextPreparationIssue(
                                code="existing_reading_prepared_reused",
                                message="reading-prepared sample was not recomputed",
                            ),
                        ),
                        skipped_existing=True,
                    )
                )
                continue

            normalization: NormalizationResult | None = None
            working = sample
            if input_stage is TTSTextStage.RAW or force:
                try:
                    working, normalization = self._normalize(sample)
                except NormalizationSampleError as exc:
                    if input_stage is TTSTextStage.READING_PREPARED:
                        reading_samples.append(sample)
                        reports.append(
                            TextPreparationReport(
                                sample_id=sample.sample_id,
                                policy_version=JAPANESE_TEXT_PREP_POLICY_VERSION,
                                status=TextPreparationStatus.FAILED,
                                input_stage=input_stage,
                                output_stage=input_stage,
                                dependencies=self.dependencies,
                                errors=(
                                    TextPreparationIssue(
                                        code="normalization_recompute_failed_existing_preserved",
                                        message=str(exc),
                                    ),
                                ),
                            )
                        )
                        continue
                    reports.append(
                        self._failed_report(
                            sample, "normalization_failed", str(exc)
                        )
                    )
                    continue
            elif input_stage is TTSTextStage.NORMALIZED:
                if (
                    sample.text.normalized_text is None
                    or not sample.text.normalized_text.strip()
                ):
                    reports.append(
                        self._failed_report(
                            sample,
                            "normalized_text_missing",
                            "NORMALIZED sample has no normalized_text",
                        )
                    )
                    continue

            if working.state.stage is TTSTextStage.NORMALIZED:
                normalized_samples.append(working)

            try:
                prepared, reading = self._prepare_reading(working)
            except ReadingSampleError as exc:
                if input_stage is TTSTextStage.READING_PREPARED:
                    reading_samples.append(sample)
                    reports.append(
                        TextPreparationReport(
                            sample_id=sample.sample_id,
                            policy_version=JAPANESE_TEXT_PREP_POLICY_VERSION,
                            status=TextPreparationStatus.FAILED,
                            input_stage=input_stage,
                            output_stage=input_stage,
                            normalization=normalization,
                            dependencies=self.dependencies,
                            errors=(
                                TextPreparationIssue(
                                    code="reading_recompute_failed_existing_preserved",
                                    message=str(exc),
                                ),
                            ),
                        )
                    )
                    continue
                reports.append(
                    TextPreparationReport(
                        sample_id=sample.sample_id,
                        policy_version=JAPANESE_TEXT_PREP_POLICY_VERSION,
                        status=TextPreparationStatus.NORMALIZED_ONLY,
                        input_stage=input_stage,
                        output_stage=working.state.stage,
                        normalization=normalization,
                        dependencies=self.dependencies,
                        errors=(
                            TextPreparationIssue(
                                code="reading_failed", message=str(exc)
                            ),
                        ),
                    )
                )
                continue

            reading_samples.append(prepared)
            reports.append(
                TextPreparationReport(
                    sample_id=sample.sample_id,
                    policy_version=JAPANESE_TEXT_PREP_POLICY_VERSION,
                    status=TextPreparationStatus.READING_PREPARED,
                    input_stage=input_stage,
                    output_stage=prepared.state.stage,
                    normalization=normalization,
                    reading=reading,
                    dependencies=self.dependencies,
                )
            )

        normalized_manifest = self.manifest_dir / "normalized_text_samples.jsonl"
        reading_manifest = (
            self.manifest_dir / "reading_prepared_text_samples.jsonl"
        )
        report_file = self.report_dir / "text_preparation_report.jsonl"
        write_text_manifest(normalized_manifest, normalized_samples)
        write_text_manifest(reading_manifest, reading_samples)
        write_text_preparation_reports(report_file, reports)
        summary = TextPreparationBatchSummary(
            total_samples=len(samples),
            reading_prepared=sum(
                report.status is TextPreparationStatus.READING_PREPARED
                for report in reports
            ),
            normalized_only=sum(
                report.status is TextPreparationStatus.NORMALIZED_ONLY
                for report in reports
            ),
            skipped=sum(
                report.status is TextPreparationStatus.SKIPPED
                for report in reports
            ),
            failed=sum(
                report.status is TextPreparationStatus.FAILED
                for report in reports
            ),
        )
        return TextPreparationBatchOutputs(
            normalized_manifest=normalized_manifest,
            reading_prepared_manifest=reading_manifest,
            report_file=report_file,
            summary=summary,
        )

    def _failed_report(
        self, sample: TTSTextSample, code: str, message: str
    ) -> TextPreparationReport:
        return TextPreparationReport(
            sample_id=sample.sample_id,
            policy_version=JAPANESE_TEXT_PREP_POLICY_VERSION,
            status=TextPreparationStatus.FAILED,
            input_stage=sample.state.stage,
            output_stage=sample.state.stage,
            dependencies=self.dependencies,
            errors=(TextPreparationIssue(code=code, message=message),),
        )
