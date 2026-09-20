"""Stage 3 batch ASR orchestration and operational reports."""

from __future__ import annotations

from collections.abc import Iterator
from dataclasses import dataclass
from enum import Enum
from os import PathLike
from pathlib import Path

from pydantic import BaseModel, ConfigDict, Field, JsonValue

from .asr import ASRAdapter, ASRResult
from .audio import AudioValidationError, decode_audio
from .lifecycle import transition_stage
from .manifest import iter_manifest, write_manifest
from .schemas import CurationDecision, ProcessingStage, SpeechSample


class ASRRunStatus(str, Enum):
    TRANSCRIBED = "transcribed"
    SKIPPED = "skipped"
    FAILED = "failed"


class ASRRunError(BaseModel):
    model_config = ConfigDict(extra="forbid")

    code: str = Field(min_length=1)
    message: str = Field(min_length=1)


class ASRRunReport(BaseModel):
    model_config = ConfigDict(extra="forbid")

    sample_id: str = Field(min_length=1)
    status: ASRRunStatus
    source_audio_path: str = Field(min_length=1)
    backend: str = Field(min_length=1)
    backend_version: str = Field(min_length=1)
    model_name: str = Field(min_length=1)
    device: str = Field(min_length=1)
    compute_type: str = Field(min_length=1)
    requested_language: str = Field(min_length=1)
    effective_config: dict[str, JsonValue]
    detected_language: str | None = None
    language_probability: float | None = Field(default=None, ge=0, le=1)
    audio_duration_sec: float | None = Field(default=None, ge=0)
    transcription_segment_count: int = Field(default=0, ge=0)
    text_length_chars: int = Field(default=0, ge=0)
    errors: list[ASRRunError] = Field(default_factory=list)


@dataclass(frozen=True)
class ASRBatchSummary:
    input_samples: int
    transcribed: int
    skipped: int
    failed: int


@dataclass(frozen=True)
class ASRBatchOutputs:
    transcribed_manifest: Path
    report_file: Path
    summary: ASRBatchSummary


def write_asr_run_reports(
    path: str | PathLike[str], reports: list[ASRRunReport]
) -> int:
    destination = Path(path)
    with destination.open("w", encoding="utf-8", newline="\n") as handle:
        for report in reports:
            handle.write(report.model_dump_json())
            handle.write("\n")
    return len(reports)


def iter_asr_run_reports(path: str | PathLike[str]) -> Iterator[ASRRunReport]:
    report_path = Path(path)
    with report_path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            try:
                yield ASRRunReport.model_validate_json(line)
            except ValueError as exc:
                raise ValueError(
                    f"{report_path}:{line_number}: invalid ASR run report: {exc}"
                ) from exc


def _append_drop_reason(sample: SpeechSample, reason: str) -> SpeechSample:
    updated = sample.model_copy(deep=True)
    updated.state.curation_decision = CurationDecision.DROP
    if reason not in updated.state.decision_reasons:
        updated.state.decision_reasons.append(reason)
    return updated


class ASRBatchPipeline:
    """Apply one already-initialized model-neutral ASR adapter to a manifest."""

    def __init__(
        self,
        *,
        dataset_root: Path,
        output_root: Path,
        adapter: ASRAdapter,
    ) -> None:
        self.dataset_root = dataset_root.resolve()
        self.output_root = output_root.resolve()
        if not self.output_root.is_relative_to(self.dataset_root):
            raise ValueError("output_root must be inside dataset_root")
        self.adapter = adapter
        self.manifest_dir = self.output_root / "manifests"
        self.report_dir = self.output_root / "reports"

    def run(self, input_manifest: Path, *, force: bool = False) -> ASRBatchOutputs:
        samples = list(iter_manifest(input_manifest))
        self.manifest_dir.mkdir(parents=True, exist_ok=True)
        self.report_dir.mkdir(parents=True, exist_ok=True)
        output_manifest = self.manifest_dir / "transcribed_segments.jsonl"
        report_file = self.report_dir / "asr_run_report.jsonl"

        previous = self._load_previous(output_manifest) if not force else {}
        outputs: list[SpeechSample] = []
        reports: list[ASRRunReport] = []

        for sample in samples:
            reusable = previous.get(sample.sample_id)
            if (
                reusable is not None
                and reusable.audio.path == sample.audio.path
                and reusable.state.stage is ProcessingStage.TRANSCRIBED
            ):
                reused = sample.model_copy(deep=True)
                reused.text.asr_text = reusable.text.asr_text
                reused = transition_stage(reused, ProcessingStage.TRANSCRIBED)
                outputs.append(reused)
                reports.append(self._skip_report(reused))
                continue
            if sample.state.stage is ProcessingStage.TRANSCRIBED and not force:
                outputs.append(sample.model_copy(deep=True))
                reports.append(self._skip_report(sample))
                continue

            output, report = self._process_sample(sample)
            outputs.append(output)
            reports.append(report)

        write_manifest(output_manifest, outputs)
        write_asr_run_reports(report_file, reports)
        summary = ASRBatchSummary(
            input_samples=len(samples),
            transcribed=sum(r.status is ASRRunStatus.TRANSCRIBED for r in reports),
            skipped=sum(r.status is ASRRunStatus.SKIPPED for r in reports),
            failed=sum(r.status is ASRRunStatus.FAILED for r in reports),
        )
        return ASRBatchOutputs(output_manifest, report_file, summary)

    @staticmethod
    def _load_previous(path: Path) -> dict[str, SpeechSample]:
        if not path.exists():
            return {}
        samples = list(iter_manifest(path))
        return {sample.sample_id: sample for sample in samples}

    def _process_sample(
        self, sample: SpeechSample
    ) -> tuple[SpeechSample, ASRRunReport]:
        if sample.state.stage is not ProcessingStage.AUDIO_PREPARED and sample.state.stage is not ProcessingStage.TRANSCRIBED:
            return self._data_failure(
                sample,
                "asr_invalid_input_stage",
                f"ASR requires audio_prepared input, got {sample.state.stage.value}",
            )

        source_path = Path(sample.audio.path)
        resolved = source_path if source_path.is_absolute() else self.dataset_root / source_path
        try:
            decoded = decode_audio(resolved)
        except AudioValidationError as exc:
            return self._data_failure(
                sample,
                exc.code,
                f"audio validation failed for {sample.audio.path}",
            )

        result = self.adapter.transcribe(resolved)
        if not result.text.strip():
            return self._data_failure(
                sample,
                "asr_empty_transcript",
                "ASR completed without producing transcript text",
                audio_duration_sec=decoded.properties.duration_sec,
            )

        updated = sample.model_copy(deep=True)
        updated.text.asr_text = result.text
        updated = transition_stage(updated, ProcessingStage.TRANSCRIBED)
        return updated, self._success_report(sample, result, decoded.properties.duration_sec)

    def _base_report(self, sample: SpeechSample, status: ASRRunStatus) -> dict:
        runtime = self.adapter.runtime_info
        return {
            "sample_id": sample.sample_id,
            "status": status,
            "source_audio_path": sample.audio.path,
            "backend": runtime.backend_name,
            "backend_version": runtime.backend_version,
            "model_name": runtime.model_name,
            "device": runtime.device,
            "compute_type": runtime.compute_type,
            "requested_language": runtime.language,
            "effective_config": runtime.effective_config,
        }

    def _skip_report(self, sample: SpeechSample) -> ASRRunReport:
        return ASRRunReport(
            **self._base_report(sample, ASRRunStatus.SKIPPED),
            text_length_chars=len(sample.text.asr_text or ""),
        )

    def _success_report(
        self, sample: SpeechSample, result: ASRResult, audio_duration_sec: float
    ) -> ASRRunReport:
        return ASRRunReport(
            **self._base_report(sample, ASRRunStatus.TRANSCRIBED),
            detected_language=result.language,
            language_probability=result.language_probability,
            audio_duration_sec=audio_duration_sec,
            transcription_segment_count=len(result.segments),
            text_length_chars=len(result.text),
        )

    def _data_failure(
        self,
        sample: SpeechSample,
        code: str,
        message: str,
        *,
        audio_duration_sec: float | None = None,
    ) -> tuple[SpeechSample, ASRRunReport]:
        failed = _append_drop_reason(sample, code)
        report = ASRRunReport(
            **self._base_report(sample, ASRRunStatus.FAILED),
            audio_duration_sec=audio_duration_sec,
            errors=[ASRRunError(code=code, message=message)],
        )
        return failed, report
