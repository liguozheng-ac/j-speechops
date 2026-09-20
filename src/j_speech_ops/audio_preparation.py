"""Stage 2 batch orchestration, segmentation, manifests, and operational reports."""

from __future__ import annotations

import hashlib
import re
from dataclasses import dataclass
from enum import Enum
from collections.abc import Iterator
from os import PathLike
from pathlib import Path
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field

from .audio import (
    AudioProperties,
    AudioValidationError,
    decode_audio,
    prepare_audio,
    write_pcm16_wav,
)
from .lifecycle import transition_stage
from .manifest import iter_manifest, write_manifest
from .schemas import (
    AudioInfo,
    CurationDecision,
    LineageInfo,
    ProcessingStage,
    ProcessingState,
    SpeechSample,
    TextInfo,
)
from .vad import SpeechRegion, VADConfig, VoiceActivityDetector


class AudioPreparationConfig(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    target_sample_rate_hz: Literal[16000] = 16_000
    target_channels: Literal[1] = 1
    output_format: Literal["WAV"] = "WAV"
    output_subtype: Literal["PCM_16"] = "PCM_16"
    vad: VADConfig = Field(default_factory=VADConfig)


class AudioPreparationStatus(str, Enum):
    PREPARED = "prepared"
    DROPPED = "dropped"


class ReportAudioProperties(BaseModel):
    model_config = ConfigDict(extra="forbid")

    sample_rate_hz: int = Field(gt=0)
    channels: int = Field(gt=0)
    duration_sec: float = Field(gt=0)

    @classmethod
    def from_audio_properties(cls, properties: AudioProperties) -> "ReportAudioProperties":
        return cls(
            sample_rate_hz=properties.sample_rate_hz,
            channels=properties.channels,
            duration_sec=properties.duration_sec,
        )


class ProcessingMessage(BaseModel):
    model_config = ConfigDict(extra="forbid")

    code: str = Field(min_length=1)
    message: str = Field(min_length=1)


class AudioPreparationReport(BaseModel):
    model_config = ConfigDict(extra="forbid")

    sample_id: str = Field(min_length=1)
    status: AudioPreparationStatus
    source_audio_path: str = Field(min_length=1)
    original: ReportAudioProperties | None = None
    prepared: ReportAudioProperties | None = None
    vad_backend: str
    vad_config: VADConfig
    vad_version: str
    segment_count: int = Field(ge=0)
    speech_duration_sec: float = Field(ge=0)
    speech_ratio: float = Field(ge=0)
    child_sample_ids: list[str] = Field(default_factory=list)
    warnings: list[ProcessingMessage] = Field(default_factory=list)
    errors: list[ProcessingMessage] = Field(default_factory=list)


@dataclass(frozen=True)
class BatchSummary:
    input_samples: int
    prepared: int
    dropped: int
    segments_created: int
    errors: int


@dataclass(frozen=True)
class BatchOutputs:
    source_manifest: Path
    segment_manifest: Path
    report_file: Path
    summary: BatchSummary


_SAFE_ID = re.compile(r"[^A-Za-z0-9._-]+")


def _segment_id(parent_sample_id: str, one_based_index: int) -> str:
    safe_parent = _SAFE_ID.sub("_", parent_sample_id).strip("._-") or "sample"
    if safe_parent != parent_sample_id:
        digest = hashlib.sha256(parent_sample_id.encode("utf-8")).hexdigest()[:8]
        safe_parent = f"{safe_parent}_{digest}"
    return f"{safe_parent}__seg{one_based_index:04d}"


def _append_reason(sample: SpeechSample, reason: str) -> SpeechSample:
    updated = sample.model_copy(deep=True)
    updated.state.curation_decision = CurationDecision.DROP
    if reason not in updated.state.decision_reasons:
        updated.state.decision_reasons.append(reason)
    return updated


def _validate_regions(
    regions: list[SpeechRegion], waveform_length: int, sample_rate_hz: int
) -> None:
    previous_end = 0
    for region in regions:
        if region.sample_rate_hz != sample_rate_hz:
            raise ValueError("VAD returned a region with the wrong sample rate")
        if region.end_sample > waveform_length:
            raise ValueError("VAD returned a region beyond the prepared waveform")
        if region.start_sample < previous_end:
            raise ValueError("VAD returned overlapping or unsorted regions")
        previous_end = region.end_sample


def _manifest_audio_path(path: Path, dataset_root: Path) -> str:
    return path.resolve().relative_to(dataset_root.resolve()).as_posix()


def write_audio_preparation_reports(
    path: str | PathLike[str], reports: list[AudioPreparationReport]
) -> int:
    destination = Path(path)
    with destination.open("w", encoding="utf-8", newline="\n") as handle:
        for report in reports:
            handle.write(report.model_dump_json())
            handle.write("\n")
    return len(reports)


def iter_audio_preparation_reports(
    path: str | PathLike[str],
) -> Iterator[AudioPreparationReport]:
    report_path = Path(path)
    with report_path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            try:
                yield AudioPreparationReport.model_validate_json(line)
            except ValueError as exc:
                raise ValueError(f"{report_path}:{line_number}: invalid audio preparation report: {exc}") from exc


class AudioPreparationPipeline:
    def __init__(
        self,
        *,
        dataset_root: Path,
        output_root: Path,
        vad: VoiceActivityDetector,
        config: AudioPreparationConfig | None = None,
    ) -> None:
        self.dataset_root = dataset_root.resolve()
        self.output_root = output_root.resolve()
        if not self.output_root.is_relative_to(self.dataset_root):
            raise ValueError("output_root must be inside dataset_root so manifest paths stay portable")
        self.vad = vad
        self.config = config or AudioPreparationConfig()

        self.segment_dir = self.output_root / "interim" / "audio_segments"
        self.manifest_dir = self.output_root / "manifests"
        self.report_dir = self.output_root / "reports"

    def run(self, input_manifest: Path) -> BatchOutputs:
        parents = list(iter_manifest(input_manifest))
        self.segment_dir.mkdir(parents=True, exist_ok=True)
        self.manifest_dir.mkdir(parents=True, exist_ok=True)
        self.report_dir.mkdir(parents=True, exist_ok=True)

        updated_parents: list[SpeechSample] = []
        children: list[SpeechSample] = []
        reports: list[AudioPreparationReport] = []

        for parent in parents:
            updated_parent, sample_children, report = self._process_sample(parent)
            updated_parents.append(updated_parent)
            children.extend(sample_children)
            reports.append(report)

        source_manifest = self.manifest_dir / "audio_prepared_sources.jsonl"
        segment_manifest = self.manifest_dir / "audio_segments.jsonl"
        report_file = self.report_dir / "audio_preparation_report.jsonl"
        write_manifest(source_manifest, updated_parents)
        write_manifest(segment_manifest, children)
        write_audio_preparation_reports(report_file, reports)

        summary = BatchSummary(
            input_samples=len(parents),
            prepared=sum(report.status is AudioPreparationStatus.PREPARED for report in reports),
            dropped=sum(report.status is AudioPreparationStatus.DROPPED for report in reports),
            segments_created=len(children),
            errors=sum(len(report.errors) for report in reports),
        )
        return BatchOutputs(source_manifest, segment_manifest, report_file, summary)

    def _process_sample(
        self, parent: SpeechSample
    ) -> tuple[SpeechSample, list[SpeechSample], AudioPreparationReport]:
        source_path = Path(parent.audio.path)
        resolved_source = source_path if source_path.is_absolute() else self.dataset_root / source_path

        try:
            decoded = decode_audio(resolved_source)
            prepared = prepare_audio(decoded, self.config.target_sample_rate_hz)
        except AudioValidationError as exc:
            dropped = _append_reason(parent, exc.code)
            error_messages = {
                "audio_missing": "audio file does not exist",
                "audio_unreadable": "audio path is unreadable",
                "audio_decode_failed": "SoundFile could not decode audio",
                "audio_empty": "audio contains no usable frames",
                "audio_invalid_sample_rate": "audio sample rate is invalid",
                "audio_invalid_channels": "audio channel count is invalid",
                "audio_non_finite": "audio contains NaN or infinity",
            }
            report = AudioPreparationReport(
                sample_id=parent.sample_id,
                status=AudioPreparationStatus.DROPPED,
                source_audio_path=parent.audio.path,
                vad_backend=self.vad.backend_name,
                vad_config=self.config.vad,
                vad_version=self.vad.backend_version,
                segment_count=0,
                speech_duration_sec=0,
                speech_ratio=0,
                errors=[
                    ProcessingMessage(
                        code=exc.code,
                        message=f"{error_messages.get(exc.code, 'audio validation failed')}: {parent.audio.path}",
                    )
                ],
            )
            return dropped, [], report

        regions = self.vad.detect(prepared.waveform, prepared.properties.sample_rate_hz)
        _validate_regions(
            regions, prepared.waveform.numel(), prepared.properties.sample_rate_hz
        )
        audio_prepared_parent = transition_stage(parent, ProcessingStage.AUDIO_PREPARED)

        if not regions:
            dropped = _append_reason(audio_prepared_parent, "no_speech_detected")
            report = AudioPreparationReport(
                sample_id=parent.sample_id,
                status=AudioPreparationStatus.DROPPED,
                source_audio_path=parent.audio.path,
                original=ReportAudioProperties.from_audio_properties(decoded.properties),
                prepared=ReportAudioProperties.from_audio_properties(prepared.properties),
                vad_backend=self.vad.backend_name,
                vad_config=self.config.vad,
                vad_version=self.vad.backend_version,
                segment_count=0,
                speech_duration_sec=0,
                speech_ratio=0,
                warnings=[
                    ProcessingMessage(
                        code="no_speech_detected",
                        message="audio preparation completed but VAD found no speech",
                    )
                ],
            )
            return dropped, [], report

        children: list[SpeechSample] = []
        for index, region in enumerate(regions, start=1):
            child_id = _segment_id(parent.sample_id, index)
            segment_path = self.segment_dir / f"{child_id}.wav"
            segment_waveform = prepared.waveform[region.start_sample : region.end_sample]
            write_pcm16_wav(segment_path, segment_waveform, prepared.properties.sample_rate_hz)
            children.append(
                self._make_child(
                    parent=parent,
                    child_id=child_id,
                    audio_path=_manifest_audio_path(segment_path, self.dataset_root),
                    region=region,
                )
            )

        speech_duration_sec = sum(region.duration_sec for region in regions)
        report = AudioPreparationReport(
            sample_id=parent.sample_id,
            status=AudioPreparationStatus.PREPARED,
            source_audio_path=parent.audio.path,
            original=ReportAudioProperties.from_audio_properties(decoded.properties),
            prepared=ReportAudioProperties.from_audio_properties(prepared.properties),
            vad_backend=self.vad.backend_name,
            vad_config=self.config.vad,
            vad_version=self.vad.backend_version,
            segment_count=len(children),
            speech_duration_sec=speech_duration_sec,
            speech_ratio=speech_duration_sec / prepared.properties.duration_sec,
            child_sample_ids=[child.sample_id for child in children],
        )
        return audio_prepared_parent, children, report

    def _make_child(
        self,
        *,
        parent: SpeechSample,
        child_id: str,
        audio_path: str,
        region: SpeechRegion,
    ) -> SpeechSample:
        return SpeechSample(
            sample_id=child_id,
            dataset_id=parent.dataset_id,
            language=parent.language,
            locale=parent.locale,
            audio=AudioInfo(
                path=audio_path,
                format="wav",
                duration_sec=region.duration_sec,
                sample_rate_hz=self.config.target_sample_rate_hz,
                channels=self.config.target_channels,
            ),
            text=TextInfo(),
            speaker=parent.speaker.model_copy(deep=True),
            source=parent.source.model_copy(deep=True),
            lineage=LineageInfo(
                parent_sample_id=parent.sample_id,
                segment_start_sec=region.start_sec,
                segment_end_sec=region.end_sec,
            ),
            split=parent.split,
            state=ProcessingState(
                stage=ProcessingStage.AUDIO_PREPARED,
                curation_decision=CurationDecision.UNDECIDED,
            ),
            metadata={},
        )
