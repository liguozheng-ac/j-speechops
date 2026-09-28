"""Model-neutral Stage 8 TTS planning, runtime orchestration, and manifests."""

from __future__ import annotations

import hashlib
import json
import math
import re
import subprocess
import tempfile
from collections.abc import Iterable, Iterator
from dataclasses import dataclass
from enum import Enum
from os import PathLike
from pathlib import Path
from typing import Annotated, Literal, Protocol

import soundfile as sf
from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    StringConstraints,
    model_validator,
)

from .japanese_reading import AppliedReadingOverride
from .schemas import CurationDecision
from .text_manifest import iter_text_manifest
from .text_preparation import (
    TextPreparationReport,
    TextPreparationStatus,
    iter_text_preparation_reports,
)
from .tts_text import TTSTextSample, TTSTextStage
from .tts_text_curation import (
    TTSTextCurationReport,
    TTSTextCurationStatus,
    iter_tts_text_curation_reports,
)


SYNTHESIS_SCHEMA_VERSION = "1.0"
SYNTHESIS_TEXT_STRATEGY = "normalized-with-confirmed-overrides-v1"
SEED_POLICY = "sha256-sample-id-plus-base-mod-2147483647-v1"
AUDIO_POST_PROCESSING_POLICY_VERSION = "tail-padding-v1"
AUDIO_POST_PROCESSING_TYPE = "tail_padding"
DEFAULT_POST_ROLL_MS = 300
NonEmptyStr = Annotated[str, StringConstraints(strip_whitespace=True, min_length=1)]


class SynthesisModel(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)


class GenerationParameters(SynthesisModel):
    """Pinned effective Qwen3-TTS generation configuration."""

    do_sample: bool = True
    top_k: int = Field(default=50, ge=0)
    top_p: float = Field(default=1.0, gt=0.0, le=1.0)
    temperature: float = Field(default=0.9, gt=0.0)
    repetition_penalty: float = Field(default=1.05, gt=0.0)
    subtalker_dosample: bool = True
    subtalker_top_k: int = Field(default=50, ge=0)
    subtalker_top_p: float = Field(default=1.0, gt=0.0, le=1.0)
    subtalker_temperature: float = Field(default=0.9, gt=0.0)
    max_new_tokens: int = Field(default=8192, gt=0)


class AudioPostProcessingPolicy(SynthesisModel):
    """Deterministic processing applied after inference and before WAV export."""

    type: Literal[AUDIO_POST_PROCESSING_TYPE] = AUDIO_POST_PROCESSING_TYPE
    duration_ms: int = Field(default=DEFAULT_POST_ROLL_MS, ge=0)
    policy_version: Literal[AUDIO_POST_PROCESSING_POLICY_VERSION] = (
        AUDIO_POST_PROCESSING_POLICY_VERSION
    )


class AudioPostProcessingProvenance(AudioPostProcessingPolicy):
    policy_fingerprint_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")


class SynthesisConfig(SynthesisModel):
    backend: NonEmptyStr = "qwen3-tts"
    model_name: NonEmptyStr = "Qwen/Qwen3-TTS-12Hz-1.7B-CustomVoice"
    model_path: NonEmptyStr
    speaker: NonEmptyStr = "Ono_Anna"
    language: NonEmptyStr = "Japanese"
    device: NonEmptyStr = "cuda:0"
    dtype: NonEmptyStr = "bfloat16"
    attention_implementation: NonEmptyStr = "sdpa"
    instruct: None = None
    non_streaming_mode: Literal[True] = True
    generation_parameters: GenerationParameters = Field(
        default_factory=GenerationParameters
    )
    audio_post_processing: AudioPostProcessingPolicy = Field(
        default_factory=AudioPostProcessingPolicy
    )
    base_seed: int = Field(default=20260921, ge=0, lt=2_147_483_647)
    seed_policy: Literal[SEED_POLICY] = SEED_POLICY
    synthesis_text_strategy: Literal[SYNTHESIS_TEXT_STRATEGY] = (
        SYNTHESIS_TEXT_STRATEGY
    )


class SynthesisRunStatus(str, Enum):
    SUCCESS = "success"
    FAILED = "failed"


class RuntimeResultStatus(str, Enum):
    SUCCESS = "success"
    FAILED = "failed"


class SynthesisError(SynthesisModel):
    phase: NonEmptyStr
    code: NonEmptyStr
    message: NonEmptyStr


class SynthesisTextPlan(SynthesisModel):
    normalized_text: NonEmptyStr
    synthesis_text: NonEmptyStr
    expected_reading_kana: NonEmptyStr
    strategy: Literal[SYNTHESIS_TEXT_STRATEGY] = SYNTHESIS_TEXT_STRATEGY
    applied_pronunciation_overrides: tuple[AppliedReadingOverride, ...] = ()


class RuntimeSynthesisRequest(SynthesisModel):
    run_id: str = Field(pattern=r"^[0-9a-f]{64}$")
    sample_id: NonEmptyStr
    synthesis_text: NonEmptyStr
    seed: int = Field(ge=0, lt=2_147_483_647)
    output_audio_path: NonEmptyStr


class RuntimeBatchJob(SynthesisModel):
    schema_version: Literal[SYNTHESIS_SCHEMA_VERSION] = SYNTHESIS_SCHEMA_VERSION
    backend: NonEmptyStr
    model_name: NonEmptyStr
    model_path: NonEmptyStr
    speaker: NonEmptyStr
    language: NonEmptyStr
    device: NonEmptyStr
    dtype: NonEmptyStr
    attention_implementation: NonEmptyStr
    instruct: None = None
    non_streaming_mode: Literal[True] = True
    generation_parameters: GenerationParameters
    audio_post_processing: AudioPostProcessingPolicy = Field(
        default_factory=AudioPostProcessingPolicy
    )
    requests: tuple[RuntimeSynthesisRequest, ...]


class RuntimeSampleResult(SynthesisModel):
    run_id: str = Field(pattern=r"^[0-9a-f]{64}$")
    status: RuntimeResultStatus
    generation_time_sec: float | None = Field(default=None, ge=0.0)
    error_code: str | None = None
    error_message: str | None = None

    @model_validator(mode="after")
    def status_fields_agree(self) -> "RuntimeSampleResult":
        if self.status is RuntimeResultStatus.SUCCESS:
            if self.generation_time_sec is None:
                raise ValueError("successful runtime result requires generation time")
            if self.error_code is not None or self.error_message is not None:
                raise ValueError("successful runtime result must not contain an error")
        elif not self.error_code or not self.error_message:
            raise ValueError("failed runtime result requires an error")
        return self


class RuntimeBatchResult(SynthesisModel):
    schema_version: Literal[SYNTHESIS_SCHEMA_VERSION] = SYNTHESIS_SCHEMA_VERSION
    backend_version: NonEmptyStr
    model_load_time_sec: float = Field(ge=0.0)
    gpu_name: NonEmptyStr
    gpu_total_vram_bytes: int = Field(gt=0)
    peak_allocated_vram_bytes: int = Field(ge=0)
    peak_reserved_vram_bytes: int = Field(ge=0)
    supported_languages: tuple[str, ...]
    supported_speakers: tuple[str, ...]
    results: tuple[RuntimeSampleResult, ...]


class SynthesisRun(SynthesisModel):
    schema_version: Literal[SYNTHESIS_SCHEMA_VERSION] = SYNTHESIS_SCHEMA_VERSION
    run_id: str = Field(pattern=r"^[0-9a-f]{64}$")
    sample_id: NonEmptyStr
    backend: NonEmptyStr
    backend_version: str | None = None
    model_name: NonEmptyStr
    model_path: NonEmptyStr
    speaker: NonEmptyStr
    language: NonEmptyStr
    device: NonEmptyStr
    dtype: NonEmptyStr
    attention_implementation: NonEmptyStr
    generation_parameters: GenerationParameters
    audio_post_processing: AudioPostProcessingProvenance | None = None
    config_fingerprint_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    seed: int = Field(ge=0, lt=2_147_483_647)
    seed_policy: Literal[SEED_POLICY] = SEED_POLICY
    synthesis_text_strategy: Literal[SYNTHESIS_TEXT_STRATEGY] = (
        SYNTHESIS_TEXT_STRATEGY
    )
    normalized_text: str | None = None
    synthesis_text: str | None = None
    expected_reading_kana: str | None = None
    applied_pronunciation_overrides: tuple[AppliedReadingOverride, ...] = ()
    status: SynthesisRunStatus
    output_audio_path: str | None = None
    sample_rate_hz: int | None = Field(default=None, gt=0)
    channels: int | None = Field(default=None, gt=0)
    duration_sec: float | None = Field(default=None, gt=0.0)
    generation_time_sec: float | None = Field(default=None, ge=0.0)
    rtf: float | None = Field(default=None, ge=0.0)
    audio_sha256: str | None = Field(default=None, pattern=r"^[0-9a-f]{64}$")
    errors: tuple[SynthesisError, ...] = ()

    @model_validator(mode="after")
    def status_fields_agree(self) -> "SynthesisRun":
        success_values = (
            self.backend_version,
            self.normalized_text,
            self.synthesis_text,
            self.expected_reading_kana,
            self.output_audio_path,
            self.sample_rate_hz,
            self.channels,
            self.duration_sec,
            self.generation_time_sec,
            self.rtf,
            self.audio_sha256,
        )
        if self.status is SynthesisRunStatus.SUCCESS:
            if any(value is None for value in success_values):
                raise ValueError("successful synthesis run lacks required provenance")
            if self.errors:
                raise ValueError("successful synthesis run must not contain errors")
        elif not self.errors:
            raise ValueError("failed synthesis run requires at least one error")
        return self


class GeneratedAudioArtifact(SynthesisModel):
    schema_version: Literal[SYNTHESIS_SCHEMA_VERSION] = SYNTHESIS_SCHEMA_VERSION
    run_id: str = Field(pattern=r"^[0-9a-f]{64}$")
    sample_id: NonEmptyStr
    input_text: NonEmptyStr
    expected_reading_kana: NonEmptyStr
    synthesis_text: NonEmptyStr
    synthesis_text_strategy: Literal[SYNTHESIS_TEXT_STRATEGY] = (
        SYNTHESIS_TEXT_STRATEGY
    )
    applied_pronunciation_overrides: tuple[AppliedReadingOverride, ...] = ()
    audio_path: NonEmptyStr
    backend: NonEmptyStr
    backend_version: NonEmptyStr
    model_name: NonEmptyStr
    speaker: NonEmptyStr
    language: NonEmptyStr
    audio_post_processing: AudioPostProcessingProvenance | None = None
    sample_rate_hz: int = Field(gt=0)
    channels: int = Field(gt=0)
    duration_sec: float = Field(gt=0.0)
    audio_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")


class SynthesisBatchSummary(SynthesisModel):
    schema_version: Literal[SYNTHESIS_SCHEMA_VERSION] = SYNTHESIS_SCHEMA_VERSION
    total_samples: int = Field(ge=0)
    planned_runs: int = Field(ge=0)
    attempted_runs: int = Field(ge=0)
    succeeded_runs: int = Field(ge=0)
    failed_runs: int = Field(ge=0)
    skipped_runs: int = Field(ge=0)
    planning_failed_runs: int = Field(ge=0)
    config_fingerprint_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    model_load_time_sec: float | None = Field(default=None, ge=0.0)
    gpu_name: str | None = None
    gpu_total_vram_bytes: int | None = Field(default=None, gt=0)
    peak_allocated_vram_bytes: int | None = Field(default=None, ge=0)
    peak_reserved_vram_bytes: int | None = Field(default=None, ge=0)


@dataclass(frozen=True)
class SynthesisBatchOutputs:
    synthesis_runs_manifest: Path
    generated_audio_manifest: Path
    summary_file: Path
    summary: SynthesisBatchSummary


@dataclass(frozen=True)
class WavValidation:
    sample_rate_hz: int
    channels: int
    duration_sec: float
    audio_sha256: str


class SynthesisPlanningError(ValueError):
    """One sample has inconsistent or incomplete Stage 6/7 provenance."""


class TTSRuntimeInfrastructureError(RuntimeError):
    """The isolated model runtime could not initialize or remain usable."""


class TTSRuntimeAdapter(Protocol):
    def synthesize(self, job: RuntimeBatchJob, *, output_root: Path) -> RuntimeBatchResult:
        """Run one model load and process every request in the job."""


class Qwen3TTSRuntimeAdapter:
    """Core-side subprocess adapter; this module never imports qwen_tts."""

    def __init__(self, *, python_executable: Path, worker_script: Path) -> None:
        self.python_executable = Path(python_executable).resolve()
        self.worker_script = Path(worker_script).resolve()

    def synthesize(self, job: RuntimeBatchJob, *, output_root: Path) -> RuntimeBatchResult:
        if not self.python_executable.is_file():
            raise TTSRuntimeInfrastructureError(
                f"TTS Python is unavailable: {self.python_executable}"
            )
        if not self.worker_script.is_file():
            raise TTSRuntimeInfrastructureError(
                f"Qwen runtime worker is unavailable: {self.worker_script}"
            )
        output_root.mkdir(parents=True, exist_ok=True)
        with tempfile.TemporaryDirectory(prefix=".runtime-", dir=output_root) as raw:
            temporary = Path(raw)
            job_path = temporary / "job.json"
            result_path = temporary / "result.json"
            job_path.write_text(job.model_dump_json(indent=2) + "\n", encoding="utf-8")
            completed = subprocess.run(
                [
                    str(self.python_executable),
                    str(self.worker_script),
                    "--job",
                    str(job_path),
                    "--result",
                    str(result_path),
                ],
                capture_output=True,
                text=True,
                encoding="utf-8",
                errors="replace",
                check=False,
            )
            if completed.returncode != 0:
                detail = completed.stderr.strip() or completed.stdout.strip()
                raise TTSRuntimeInfrastructureError(
                    "Qwen runtime worker failed "
                    f"with exit code {completed.returncode}: {detail}"
                )
            try:
                return RuntimeBatchResult.model_validate_json(
                    result_path.read_text(encoding="utf-8")
                )
            except (OSError, ValueError) as exc:
                raise TTSRuntimeInfrastructureError(
                    f"invalid Qwen runtime result: {exc}"
                ) from exc


def _canonical_sha256(payload: object) -> str:
    encoded = json.dumps(
        payload, ensure_ascii=False, sort_keys=True, separators=(",", ":")
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def config_fingerprint(config: SynthesisConfig) -> str:
    """Hash portable generation semantics, excluding the machine-local model path."""

    payload = config.model_dump(exclude={"model_path"}, mode="json")
    return _canonical_sha256(payload)


def audio_post_processing_policy_fingerprint(
    policy: AudioPostProcessingPolicy,
) -> str:
    return _canonical_sha256(policy.model_dump(mode="json"))


def audio_post_processing_provenance(
    policy: AudioPostProcessingPolicy,
) -> AudioPostProcessingProvenance:
    return AudioPostProcessingProvenance(
        **policy.model_dump(mode="python"),
        policy_fingerprint_sha256=audio_post_processing_policy_fingerprint(policy),
    )


def derive_seed(sample_id: str, base_seed: int) -> int:
    stable_component = int.from_bytes(
        hashlib.sha256(sample_id.encode("utf-8")).digest()[:8], "big"
    )
    return (base_seed + stable_component) % 2_147_483_647


def build_run_id(
    *,
    sample_id: str,
    config: SynthesisConfig,
    seed: int,
    synthesis_text: str | None,
) -> str:
    return _canonical_sha256(
        {
            "sample_id": sample_id,
            "backend": config.backend,
            "model_name": config.model_name,
            "speaker": config.speaker,
            "config_fingerprint_sha256": config_fingerprint(config),
            "seed": seed,
            "synthesis_text_strategy": config.synthesis_text_strategy,
            "synthesis_text": synthesis_text,
        }
    )


def build_synthesis_text(
    normalized_text: str,
    expected_reading_kana: str,
    overrides: Iterable[AppliedReadingOverride],
) -> SynthesisTextPlan:
    """Apply only explicitly confirmed Stage 6 override provenance."""

    normalized = normalized_text.strip()
    reading = expected_reading_kana.strip()
    if not normalized:
        raise SynthesisPlanningError("normalized_text is unavailable")
    if not reading:
        raise SynthesisPlanningError("reading_kana is unavailable")
    applied = tuple(overrides)
    by_surface: dict[str, AppliedReadingOverride] = {}
    for item in applied:
        if item.surface in by_surface:
            raise SynthesisPlanningError(
                f"duplicate pronunciation override surface: {item.surface}"
            )
        by_surface[item.surface] = item
    if not applied:
        return SynthesisTextPlan(
            normalized_text=normalized,
            synthesis_text=normalized,
            expected_reading_kana=reading,
        )

    surfaces = sorted(by_surface, key=lambda value: (-len(value), value))
    pattern = re.compile("|".join(re.escape(value) for value in surfaces))
    observed = {surface: 0 for surface in surfaces}

    def replace(match: re.Match[str]) -> str:
        surface = match.group(0)
        observed[surface] += 1
        return by_surface[surface].reading_kana

    synthesis_text = pattern.sub(replace, normalized)
    for surface, item in by_surface.items():
        if observed[surface] != item.occurrences:
            raise SynthesisPlanningError(
                "pronunciation override occurrence mismatch for "
                f"{surface!r}: provenance={item.occurrences}, "
                f"normalized_text={observed[surface]}"
            )
    return SynthesisTextPlan(
        normalized_text=normalized,
        synthesis_text=synthesis_text,
        expected_reading_kana=reading,
        applied_pronunciation_overrides=applied,
    )


def validate_generated_wav(path: str | PathLike[str]) -> WavValidation:
    audio_path = Path(path)
    if not audio_path.is_file() or audio_path.stat().st_size <= 44:
        raise ValueError(f"generated WAV is missing or empty: {audio_path}")
    try:
        audio, sample_rate = sf.read(audio_path, always_2d=True, dtype="float32")
    except (OSError, RuntimeError, ValueError) as exc:
        raise ValueError(f"generated WAV is not decodable: {audio_path}: {exc}") from exc
    frames, channels = audio.shape
    if sample_rate <= 0 or channels <= 0 or frames <= 0:
        raise ValueError(f"generated WAV has invalid stream metadata: {audio_path}")
    if not bool((audio == audio).all()) or not bool(abs(audio).max() < float("inf")):
        raise ValueError(f"generated WAV contains non-finite samples: {audio_path}")
    if float(abs(audio).max()) <= 1e-6:
        raise ValueError(f"generated WAV is silent: {audio_path}")
    duration = frames / sample_rate
    if not math.isfinite(duration) or duration <= 0.0:
        raise ValueError(f"generated WAV has invalid duration: {audio_path}")
    return WavValidation(
        sample_rate_hz=sample_rate,
        channels=channels,
        duration_sec=duration,
        audio_sha256=hashlib.sha256(audio_path.read_bytes()).hexdigest(),
    )


def _read_unique_reports(
    reports: Iterable[TextPreparationReport | TTSTextCurationReport],
    *,
    kind: str,
) -> dict[str, TextPreparationReport | TTSTextCurationReport]:
    indexed: dict[str, TextPreparationReport | TTSTextCurationReport] = {}
    for report in reports:
        if report.sample_id in indexed:
            raise ValueError(f"duplicate {kind} report sample_id: {report.sample_id}")
        indexed[report.sample_id] = report
    return indexed


def write_synthesis_runs(path: str | PathLike[str], runs: Iterable[SynthesisRun]) -> int:
    values = sorted((SynthesisRun.model_validate(item) for item in runs), key=lambda x: x.run_id)
    if len({item.run_id for item in values}) != len(values):
        raise ValueError("duplicate synthesis run_id")
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    with destination.open("w", encoding="utf-8", newline="\n") as handle:
        for item in values:
            handle.write(item.model_dump_json())
            handle.write("\n")
    return len(values)


def iter_synthesis_runs(path: str | PathLike[str]) -> Iterator[SynthesisRun]:
    source = Path(path)
    seen: set[str] = set()
    with source.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            try:
                run = SynthesisRun.model_validate_json(line)
            except ValueError as exc:
                raise ValueError(
                    f"{source}:{line_number}: invalid SynthesisRun: {exc}"
                ) from exc
            if run.run_id in seen:
                raise ValueError(f"{source}:{line_number}: duplicate run_id {run.run_id}")
            seen.add(run.run_id)
            yield run


def write_generated_audio_manifest(
    path: str | PathLike[str], artifacts: Iterable[GeneratedAudioArtifact]
) -> int:
    values = sorted(
        (GeneratedAudioArtifact.model_validate(item) for item in artifacts),
        key=lambda x: x.run_id,
    )
    if len({item.run_id for item in values}) != len(values):
        raise ValueError("duplicate generated audio run_id")
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    with destination.open("w", encoding="utf-8", newline="\n") as handle:
        for item in values:
            handle.write(item.model_dump_json())
            handle.write("\n")
    return len(values)


class TTSSynthesisPipeline:
    def __init__(
        self,
        *,
        output_root: Path,
        config: SynthesisConfig,
        runtime_adapter: TTSRuntimeAdapter,
    ) -> None:
        self.output_root = Path(output_root).resolve()
        self.audio_dir = self.output_root / "audio"
        self.config = config
        self.runtime_adapter = runtime_adapter
        self.fingerprint = config_fingerprint(config)
        self.post_processing = audio_post_processing_provenance(
            config.audio_post_processing
        )

    def run(
        self,
        manifest: Path,
        preparation_report: Path,
        curation_report: Path,
        *,
        force: bool = False,
    ) -> SynthesisBatchOutputs:
        samples = list(iter_text_manifest(manifest))
        invalid = [
            sample.sample_id
            for sample in samples
            if sample.state.stage is not TTSTextStage.CURATED
            or sample.state.curation_decision is not CurationDecision.PASS
        ]
        if invalid:
            raise ValueError(
                "Stage 8 accepts only CURATED + PASS samples: " + ", ".join(invalid)
            )

        preparations = _read_unique_reports(
            iter_text_preparation_reports(preparation_report), kind="preparation"
        )
        curations = _read_unique_reports(
            iter_tts_text_curation_reports(curation_report), kind="curation"
        )
        preparations = {key: value for key, value in preparations.items()}
        curations = {key: value for key, value in curations.items()}

        runs_path = self.output_root / "synthesis_runs.jsonl"
        artifacts_path = self.output_root / "generated_audio_manifest.jsonl"
        summary_path = self.output_root / "tts_generation_summary.json"
        previous_summary: SynthesisBatchSummary | None = None
        if summary_path.is_file():
            try:
                candidate = SynthesisBatchSummary.model_validate_json(
                    summary_path.read_text(encoding="utf-8")
                )
            except (OSError, ValueError):
                candidate = None
            if (
                candidate is not None
                and candidate.config_fingerprint_sha256 == self.fingerprint
            ):
                previous_summary = candidate
        existing = (
            {run.run_id: run for run in iter_synthesis_runs(runs_path)}
            if runs_path.is_file()
            else {}
        )
        merged = dict(existing)
        scheduled: list[tuple[TTSTextSample, SynthesisTextPlan, int, str, Path]] = []
        current_run_ids: list[str] = []
        planning_failed = 0
        skipped = 0

        for sample in samples:
            seed = derive_seed(sample.sample_id, self.config.base_seed)
            normalized = sample.text.normalized_text
            reading = sample.text.reading_kana
            prep = preparations.get(sample.sample_id)
            curation = curations.get(sample.sample_id)
            try:
                plan = self._plan_sample(sample, prep, curation)
            except SynthesisPlanningError as exc:
                failed_overrides = (
                    prep.reading.applied_overrides
                    if isinstance(prep, TextPreparationReport)
                    and prep.reading is not None
                    else ()
                )
                run_id = build_run_id(
                    sample_id=sample.sample_id,
                    config=self.config,
                    seed=seed,
                    synthesis_text=None,
                )
                current_run_ids.append(run_id)
                merged[run_id] = self._failed_run(
                    run_id=run_id,
                    sample=sample,
                    seed=seed,
                    normalized_text=normalized,
                    expected_reading_kana=reading,
                    synthesis_text=None,
                    overrides=failed_overrides,
                    phase="planning",
                    code="synthesis_planning_failed",
                    message=str(exc),
                )
                planning_failed += 1
                continue

            run_id = build_run_id(
                sample_id=sample.sample_id,
                config=self.config,
                seed=seed,
                synthesis_text=plan.synthesis_text,
            )
            current_run_ids.append(run_id)
            output_path = (self.audio_dir / f"{run_id}.wav").resolve()
            previous = existing.get(run_id)
            if not force and previous is not None and self._is_valid_existing(previous):
                skipped += 1
                continue
            scheduled.append((sample, plan, seed, run_id, output_path))

        runtime_result: RuntimeBatchResult | None = None
        if scheduled:
            self.audio_dir.mkdir(parents=True, exist_ok=True)
            job = RuntimeBatchJob(
                backend=self.config.backend,
                model_name=self.config.model_name,
                model_path=self.config.model_path,
                speaker=self.config.speaker,
                language=self.config.language,
                device=self.config.device,
                dtype=self.config.dtype,
                attention_implementation=self.config.attention_implementation,
                instruct=self.config.instruct,
                non_streaming_mode=self.config.non_streaming_mode,
                generation_parameters=self.config.generation_parameters,
                audio_post_processing=self.config.audio_post_processing,
                requests=tuple(
                    RuntimeSynthesisRequest(
                        run_id=run_id,
                        sample_id=sample.sample_id,
                        synthesis_text=plan.synthesis_text,
                        seed=seed,
                        output_audio_path=str(output_path),
                    )
                    for sample, plan, seed, run_id, output_path in scheduled
                ),
            )
            runtime_result = self.runtime_adapter.synthesize(
                job, output_root=self.output_root
            )
            result_by_id = {item.run_id: item for item in runtime_result.results}
            if len(result_by_id) != len(runtime_result.results):
                raise TTSRuntimeInfrastructureError("worker returned duplicate run IDs")
            expected_ids = {item[3] for item in scheduled}
            if set(result_by_id) != expected_ids:
                raise TTSRuntimeInfrastructureError(
                    "worker result IDs do not match submitted request IDs"
                )
            for sample, plan, seed, run_id, output_path in scheduled:
                result = result_by_id[run_id]
                if result.status is RuntimeResultStatus.FAILED:
                    output_path.unlink(missing_ok=True)
                    merged[run_id] = self._failed_run(
                        run_id=run_id,
                        sample=sample,
                        seed=seed,
                        normalized_text=plan.normalized_text,
                        expected_reading_kana=plan.expected_reading_kana,
                        synthesis_text=plan.synthesis_text,
                        overrides=plan.applied_pronunciation_overrides,
                        phase="generation",
                        code=result.error_code or "generation_failed",
                        message=result.error_message or "runtime generation failed",
                        backend_version=runtime_result.backend_version,
                    )
                    continue
                try:
                    validation = validate_generated_wav(output_path)
                except ValueError as exc:
                    output_path.unlink(missing_ok=True)
                    merged[run_id] = self._failed_run(
                        run_id=run_id,
                        sample=sample,
                        seed=seed,
                        normalized_text=plan.normalized_text,
                        expected_reading_kana=plan.expected_reading_kana,
                        synthesis_text=plan.synthesis_text,
                        overrides=plan.applied_pronunciation_overrides,
                        phase="validation",
                        code="wav_validation_failed",
                        message=str(exc),
                        backend_version=runtime_result.backend_version,
                    )
                    continue
                generation_time = result.generation_time_sec or 0.0
                merged[run_id] = SynthesisRun(
                    run_id=run_id,
                    sample_id=sample.sample_id,
                    backend=self.config.backend,
                    backend_version=runtime_result.backend_version,
                    model_name=self.config.model_name,
                    model_path=self.config.model_path,
                    speaker=self.config.speaker,
                    language=self.config.language,
                    device=self.config.device,
                    dtype=self.config.dtype,
                    attention_implementation=self.config.attention_implementation,
                    generation_parameters=self.config.generation_parameters,
                    audio_post_processing=self.post_processing,
                    config_fingerprint_sha256=self.fingerprint,
                    seed=seed,
                    normalized_text=plan.normalized_text,
                    synthesis_text=plan.synthesis_text,
                    expected_reading_kana=plan.expected_reading_kana,
                    applied_pronunciation_overrides=(
                        plan.applied_pronunciation_overrides
                    ),
                    status=SynthesisRunStatus.SUCCESS,
                    output_audio_path=str(output_path),
                    sample_rate_hz=validation.sample_rate_hz,
                    channels=validation.channels,
                    duration_sec=validation.duration_sec,
                    generation_time_sec=generation_time,
                    rtf=generation_time / validation.duration_sec,
                    audio_sha256=validation.audio_sha256,
                )

        write_synthesis_runs(runs_path, merged.values())
        artifacts = [
            self._artifact_from_run(run)
            for run in merged.values()
            if run.status is SynthesisRunStatus.SUCCESS and self._is_valid_existing(run)
        ]
        write_generated_audio_manifest(artifacts_path, artifacts)
        current = [merged[run_id] for run_id in current_run_ids]
        summary = SynthesisBatchSummary(
            total_samples=len(samples),
            planned_runs=len(current_run_ids),
            attempted_runs=len(scheduled),
            succeeded_runs=sum(
                run.status is SynthesisRunStatus.SUCCESS for run in current
            ),
            failed_runs=sum(run.status is SynthesisRunStatus.FAILED for run in current),
            skipped_runs=skipped,
            planning_failed_runs=planning_failed,
            config_fingerprint_sha256=self.fingerprint,
            model_load_time_sec=(
                runtime_result.model_load_time_sec
                if runtime_result
                else (previous_summary.model_load_time_sec if previous_summary else None)
            ),
            gpu_name=(
                runtime_result.gpu_name
                if runtime_result
                else (previous_summary.gpu_name if previous_summary else None)
            ),
            gpu_total_vram_bytes=(
                runtime_result.gpu_total_vram_bytes
                if runtime_result
                else (
                    previous_summary.gpu_total_vram_bytes
                    if previous_summary
                    else None
                )
            ),
            peak_allocated_vram_bytes=(
                runtime_result.peak_allocated_vram_bytes
                if runtime_result
                else (
                    previous_summary.peak_allocated_vram_bytes
                    if previous_summary
                    else None
                )
            ),
            peak_reserved_vram_bytes=(
                runtime_result.peak_reserved_vram_bytes
                if runtime_result
                else (
                    previous_summary.peak_reserved_vram_bytes
                    if previous_summary
                    else None
                )
            ),
        )
        summary_path.parent.mkdir(parents=True, exist_ok=True)
        summary_path.write_text(
            summary.model_dump_json(indent=2) + "\n", encoding="utf-8"
        )
        return SynthesisBatchOutputs(
            synthesis_runs_manifest=runs_path,
            generated_audio_manifest=artifacts_path,
            summary_file=summary_path,
            summary=summary,
        )

    def _plan_sample(
        self,
        sample: TTSTextSample,
        preparation: TextPreparationReport | TTSTextCurationReport | None,
        curation: TextPreparationReport | TTSTextCurationReport | None,
    ) -> SynthesisTextPlan:
        if not isinstance(preparation, TextPreparationReport):
            raise SynthesisPlanningError("matching Stage 6 preparation report is missing")
        if not isinstance(curation, TTSTextCurationReport):
            raise SynthesisPlanningError("matching Stage 7 curation report is missing")
        if preparation.status not in {
            TextPreparationStatus.READING_PREPARED,
            TextPreparationStatus.SKIPPED,
        } or preparation.reading is None:
            raise SynthesisPlanningError("Stage 6 reading provenance is unavailable")
        if (
            curation.status not in {
                TTSTextCurationStatus.CURATED,
                TTSTextCurationStatus.SKIPPED,
            }
            or curation.decision is not CurationDecision.PASS
        ):
            raise SynthesisPlanningError("Stage 7 report does not confirm PASS")
        normalized = sample.text.normalized_text
        reading = sample.text.reading_kana
        if normalized is None or preparation.reading.input_text != normalized:
            raise SynthesisPlanningError(
                "normalized_text contradicts Stage 6 reading provenance"
            )
        if reading is None or preparation.reading.reading_kana != reading:
            raise SynthesisPlanningError(
                "reading_kana contradicts Stage 6 reading provenance"
            )
        if curation.normalized_text != normalized or curation.reading_kana != reading:
            raise SynthesisPlanningError(
                "Stage 7 report contradicts canonical prepared text"
            )
        if curation.applied_overrides != preparation.reading.applied_overrides:
            raise SynthesisPlanningError(
                "Stage 6 and Stage 7 pronunciation override provenance disagree"
            )
        return build_synthesis_text(
            normalized, reading, preparation.reading.applied_overrides
        )

    def _base_run_values(
        self,
        *,
        run_id: str,
        sample: TTSTextSample,
        seed: int,
        normalized_text: str | None,
        expected_reading_kana: str | None,
        synthesis_text: str | None,
        overrides: tuple[AppliedReadingOverride, ...],
        backend_version: str | None,
    ) -> dict[str, object]:
        return {
            "run_id": run_id,
            "sample_id": sample.sample_id,
            "backend": self.config.backend,
            "backend_version": backend_version,
            "model_name": self.config.model_name,
            "model_path": self.config.model_path,
            "speaker": self.config.speaker,
            "language": self.config.language,
            "device": self.config.device,
            "dtype": self.config.dtype,
            "attention_implementation": self.config.attention_implementation,
            "generation_parameters": self.config.generation_parameters,
            "audio_post_processing": self.post_processing,
            "config_fingerprint_sha256": self.fingerprint,
            "seed": seed,
            "normalized_text": normalized_text,
            "synthesis_text": synthesis_text,
            "expected_reading_kana": expected_reading_kana,
            "applied_pronunciation_overrides": overrides,
        }

    def _failed_run(
        self,
        *,
        run_id: str,
        sample: TTSTextSample,
        seed: int,
        normalized_text: str | None,
        expected_reading_kana: str | None,
        synthesis_text: str | None,
        overrides: tuple[AppliedReadingOverride, ...],
        phase: str,
        code: str,
        message: str,
        backend_version: str | None = None,
    ) -> SynthesisRun:
        return SynthesisRun(
            **self._base_run_values(
                run_id=run_id,
                sample=sample,
                seed=seed,
                normalized_text=normalized_text,
                expected_reading_kana=expected_reading_kana,
                synthesis_text=synthesis_text,
                overrides=overrides,
                backend_version=backend_version,
            ),
            status=SynthesisRunStatus.FAILED,
            errors=(SynthesisError(phase=phase, code=code, message=message),),
        )

    @staticmethod
    def _is_valid_existing(run: SynthesisRun) -> bool:
        if (
            run.status is not SynthesisRunStatus.SUCCESS
            or run.output_audio_path is None
            or run.audio_sha256 is None
        ):
            return False
        try:
            validation = validate_generated_wav(run.output_audio_path)
        except ValueError:
            return False
        return validation.audio_sha256 == run.audio_sha256

    @staticmethod
    def _artifact_from_run(run: SynthesisRun) -> GeneratedAudioArtifact:
        if run.status is not SynthesisRunStatus.SUCCESS:
            raise ValueError("only successful runs can become generated artifacts")
        return GeneratedAudioArtifact(
            run_id=run.run_id,
            sample_id=run.sample_id,
            input_text=run.normalized_text or "",
            expected_reading_kana=run.expected_reading_kana or "",
            synthesis_text=run.synthesis_text or "",
            applied_pronunciation_overrides=run.applied_pronunciation_overrides,
            audio_path=run.output_audio_path or "",
            backend=run.backend,
            backend_version=run.backend_version or "",
            model_name=run.model_name,
            speaker=run.speaker,
            language=run.language,
            audio_post_processing=run.audio_post_processing,
            sample_rate_hz=run.sample_rate_hz or 0,
            channels=run.channels or 0,
            duration_sec=run.duration_sec or 0.0,
            audio_sha256=run.audio_sha256 or "",
        )
