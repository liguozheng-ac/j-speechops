"""Stage 9 generated-audio integrity and ASR round-trip content QA."""

from __future__ import annotations

import hashlib
import json
import math
import subprocess
import time
import unicodedata
from collections.abc import Iterable, Iterator
from dataclasses import dataclass
from enum import Enum
from os import PathLike
from pathlib import Path
from typing import Annotated, Literal

import jaconv
from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    JsonValue,
    StringConstraints,
    ValidationError,
    model_validator,
)

from .asr import ASRAdapter, ASRInfrastructureError, ASRRuntimeInfo
from .japanese_normalization import JapaneseNormalizer, NormalizationSampleError
from .japanese_reading import (
    ReadingOverrides,
    ReadingProvider,
    ReadingSampleError,
)
from .tts_synthesis import GeneratedAudioArtifact, validate_generated_wav


QA_SCHEMA_VERSION = "1.0"
GENERATED_AUDIO_QA_POLICY_VERSION = "generated-audio-content-qa-v1"
READING_CANONICALIZER_VERSION = "japanese-reading-comparison-v1"
NonEmptyStr = Annotated[str, StringConstraints(strip_whitespace=True, min_length=1)]


class QAModel(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)


class QAExecutionStatus(str, Enum):
    COMPLETED = "completed"
    FAILED = "failed"


class QADecision(str, Enum):
    PASS = "pass"
    REVIEW = "review"
    DROP = "drop"
    UNDECIDED = "undecided"


class QAIssueSeverity(str, Enum):
    INFO = "info"
    WARNING = "warning"
    CRITICAL = "critical"


class GeneratedAudioQAPolicy(QAModel):
    policy_version: Literal[GENERATED_AUDIO_QA_POLICY_VERSION] = (
        GENERATED_AUDIO_QA_POLICY_VERSION
    )
    reading_canonicalizer_version: Literal[READING_CANONICALIZER_VERSION] = (
        READING_CANONICALIZER_VERSION
    )
    duration_tolerance_sec: float = Field(default=0.001, gt=0.0)
    exact_canonical_reading_match_required: Literal[True] = True


class QAArtifactIssue(QAModel):
    code: NonEmptyStr
    severity: QAIssueSeverity
    message: NonEmptyStr
    evidence: dict[str, JsonValue] = Field(default_factory=dict)


class AudioIntegrityEvidence(QAModel):
    audio_path: NonEmptyStr
    file_size_bytes: int | None = Field(default=None, ge=0)
    expected_audio_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    observed_audio_sha256: str | None = Field(default=None, pattern=r"^[0-9a-f]{64}$")
    expected_sample_rate_hz: int = Field(gt=0)
    observed_sample_rate_hz: int | None = Field(default=None, gt=0)
    expected_channels: int = Field(gt=0)
    observed_channels: int | None = Field(default=None, gt=0)
    expected_duration_sec: float = Field(gt=0.0)
    observed_duration_sec: float | None = Field(default=None, gt=0.0)


class ASRRoundTripEvidence(QAModel):
    backend: NonEmptyStr
    backend_version: NonEmptyStr
    model_name: NonEmptyStr
    model_identity: NonEmptyStr
    config_fingerprint_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    transcript: str
    normalized_transcript: str
    observed_reading_kana: str
    inference_time_sec: float = Field(ge=0.0)
    audio_duration_sec: float = Field(gt=0.0)
    rtf: float = Field(ge=0.0)


class ContentComparisonEvidence(QAModel):
    canonicalizer_version: Literal[READING_CANONICALIZER_VERSION] = (
        READING_CANONICALIZER_VERSION
    )
    expected_reading_kana: str
    observed_reading_kana: str
    canonical_expected_reading: str
    canonical_observed_reading: str
    edit_distance: int = Field(ge=0)
    cer: float = Field(ge=0.0)


class GeneratedAudioQAResult(QAModel):
    schema_version: Literal[QA_SCHEMA_VERSION] = QA_SCHEMA_VERSION
    qa_id: str = Field(pattern=r"^[0-9a-f]{64}$")
    run_id: str = Field(pattern=r"^[0-9a-f]{64}$")
    sample_id: NonEmptyStr
    policy_version: Literal[GENERATED_AUDIO_QA_POLICY_VERSION]
    policy_fingerprint_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    status: QAExecutionStatus
    decision: QADecision
    audio_path: NonEmptyStr
    actual_synthesis_text: NonEmptyStr
    expected_audio_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    observed_audio_sha256: str | None = Field(default=None, pattern=r"^[0-9a-f]{64}$")
    expected_reading_kana: str | None = None
    asr_model_identity: NonEmptyStr
    asr_config_fingerprint_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    reading_provider_fingerprint_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    asr_transcript: str | None = None
    normalized_asr_transcript: str | None = None
    observed_reading_kana: str | None = None
    canonical_expected_reading: str | None = None
    canonical_observed_reading: str | None = None
    edit_distance: int | None = Field(default=None, ge=0)
    cer: float | None = Field(default=None, ge=0.0)
    asr_inference_time_sec: float | None = Field(default=None, ge=0.0)
    audio_duration_sec: float | None = Field(default=None, gt=0.0)
    asr_rtf: float | None = Field(default=None, ge=0.0)
    audio_integrity: AudioIntegrityEvidence
    asr_round_trip: ASRRoundTripEvidence | None = None
    content_comparison: ContentComparisonEvidence | None = None
    issues: tuple[QAArtifactIssue, ...] = ()
    decision_reasons: tuple[str, ...] = ()

    @model_validator(mode="after")
    def execution_and_decision_agree(self) -> "GeneratedAudioQAResult":
        if self.status is QAExecutionStatus.FAILED:
            if self.decision is not QADecision.UNDECIDED:
                raise ValueError("failed QA execution must be undecided")
        elif self.decision is QADecision.UNDECIDED:
            raise ValueError("completed QA execution cannot be undecided")
        if self.decision is QADecision.PASS:
            if self.asr_round_trip is None or self.content_comparison is None:
                raise ValueError("PASS requires round-trip and comparison evidence")
            if self.canonical_expected_reading != self.canonical_observed_reading:
                raise ValueError("PASS requires exact canonical reading equality")
            if any(
                issue.severity in {QAIssueSeverity.WARNING, QAIssueSeverity.CRITICAL}
                for issue in self.issues
            ):
                raise ValueError("PASS cannot contain WARNING or CRITICAL issues")
        if self.decision is QADecision.DROP and not any(
            issue.severity is QAIssueSeverity.CRITICAL for issue in self.issues
        ):
            raise ValueError("DROP requires an objective CRITICAL issue")
        if self.decision is QADecision.REVIEW:
            if self.asr_round_trip is None or self.content_comparison is None:
                raise ValueError("REVIEW requires round-trip and comparison evidence")
            if not any(
                issue.severity is QAIssueSeverity.WARNING for issue in self.issues
            ):
                raise ValueError("REVIEW requires a WARNING issue")
        if self.decision_reasons != tuple(issue.code for issue in self.issues):
            raise ValueError("decision reasons must match issue codes")
        return self


class QAPassedAudioArtifact(QAModel):
    schema_version: Literal[QA_SCHEMA_VERSION] = QA_SCHEMA_VERSION
    qa_id: str = Field(pattern=r"^[0-9a-f]{64}$")
    run_id: str = Field(pattern=r"^[0-9a-f]{64}$")
    sample_id: NonEmptyStr
    audio_path: NonEmptyStr
    audio_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    input_text: NonEmptyStr
    synthesis_text: NonEmptyStr
    expected_reading_kana: NonEmptyStr
    asr_transcript: str
    observed_reading_kana: str
    canonical_expected_reading: NonEmptyStr
    canonical_observed_reading: NonEmptyStr
    policy_version: Literal[GENERATED_AUDIO_QA_POLICY_VERSION]
    policy_fingerprint_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    asr_model_identity: NonEmptyStr
    asr_config_fingerprint_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    reading_provider_fingerprint_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    tts_backend: NonEmptyStr
    tts_model_name: NonEmptyStr
    speaker: NonEmptyStr


class QAReviewQueueItem(QAModel):
    schema_version: Literal[QA_SCHEMA_VERSION] = QA_SCHEMA_VERSION
    qa_id: str = Field(pattern=r"^[0-9a-f]{64}$")
    run_id: str = Field(pattern=r"^[0-9a-f]{64}$")
    sample_id: NonEmptyStr
    audio_path: NonEmptyStr
    actual_synthesis_text: NonEmptyStr
    expected_reading_kana: NonEmptyStr
    asr_transcript: str
    observed_reading_kana: str
    canonical_expected_reading: str
    canonical_observed_reading: str
    edit_distance: int = Field(ge=0)
    cer: float = Field(ge=0.0)
    issues: tuple[QAArtifactIssue, ...]


class QABatchSummary(QAModel):
    schema_version: Literal[QA_SCHEMA_VERSION] = QA_SCHEMA_VERSION
    policy_version: Literal[GENERATED_AUDIO_QA_POLICY_VERSION]
    policy_fingerprint_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    asr_model_identity: NonEmptyStr
    asr_config_fingerprint_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    reading_provider_fingerprint_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    total_samples: int = Field(ge=0)
    attempted_samples: int = Field(ge=0)
    completed_samples: int = Field(ge=0)
    failed_samples: int = Field(ge=0)
    passed_samples: int = Field(ge=0)
    review_samples: int = Field(ge=0)
    dropped_samples: int = Field(ge=0)
    undecided_samples: int = Field(ge=0)
    skipped_samples: int = Field(ge=0)
    asr_model_load_time_sec: float = Field(ge=0.0)
    batch_total_time_sec: float = Field(ge=0.0)
    gpu_name: str | None = None

    @model_validator(mode="after")
    def counts_agree(self) -> "QABatchSummary":
        if self.attempted_samples + self.skipped_samples != self.total_samples:
            raise ValueError("attempted plus skipped must equal total")
        if (
            self.passed_samples
            + self.review_samples
            + self.dropped_samples
            + self.undecided_samples
            != self.total_samples
        ):
            raise ValueError("decision counts must equal total")
        if self.completed_samples != (
            self.passed_samples + self.review_samples + self.dropped_samples
        ):
            raise ValueError("completed count must equal PASS + REVIEW + DROP")
        if self.failed_samples != self.undecided_samples:
            raise ValueError("failed count must equal UNDECIDED count")
        return self


@dataclass(frozen=True)
class QABatchOutputs:
    results_manifest: Path
    pass_manifest: Path
    review_queue: Path
    summary_file: Path
    summary: QABatchSummary


@dataclass(frozen=True)
class _InputRecord:
    artifact: GeneratedAudioArtifact
    expected_reading_missing: bool = False


@dataclass(frozen=True)
class _ProcessedRecord:
    artifact: GeneratedAudioArtifact
    result: GeneratedAudioQAResult


def _canonical_sha256(payload: object) -> str:
    encoded = json.dumps(
        payload, ensure_ascii=False, sort_keys=True, separators=(",", ":")
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def qa_policy_fingerprint(policy: GeneratedAudioQAPolicy) -> str:
    return _canonical_sha256(policy.model_dump(mode="json"))


def asr_config_fingerprint(runtime: ASRRuntimeInfo) -> str:
    effective = dict(runtime.effective_config)
    effective.pop("model_path", None)
    effective.pop("download_root", None)
    return _canonical_sha256(
        {
            "backend_name": runtime.backend_name,
            "backend_version": runtime.backend_version,
            "model_name": runtime.model_name,
            "device": runtime.device,
            "compute_type": runtime.compute_type,
            "language": runtime.language,
            "effective_config": effective,
        }
    )


def build_qa_id(
    *,
    run_id: str,
    audio_sha256: str,
    policy_fingerprint_sha256: str,
    asr_model_identity: str,
    asr_config_fingerprint_sha256: str,
    reading_provider_fingerprint_sha256: str,
    expected_reading_kana: str | None,
    expected_sample_rate_hz: int,
    expected_channels: int,
    expected_duration_sec: float,
    canonicalizer_version: str = READING_CANONICALIZER_VERSION,
) -> str:
    return _canonical_sha256(
        {
            "run_id": run_id,
            "audio_sha256": audio_sha256,
            "policy_fingerprint_sha256": policy_fingerprint_sha256,
            "asr_model_identity": asr_model_identity,
            "asr_config_fingerprint_sha256": asr_config_fingerprint_sha256,
            "reading_provider_fingerprint_sha256": reading_provider_fingerprint_sha256,
            "expected_reading_kana": expected_reading_kana,
            "expected_sample_rate_hz": expected_sample_rate_hz,
            "expected_channels": expected_channels,
            "expected_duration_sec": expected_duration_sec,
            "reading_canonicalizer_version": canonicalizer_version,
        }
    )


def canonicalize_japanese_reading(reading: str) -> str:
    """Normalize comparison-only presentation without erasing phonology."""

    normalized = unicodedata.normalize("NFKC", reading)
    katakana = jaconv.hira2kata(normalized)
    return "".join(
        character
        for character in katakana
        if not character.isspace()
        and not unicodedata.category(character).startswith("P")
    )


def detect_nvidia_gpu_name(device_index: int = 0) -> str | None:
    """Read optional GPU identity without requiring a CUDA-enabled PyTorch build."""

    try:
        completed = subprocess.run(
            [
                "nvidia-smi",
                "--query-gpu=name",
                "--format=csv,noheader",
            ],
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
            check=False,
        )
    except OSError:
        return None
    if completed.returncode != 0:
        return None
    names = [line.strip() for line in completed.stdout.splitlines() if line.strip()]
    return names[device_index] if device_index < len(names) else None


def levenshtein_distance(expected: str, observed: str) -> int:
    if len(expected) < len(observed):
        expected, observed = observed, expected
    previous = list(range(len(observed) + 1))
    for expected_index, expected_character in enumerate(expected, start=1):
        current = [expected_index]
        for observed_index, observed_character in enumerate(observed, start=1):
            current.append(
                min(
                    current[-1] + 1,
                    previous[observed_index] + 1,
                    previous[observed_index - 1]
                    + (expected_character != observed_character),
                )
            )
        previous = current
    return previous[-1]


def _iter_input_records(path: Path) -> Iterator[_InputRecord]:
    seen: set[str] = set()
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            try:
                payload = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"{path}:{line_number}: malformed JSON: {exc.msg}") from exc
            try:
                artifact = GeneratedAudioArtifact.model_validate(payload)
                missing = False
            except ValidationError as exc:
                errors = exc.errors()
                expected_only = errors and all(
                    error.get("loc") == ("expected_reading_kana",) for error in errors
                )
                if not expected_only:
                    raise ValueError(
                        f"{path}:{line_number}: invalid GeneratedAudioArtifact: {exc}"
                    ) from exc
                patched = dict(payload)
                patched["expected_reading_kana"] = "__missing_expected_reading__"
                try:
                    artifact = GeneratedAudioArtifact.model_validate(patched)
                except ValidationError as patched_exc:
                    raise ValueError(
                        f"{path}:{line_number}: invalid GeneratedAudioArtifact: {exc}"
                    ) from patched_exc
                missing = True
            if artifact.run_id in seen:
                raise ValueError(f"{path}:{line_number}: duplicate run_id {artifact.run_id}")
            seen.add(artifact.run_id)
            yield _InputRecord(artifact=artifact, expected_reading_missing=missing)


def write_qa_results(
    path: str | PathLike[str], results: Iterable[GeneratedAudioQAResult]
) -> int:
    values = sorted(
        (GeneratedAudioQAResult.model_validate(item) for item in results),
        key=lambda item: item.qa_id,
    )
    if len({item.qa_id for item in values}) != len(values):
        raise ValueError("duplicate qa_id")
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    with destination.open("w", encoding="utf-8", newline="\n") as handle:
        for item in values:
            handle.write(item.model_dump_json())
            handle.write("\n")
    return len(values)


def iter_qa_results(path: str | PathLike[str]) -> Iterator[GeneratedAudioQAResult]:
    source = Path(path)
    seen: set[str] = set()
    with source.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            try:
                item = GeneratedAudioQAResult.model_validate_json(line)
            except ValueError as exc:
                raise ValueError(
                    f"{source}:{line_number}: invalid GeneratedAudioQAResult: {exc}"
                ) from exc
            if item.qa_id in seen:
                raise ValueError(f"{source}:{line_number}: duplicate qa_id {item.qa_id}")
            seen.add(item.qa_id)
            yield item


def _write_models(path: Path, values: Iterable[QAModel], *, key: str) -> int:
    materialized = sorted(values, key=lambda item: str(getattr(item, key)))
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="\n") as handle:
        for item in materialized:
            handle.write(item.model_dump_json())
            handle.write("\n")
    return len(materialized)


class GeneratedAudioQAPipeline:
    """Evaluate generated artifacts without mutating Stage 8 provenance."""

    def __init__(
        self,
        *,
        output_root: Path,
        adapter: ASRAdapter,
        normalizer: JapaneseNormalizer,
        reading_provider: ReadingProvider,
        policy: GeneratedAudioQAPolicy | None = None,
        asr_model_load_time_sec: float = 0.0,
        gpu_name: str | None = None,
    ) -> None:
        self.output_root = Path(output_root).resolve()
        self.adapter = adapter
        self.normalizer = normalizer
        self.reading_provider = reading_provider
        self.reading_overrides = ReadingOverrides.empty()
        self.policy = policy or GeneratedAudioQAPolicy()
        self.policy_fingerprint = qa_policy_fingerprint(self.policy)
        self.runtime = adapter.runtime_info
        self.asr_model_identity = (
            f"{self.runtime.backend_name}:{self.runtime.backend_version}:"
            f"{self.runtime.model_name}"
        )
        self.asr_fingerprint = asr_config_fingerprint(self.runtime)
        self.reading_provider_fingerprint = _canonical_sha256(
            self.reading_provider.info.model_dump(mode="json")
        )
        self.asr_model_load_time_sec = asr_model_load_time_sec
        self.gpu_name = gpu_name

    def run(self, input_manifest: Path, *, force: bool = False) -> QABatchOutputs:
        started = time.perf_counter()
        source = Path(input_manifest).resolve()
        records = list(_iter_input_records(source))
        results_path = self.output_root / "generated_audio_qa_results.jsonl"
        pass_path = self.output_root / "qa_pass_audio_manifest.jsonl"
        review_path = self.output_root / "qa_review_queue.jsonl"
        summary_path = self.output_root / "qa_summary.json"
        previous = (
            {item.qa_id: item for item in iter_qa_results(results_path)}
            if results_path.exists() and not force
            else {}
        )

        processed: list[_ProcessedRecord] = []
        skipped = 0
        for record in records:
            qa_id = self._qa_id(
                record.artifact,
                expected_reading_kana=(
                    None
                    if record.expected_reading_missing
                    else record.artifact.expected_reading_kana
                ),
            )
            reusable = previous.get(qa_id)
            current_audio_path = self._resolve_audio(record.artifact, source.parent)
            if reusable is not None and self._can_reuse(
                record.artifact, reusable, current_audio_path
            ):
                processed.append(_ProcessedRecord(record.artifact, reusable))
                skipped += 1
                continue
            result = self._process_record(record, source.parent, qa_id)
            processed.append(_ProcessedRecord(record.artifact, result))

        results = [item.result for item in processed]
        passes = [self._pass_artifact(item) for item in processed if item.result.decision is QADecision.PASS]
        reviews = [self._review_item(item) for item in processed if item.result.decision is QADecision.REVIEW]
        write_qa_results(results_path, results)
        _write_models(pass_path, passes, key="qa_id")
        _write_models(review_path, reviews, key="qa_id")

        summary = QABatchSummary(
            policy_version=self.policy.policy_version,
            policy_fingerprint_sha256=self.policy_fingerprint,
            asr_model_identity=self.asr_model_identity,
            asr_config_fingerprint_sha256=self.asr_fingerprint,
            reading_provider_fingerprint_sha256=self.reading_provider_fingerprint,
            total_samples=len(results),
            attempted_samples=len(results) - skipped,
            completed_samples=sum(item.status is QAExecutionStatus.COMPLETED for item in results),
            failed_samples=sum(item.status is QAExecutionStatus.FAILED for item in results),
            passed_samples=sum(item.decision is QADecision.PASS for item in results),
            review_samples=sum(item.decision is QADecision.REVIEW for item in results),
            dropped_samples=sum(item.decision is QADecision.DROP for item in results),
            undecided_samples=sum(item.decision is QADecision.UNDECIDED for item in results),
            skipped_samples=skipped,
            asr_model_load_time_sec=max(
                self.asr_model_load_time_sec,
                float(getattr(self.adapter, "model_load_time_sec", 0.0)),
            ),
            batch_total_time_sec=time.perf_counter() - started,
            gpu_name=self.gpu_name,
        )
        self.output_root.mkdir(parents=True, exist_ok=True)
        summary_path.write_text(summary.model_dump_json(indent=2) + "\n", encoding="utf-8")
        return QABatchOutputs(results_path, pass_path, review_path, summary_path, summary)

    def _qa_id(
        self,
        artifact: GeneratedAudioArtifact,
        *,
        expected_reading_kana: str | None,
    ) -> str:
        return build_qa_id(
            run_id=artifact.run_id,
            audio_sha256=artifact.audio_sha256,
            policy_fingerprint_sha256=self.policy_fingerprint,
            asr_model_identity=self.asr_model_identity,
            asr_config_fingerprint_sha256=self.asr_fingerprint,
            reading_provider_fingerprint_sha256=self.reading_provider_fingerprint,
            expected_reading_kana=expected_reading_kana,
            expected_sample_rate_hz=artifact.sample_rate_hz,
            expected_channels=artifact.channels,
            expected_duration_sec=artifact.duration_sec,
        )

    def _resolve_audio(self, artifact: GeneratedAudioArtifact, manifest_dir: Path) -> Path:
        raw = Path(artifact.audio_path)
        return raw.resolve() if raw.is_absolute() else (manifest_dir / raw).resolve()

    def _process_record(
        self, record: _InputRecord, manifest_dir: Path, qa_id: str
    ) -> GeneratedAudioQAResult:
        artifact = record.artifact
        audio_path = self._resolve_audio(artifact, manifest_dir)
        if record.expected_reading_missing:
            issue = QAArtifactIssue(
                code="expected_reading_missing",
                severity=QAIssueSeverity.CRITICAL,
                message="Stage 8 success artifact lacks expected_reading_kana provenance",
            )
            return self._result(
                artifact=artifact,
                qa_id=qa_id,
                audio_path=audio_path,
                expected_reading_kana=None,
                status=QAExecutionStatus.FAILED,
                decision=QADecision.UNDECIDED,
                integrity=self._empty_integrity(artifact, audio_path),
                issues=(issue,),
            )

        integrity, integrity_issues = self._validate_integrity(artifact, audio_path)
        if any(issue.severity is QAIssueSeverity.CRITICAL for issue in integrity_issues):
            return self._result(
                artifact=artifact,
                qa_id=qa_id,
                audio_path=audio_path,
                expected_reading_kana=artifact.expected_reading_kana,
                status=QAExecutionStatus.COMPLETED,
                decision=QADecision.DROP,
                integrity=integrity,
                issues=integrity_issues,
            )

        assert integrity.observed_duration_sec is not None
        load_time_before = float(getattr(self.adapter, "model_load_time_sec", 0.0))
        inference_started = time.perf_counter()
        try:
            asr_result = self.adapter.transcribe(audio_path)
        except ASRInfrastructureError:
            raise
        except Exception as exc:
            issue = QAArtifactIssue(
                code="asr_sample_failure",
                severity=QAIssueSeverity.WARNING,
                message=f"ASR failed for this sample: {type(exc).__name__}: {exc}",
            )
            return self._result(
                artifact=artifact,
                qa_id=qa_id,
                audio_path=audio_path,
                expected_reading_kana=artifact.expected_reading_kana,
                status=QAExecutionStatus.FAILED,
                decision=QADecision.UNDECIDED,
                integrity=integrity,
                issues=(*integrity_issues, issue),
            )
        elapsed_with_optional_load = time.perf_counter() - inference_started
        load_time_after = float(getattr(self.adapter, "model_load_time_sec", 0.0))
        inference_time = max(
            0.0,
            elapsed_with_optional_load - max(0.0, load_time_after - load_time_before),
        )

        transcript = asr_result.text.strip()
        issues = list(integrity_issues)
        if not transcript:
            normalized_transcript = ""
            observed_reading = ""
            issues.append(
                QAArtifactIssue(
                    code="asr_empty_transcript",
                    severity=QAIssueSeverity.WARNING,
                    message="Whisper returned an empty transcript for valid non-silent audio",
                )
            )
        else:
            try:
                normalized_transcript = self.normalizer.normalize(transcript).normalized_text
                observed_reading = self.reading_provider.prepare(
                    artifact.sample_id,
                    normalized_transcript,
                    self.reading_overrides,
                ).reading_kana
            except (NormalizationSampleError, ReadingSampleError) as exc:
                issue = QAArtifactIssue(
                    code="reading_preparation_failed",
                    severity=QAIssueSeverity.WARNING,
                    message=f"ASR text could not be prepared for comparison: {exc}",
                )
                return self._result(
                    artifact=artifact,
                    qa_id=qa_id,
                    audio_path=audio_path,
                    expected_reading_kana=artifact.expected_reading_kana,
                    status=QAExecutionStatus.FAILED,
                    decision=QADecision.UNDECIDED,
                    integrity=integrity,
                    issues=(*issues, issue),
                    asr_transcript=transcript,
                    normalized_asr_transcript=None,
                    asr_inference_time_sec=inference_time,
                )

        expected_canonical = canonicalize_japanese_reading(artifact.expected_reading_kana)
        observed_canonical = canonicalize_japanese_reading(observed_reading)
        if not expected_canonical:
            issue = QAArtifactIssue(
                code="expected_reading_empty_after_canonicalization",
                severity=QAIssueSeverity.CRITICAL,
                message="expected reading has no comparable characters after canonicalization",
            )
            return self._result(
                artifact=artifact,
                qa_id=qa_id,
                audio_path=audio_path,
                expected_reading_kana=artifact.expected_reading_kana,
                status=QAExecutionStatus.FAILED,
                decision=QADecision.UNDECIDED,
                integrity=integrity,
                issues=(*issues, issue),
                asr_transcript=transcript,
                normalized_asr_transcript=normalized_transcript,
                observed_reading_kana=observed_reading,
                asr_inference_time_sec=inference_time,
            )
        distance = levenshtein_distance(expected_canonical, observed_canonical)
        cer = distance / len(expected_canonical)
        if expected_canonical != observed_canonical:
            issues.append(
                QAArtifactIssue(
                    code="content_reading_mismatch",
                    severity=QAIssueSeverity.WARNING,
                    message="canonical expected and observed readings differ",
                    evidence={
                        "expected": expected_canonical,
                        "observed": observed_canonical,
                        "edit_distance": distance,
                        "cer": cer,
                        "asr_transcript": transcript,
                    },
                )
            )
        comparison = ContentComparisonEvidence(
            expected_reading_kana=artifact.expected_reading_kana,
            observed_reading_kana=observed_reading,
            canonical_expected_reading=expected_canonical,
            canonical_observed_reading=observed_canonical,
            edit_distance=distance,
            cer=cer,
        )
        round_trip = ASRRoundTripEvidence(
            backend=self.runtime.backend_name,
            backend_version=self.runtime.backend_version,
            model_name=self.runtime.model_name,
            model_identity=self.asr_model_identity,
            config_fingerprint_sha256=self.asr_fingerprint,
            transcript=transcript,
            normalized_transcript=normalized_transcript,
            observed_reading_kana=observed_reading,
            inference_time_sec=inference_time,
            audio_duration_sec=integrity.observed_duration_sec,
            rtf=inference_time / integrity.observed_duration_sec,
        )
        decision = (
            QADecision.REVIEW
            if any(issue.severity is QAIssueSeverity.WARNING for issue in issues)
            else QADecision.PASS
        )
        return self._result(
            artifact=artifact,
            qa_id=qa_id,
            audio_path=audio_path,
            expected_reading_kana=artifact.expected_reading_kana,
            status=QAExecutionStatus.COMPLETED,
            decision=decision,
            integrity=integrity,
            issues=tuple(issues),
            round_trip=round_trip,
            comparison=comparison,
        )

    def _validate_integrity(
        self, artifact: GeneratedAudioArtifact, audio_path: Path
    ) -> tuple[AudioIntegrityEvidence, tuple[QAArtifactIssue, ...]]:
        if not audio_path.is_file():
            issue = QAArtifactIssue(
                code="audio_missing",
                severity=QAIssueSeverity.CRITICAL,
                message=f"generated audio file is missing: {audio_path}",
            )
            return self._empty_integrity(artifact, audio_path), (issue,)
        size = audio_path.stat().st_size
        if size <= 44:
            issue = QAArtifactIssue(
                code="audio_empty",
                severity=QAIssueSeverity.CRITICAL,
                message=f"generated audio file is empty or has no frames: {audio_path}",
                evidence={"file_size_bytes": size},
            )
            return self._empty_integrity(artifact, audio_path, size=size), (issue,)
        try:
            validation = validate_generated_wav(audio_path)
        except ValueError as exc:
            message = str(exc)
            if "not decodable" in message:
                code = "audio_unreadable"
            elif "non-finite" in message:
                code = "audio_non_finite"
            elif "silent" in message:
                code = "audio_silent"
            elif "metadata" in message:
                code = "audio_invalid_metadata"
            elif "duration" in message:
                code = "audio_invalid_duration"
            else:
                code = "audio_invalid"
            issue = QAArtifactIssue(
                code=code,
                severity=QAIssueSeverity.CRITICAL,
                message=message,
            )
            return self._empty_integrity(artifact, audio_path, size=size), (issue,)

        evidence = AudioIntegrityEvidence(
            audio_path=str(audio_path),
            file_size_bytes=size,
            expected_audio_sha256=artifact.audio_sha256,
            observed_audio_sha256=validation.audio_sha256,
            expected_sample_rate_hz=artifact.sample_rate_hz,
            observed_sample_rate_hz=validation.sample_rate_hz,
            expected_channels=artifact.channels,
            observed_channels=validation.channels,
            expected_duration_sec=artifact.duration_sec,
            observed_duration_sec=validation.duration_sec,
        )
        issues: list[QAArtifactIssue] = []
        if validation.audio_sha256 != artifact.audio_sha256:
            issues.append(
                QAArtifactIssue(
                    code="audio_hash_mismatch",
                    severity=QAIssueSeverity.CRITICAL,
                    message="decoded audio SHA-256 differs from Stage 8 provenance",
                    evidence={
                        "expected": artifact.audio_sha256,
                        "observed": validation.audio_sha256,
                    },
                )
            )
        if validation.sample_rate_hz != artifact.sample_rate_hz:
            issues.append(
                QAArtifactIssue(
                    code="audio_sample_rate_mismatch",
                    severity=QAIssueSeverity.CRITICAL,
                    message="decoded sample rate differs from Stage 8 metadata",
                    evidence={
                        "expected": artifact.sample_rate_hz,
                        "observed": validation.sample_rate_hz,
                    },
                )
            )
        if validation.channels != artifact.channels:
            issues.append(
                QAArtifactIssue(
                    code="audio_channel_count_mismatch",
                    severity=QAIssueSeverity.CRITICAL,
                    message="decoded channel count differs from Stage 8 metadata",
                    evidence={
                        "expected": artifact.channels,
                        "observed": validation.channels,
                    },
                )
            )
        if not math.isclose(
            validation.duration_sec,
            artifact.duration_sec,
            rel_tol=0.0,
            abs_tol=self.policy.duration_tolerance_sec,
        ):
            issues.append(
                QAArtifactIssue(
                    code="audio_duration_mismatch",
                    severity=QAIssueSeverity.CRITICAL,
                    message="decoded duration differs materially from Stage 8 metadata",
                    evidence={
                        "expected": artifact.duration_sec,
                        "observed": validation.duration_sec,
                        "tolerance_sec": self.policy.duration_tolerance_sec,
                    },
                )
            )
        return evidence, tuple(issues)

    @staticmethod
    def _empty_integrity(
        artifact: GeneratedAudioArtifact,
        audio_path: Path,
        *,
        size: int | None = None,
    ) -> AudioIntegrityEvidence:
        return AudioIntegrityEvidence(
            audio_path=str(audio_path),
            file_size_bytes=size,
            expected_audio_sha256=artifact.audio_sha256,
            expected_sample_rate_hz=artifact.sample_rate_hz,
            expected_channels=artifact.channels,
            expected_duration_sec=artifact.duration_sec,
        )

    def _result(
        self,
        *,
        artifact: GeneratedAudioArtifact,
        qa_id: str,
        audio_path: Path,
        expected_reading_kana: str | None,
        status: QAExecutionStatus,
        decision: QADecision,
        integrity: AudioIntegrityEvidence,
        issues: tuple[QAArtifactIssue, ...],
        round_trip: ASRRoundTripEvidence | None = None,
        comparison: ContentComparisonEvidence | None = None,
        asr_transcript: str | None = None,
        normalized_asr_transcript: str | None = None,
        observed_reading_kana: str | None = None,
        asr_inference_time_sec: float | None = None,
    ) -> GeneratedAudioQAResult:
        if round_trip is not None:
            asr_transcript = round_trip.transcript
            normalized_asr_transcript = round_trip.normalized_transcript
            observed_reading_kana = round_trip.observed_reading_kana
            asr_inference_time_sec = round_trip.inference_time_sec
        return GeneratedAudioQAResult(
            qa_id=qa_id,
            run_id=artifact.run_id,
            sample_id=artifact.sample_id,
            policy_version=self.policy.policy_version,
            policy_fingerprint_sha256=self.policy_fingerprint,
            status=status,
            decision=decision,
            audio_path=str(audio_path),
            actual_synthesis_text=artifact.synthesis_text,
            expected_audio_sha256=artifact.audio_sha256,
            observed_audio_sha256=integrity.observed_audio_sha256,
            expected_reading_kana=expected_reading_kana,
            asr_model_identity=self.asr_model_identity,
            asr_config_fingerprint_sha256=self.asr_fingerprint,
            reading_provider_fingerprint_sha256=self.reading_provider_fingerprint,
            asr_transcript=asr_transcript,
            normalized_asr_transcript=normalized_asr_transcript,
            observed_reading_kana=observed_reading_kana,
            canonical_expected_reading=(
                comparison.canonical_expected_reading if comparison else None
            ),
            canonical_observed_reading=(
                comparison.canonical_observed_reading if comparison else None
            ),
            edit_distance=comparison.edit_distance if comparison else None,
            cer=comparison.cer if comparison else None,
            asr_inference_time_sec=asr_inference_time_sec,
            audio_duration_sec=integrity.observed_duration_sec,
            asr_rtf=round_trip.rtf if round_trip else None,
            audio_integrity=integrity,
            asr_round_trip=round_trip,
            content_comparison=comparison,
            issues=issues,
            decision_reasons=tuple(issue.code for issue in issues),
        )

    @staticmethod
    def _can_reuse(
        artifact: GeneratedAudioArtifact,
        previous: GeneratedAudioQAResult,
        current_audio_path: Path,
    ) -> bool:
        if previous.status is not QAExecutionStatus.COMPLETED:
            return False
        if Path(previous.audio_path).resolve() != current_audio_path.resolve():
            return False
        if not current_audio_path.is_file():
            return False
        observed = hashlib.sha256(current_audio_path.read_bytes()).hexdigest()
        return observed == artifact.audio_sha256 == previous.observed_audio_sha256

    @staticmethod
    def _pass_artifact(item: _ProcessedRecord) -> QAPassedAudioArtifact:
        artifact = item.artifact
        result = item.result
        assert result.decision is QADecision.PASS
        return QAPassedAudioArtifact(
            qa_id=result.qa_id,
            run_id=result.run_id,
            sample_id=result.sample_id,
            audio_path=result.audio_path,
            audio_sha256=result.observed_audio_sha256 or "",
            input_text=artifact.input_text,
            synthesis_text=artifact.synthesis_text,
            expected_reading_kana=result.expected_reading_kana or "",
            asr_transcript=result.asr_transcript or "",
            observed_reading_kana=result.observed_reading_kana or "",
            canonical_expected_reading=result.canonical_expected_reading or "",
            canonical_observed_reading=result.canonical_observed_reading or "",
            policy_version=result.policy_version,
            policy_fingerprint_sha256=result.policy_fingerprint_sha256,
            asr_model_identity=result.asr_model_identity,
            asr_config_fingerprint_sha256=result.asr_config_fingerprint_sha256,
            reading_provider_fingerprint_sha256=(
                result.reading_provider_fingerprint_sha256
            ),
            tts_backend=artifact.backend,
            tts_model_name=artifact.model_name,
            speaker=artifact.speaker,
        )

    @staticmethod
    def _review_item(item: _ProcessedRecord) -> QAReviewQueueItem:
        result = item.result
        assert result.decision is QADecision.REVIEW
        return QAReviewQueueItem(
            qa_id=result.qa_id,
            run_id=result.run_id,
            sample_id=result.sample_id,
            audio_path=result.audio_path,
            actual_synthesis_text=result.actual_synthesis_text,
            expected_reading_kana=result.expected_reading_kana or "",
            asr_transcript=result.asr_transcript or "",
            observed_reading_kana=result.observed_reading_kana or "",
            canonical_expected_reading=result.canonical_expected_reading or "",
            canonical_observed_reading=result.canonical_observed_reading or "",
            edit_distance=result.edit_distance or 0,
            cer=result.cer or 0.0,
            issues=result.issues,
        )
