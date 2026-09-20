"""Deterministic Stage 4 speech-data curation and reporting."""

from __future__ import annotations

import hashlib
import unicodedata
from collections import Counter
from collections.abc import Iterator
from dataclasses import dataclass, field
from enum import Enum
from os import PathLike
from pathlib import Path

from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    JsonValue,
    field_validator,
    model_validator,
)

from .lifecycle import transition_stage
from .manifest import iter_manifest, write_manifest
from .schemas import CurationDecision, ProcessingStage, RightsStatus, SpeechSample


class CurationPolicy(BaseModel):
    """Versioned operational baselines, not linguistic quality standards."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    policy_version: str = Field(default="speech-curation-v1", min_length=1)
    min_duration_sec: float = Field(default=0.5, ge=0)
    max_duration_sec: float = Field(default=30.0, gt=0)
    min_chars_per_sec: float = Field(default=1.0, ge=0)
    max_chars_per_sec: float = Field(default=20.0, gt=0)
    min_japanese_script_ratio: float = Field(default=0.3, ge=0, le=1)
    repetition_min_count: int = Field(default=3, ge=2)
    repetition_max_unit_chars: int = Field(default=20, gt=0)
    require_cleared_rights: bool = True

    @model_validator(mode="after")
    def validate_ranges(self) -> "CurationPolicy":
        if self.max_duration_sec <= self.min_duration_sec:
            raise ValueError("max_duration_sec must be greater than min_duration_sec")
        if self.max_chars_per_sec <= self.min_chars_per_sec:
            raise ValueError("max_chars_per_sec must be greater than min_chars_per_sec")
        return self


class CurationIssueSeverity(str, Enum):
    INFO = "info"
    WARNING = "warning"
    CRITICAL = "critical"


class CurationIssue(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    code: str = Field(min_length=1)
    severity: CurationIssueSeverity
    message: str = Field(min_length=1)
    evidence: dict[str, JsonValue] = Field(default_factory=dict)


class SpeechCurationReport(BaseModel):
    model_config = ConfigDict(extra="forbid")

    sample_id: str = Field(min_length=1)
    policy_version: str = Field(min_length=1)
    decision: CurationDecision
    audio_duration_sec: float | None = Field(default=None, ge=0)
    transcript_chars: int = Field(ge=0)
    characters_per_second: float | None = Field(default=None, ge=0)
    japanese_script_ratio: float | None = Field(default=None, ge=0, le=1)
    audio_sha256: str | None = Field(default=None, pattern=r"^[0-9a-f]{64}$")
    issues: list[CurationIssue] = Field(default_factory=list)
    decision_reasons: list[str] = Field(default_factory=list)
    skipped_existing: bool = False

    @field_validator("decision")
    @classmethod
    def decision_must_be_final(cls, value: CurationDecision) -> CurationDecision:
        if value is CurationDecision.UNDECIDED:
            raise ValueError("Stage 4 reports require PASS, REVIEW, or DROP")
        return value


class IssueCount(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    code: str
    count: int = Field(gt=0)


class CurationBatchSummary(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    total_samples: int = Field(ge=0)
    pass_count: int = Field(ge=0)
    review_count: int = Field(ge=0)
    drop_count: int = Field(ge=0)
    total_audio_duration_sec: float = Field(ge=0)
    pass_duration_sec: float = Field(ge=0)
    review_duration_sec: float = Field(ge=0)
    drop_duration_sec: float = Field(ge=0)
    top_issue_codes: list[IssueCount] = Field(default_factory=list)


@dataclass(frozen=True)
class CurationBatchOutputs:
    curated_manifest: Path
    usable_manifest: Path
    report_file: Path
    summary: CurationBatchSummary


@dataclass
class _SampleAnalysis:
    sample: SpeechSample
    duration_sec: float | None
    transcript_chars: int
    characters_per_second: float | None
    japanese_script_ratio: float | None
    audio_sha256: str | None
    issues: list[CurationIssue] = field(default_factory=list)


def _issue(
    code: str,
    severity: CurationIssueSeverity,
    message: str,
    **evidence: JsonValue,
) -> CurationIssue:
    return CurationIssue(
        code=code, severity=severity, message=message, evidence=evidence
    )


def _analysis_text(text: str | None) -> str:
    return "" if text is None else "".join(char for char in text if not char.isspace())


def _is_japanese_script(char: str) -> bool:
    codepoint = ord(char)
    return (
        0x3040 <= codepoint <= 0x309F
        or 0x30A0 <= codepoint <= 0x30FF
        or 0x3400 <= codepoint <= 0x4DBF
        or 0x4E00 <= codepoint <= 0x9FFF
        or 0xF900 <= codepoint <= 0xFAFF
    )


def japanese_script_ratio(text: str) -> float | None:
    """Return a Unicode-range heuristic, not Japanese language identification."""
    eligible = [char for char in text if unicodedata.category(char)[0] in {"L", "N"}]
    if not eligible:
        return None
    return sum(_is_japanese_script(char) for char in eligible) / len(eligible)


def has_excessive_repetition(
    text: str, *, min_count: int, max_unit_chars: int
) -> tuple[bool, str | None, int | None]:
    compact = "".join(
        char
        for char in text
        if not char.isspace() and unicodedata.category(char)[0] != "P"
    )
    maximum = min(max_unit_chars, len(compact) // min_count)
    for unit_length in range(1, maximum + 1):
        if len(compact) % unit_length:
            continue
        count = len(compact) // unit_length
        unit = compact[:unit_length]
        if count >= min_count and unit * count == compact:
            return True, unit, count
    return False, None, None


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _decision_for(issues: list[CurationIssue]) -> CurationDecision:
    if any(issue.severity is CurationIssueSeverity.CRITICAL for issue in issues):
        return CurationDecision.DROP
    if any(issue.severity is CurationIssueSeverity.WARNING for issue in issues):
        return CurationDecision.REVIEW
    return CurationDecision.PASS


def write_speech_curation_reports(
    path: str | PathLike[str], reports: list[SpeechCurationReport]
) -> int:
    destination = Path(path)
    with destination.open("w", encoding="utf-8", newline="\n") as handle:
        for report in reports:
            handle.write(report.model_dump_json())
            handle.write("\n")
    return len(reports)


def iter_speech_curation_reports(
    path: str | PathLike[str],
) -> Iterator[SpeechCurationReport]:
    report_path = Path(path)
    with report_path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            try:
                yield SpeechCurationReport.model_validate_json(line)
            except ValueError as exc:
                raise ValueError(
                    f"{report_path}:{line_number}: invalid speech curation report: {exc}"
                ) from exc


class SpeechCurationPipeline:
    def __init__(
        self,
        *,
        dataset_root: Path,
        output_root: Path,
        policy: CurationPolicy | None = None,
    ) -> None:
        self.dataset_root = dataset_root.resolve()
        self.output_root = output_root.resolve()
        if not self.output_root.is_relative_to(self.dataset_root):
            raise ValueError("output_root must be inside dataset_root")
        self.policy = policy or CurationPolicy()
        self.manifest_dir = self.output_root / "manifests"
        self.report_dir = self.output_root / "reports"

    def run(self, input_manifest: Path, *, force: bool = False) -> CurationBatchOutputs:
        samples = list(iter_manifest(input_manifest))
        sample_ids = [sample.sample_id for sample in samples]
        if len(sample_ids) != len(set(sample_ids)):
            raise ValueError("input manifest contains duplicate sample_id values")
        self.manifest_dir.mkdir(parents=True, exist_ok=True)
        self.report_dir.mkdir(parents=True, exist_ok=True)
        curated_manifest = self.manifest_dir / "curated_segments.jsonl"
        usable_manifest = self.manifest_dir / "usable_speech.jsonl"
        report_file = self.report_dir / "speech_curation_report.jsonl"
        previous_reports = self._load_previous_reports(report_file) if not force else {}

        curated: list[SpeechSample] = []
        reports: list[SpeechCurationReport] = []
        analyses: list[_SampleAnalysis] = []

        for sample in samples:
            if sample.state.stage is ProcessingStage.CURATED and not force:
                if sample.state.curation_decision is CurationDecision.UNDECIDED:
                    raise ValueError(
                        f"curated sample {sample.sample_id} has an undecided curation state"
                    )
                curated.append(sample.model_copy(deep=True))
                previous = previous_reports.get(sample.sample_id)
                if (
                    previous is not None
                    and previous.decision is sample.state.curation_decision
                ):
                    reports.append(previous)
                else:
                    reports.append(self._skipped_report(sample))
                continue
            if (
                sample.state.stage is not ProcessingStage.TRANSCRIBED
                and sample.state.stage is not ProcessingStage.CURATED
            ):
                raise ValueError(
                    f"sample {sample.sample_id} must be transcribed before Stage 4; "
                    f"got {sample.state.stage.value}"
                )
            analyses.append(self._analyze(sample))

        duplicate_groups: dict[str, list[_SampleAnalysis]] = {}
        for analysis in analyses:
            if analysis.audio_sha256 is not None:
                duplicate_groups.setdefault(analysis.audio_sha256, []).append(analysis)
        for digest, group in duplicate_groups.items():
            if len(group) < 2:
                continue
            sample_ids = [analysis.sample.sample_id for analysis in group]
            for analysis in group:
                analysis.issues.append(
                    _issue(
                        "duplicate_audio",
                        CurationIssueSeverity.WARNING,
                        "audio content hash is shared by multiple samples",
                        audio_sha256=digest,
                        duplicate_sample_ids=sample_ids,
                    )
                )

        analyzed_outputs = [self._finalize(analysis) for analysis in analyses]
        curated.extend(sample for sample, _ in analyzed_outputs)
        reports.extend(report for _, report in analyzed_outputs)

        input_order = {sample.sample_id: index for index, sample in enumerate(samples)}
        curated.sort(key=lambda sample: input_order[sample.sample_id])
        reports.sort(key=lambda report: input_order[report.sample_id])
        usable = [
            sample
            for sample in curated
            if sample.state.curation_decision is CurationDecision.PASS
        ]

        write_manifest(curated_manifest, curated)
        write_manifest(usable_manifest, usable)
        write_speech_curation_reports(report_file, reports)
        summary = self._summarize(curated, reports)
        return CurationBatchOutputs(
            curated_manifest, usable_manifest, report_file, summary
        )

    @staticmethod
    def _load_previous_reports(path: Path) -> dict[str, SpeechCurationReport]:
        if not path.exists():
            return {}
        return {report.sample_id: report for report in iter_speech_curation_reports(path)}

    def _analyze(self, sample: SpeechSample) -> _SampleAnalysis:
        text = _analysis_text(sample.text.asr_text)
        duration = sample.audio.duration_sec
        transcript_chars = len(text)
        chars_per_sec = (
            transcript_chars / duration
            if duration is not None and duration > 0
            else None
        )
        script_ratio = japanese_script_ratio(text) if text else None
        issues: list[CurationIssue] = []

        if not text:
            issues.append(
                _issue(
                    "empty_transcript",
                    CurationIssueSeverity.CRITICAL,
                    "ASR transcript is empty",
                )
            )

        source_path = Path(sample.audio.path)
        resolved = (
            source_path
            if source_path.is_absolute()
            else self.dataset_root / source_path
        )
        audio_sha256: str | None = None
        if not resolved.exists() or not resolved.is_file():
            issues.append(
                _issue(
                    "audio_missing",
                    CurationIssueSeverity.CRITICAL,
                    "canonical audio path does not reference an accessible file",
                    audio_path=sample.audio.path,
                )
            )
        else:
            try:
                audio_sha256 = _sha256(resolved)
            except OSError:
                issues.append(
                    _issue(
                        "audio_unreadable",
                        CurationIssueSeverity.CRITICAL,
                        "canonical audio file could not be read",
                        audio_path=sample.audio.path,
                    )
                )

        if sample.source.rights_status is RightsStatus.RESTRICTED:
            issues.append(
                _issue(
                    "rights_restricted",
                    CurationIssueSeverity.CRITICAL,
                    "source rights explicitly prohibit usable-data routing",
                    rights_status=sample.source.rights_status.value,
                )
            )
        elif (
            sample.source.rights_status is RightsStatus.UNKNOWN
            and self.policy.require_cleared_rights
        ):
            issues.append(
                _issue(
                    "rights_unknown",
                    CurationIssueSeverity.WARNING,
                    "source rights have not been cleared",
                    rights_status=sample.source.rights_status.value,
                )
            )

        if duration is None:
            issues.append(
                _issue(
                    "duration_unavailable",
                    CurationIssueSeverity.WARNING,
                    "audio duration is unavailable for curation signals",
                )
            )
        elif duration < self.policy.min_duration_sec:
            issues.append(
                _issue(
                    "duration_too_short",
                    CurationIssueSeverity.WARNING,
                    "segment is shorter than the operational baseline",
                    actual_duration_sec=duration,
                    min_duration_sec=self.policy.min_duration_sec,
                )
            )
        elif duration > self.policy.max_duration_sec:
            issues.append(
                _issue(
                    "duration_too_long",
                    CurationIssueSeverity.WARNING,
                    "segment is longer than the operational baseline",
                    actual_duration_sec=duration,
                    max_duration_sec=self.policy.max_duration_sec,
                )
            )

        if text and chars_per_sec is not None and (
            chars_per_sec < self.policy.min_chars_per_sec
            or chars_per_sec > self.policy.max_chars_per_sec
        ):
            issues.append(
                _issue(
                    "transcript_density_anomaly",
                    CurationIssueSeverity.WARNING,
                    "transcript character density is outside the operational baseline",
                    actual_chars_per_sec=chars_per_sec,
                    min_chars_per_sec=self.policy.min_chars_per_sec,
                    max_chars_per_sec=self.policy.max_chars_per_sec,
                )
            )

        if (
            text
            and script_ratio is not None
            and script_ratio < self.policy.min_japanese_script_ratio
        ):
            issues.append(
                _issue(
                    "low_japanese_script_ratio",
                    CurationIssueSeverity.WARNING,
                    "Japanese Unicode script ratio is below the operational baseline",
                    actual_ratio=script_ratio,
                    min_ratio=self.policy.min_japanese_script_ratio,
                )
            )

        repeated, repeated_unit, repetition_count = has_excessive_repetition(
            text,
            min_count=self.policy.repetition_min_count,
            max_unit_chars=self.policy.repetition_max_unit_chars,
        )
        if repeated:
            issues.append(
                _issue(
                    "excessive_text_repetition",
                    CurationIssueSeverity.WARNING,
                    "transcript consists of an exactly repeated short unit",
                    repeated_unit=repeated_unit,
                    repetition_count=repetition_count,
                )
            )

        return _SampleAnalysis(
            sample=sample,
            duration_sec=duration,
            transcript_chars=transcript_chars,
            characters_per_second=chars_per_sec,
            japanese_script_ratio=script_ratio,
            audio_sha256=audio_sha256,
            issues=issues,
        )

    def _finalize(
        self, analysis: _SampleAnalysis
    ) -> tuple[SpeechSample, SpeechCurationReport]:
        decision = _decision_for(analysis.issues)
        reasons = [issue.code for issue in analysis.issues]
        updated = transition_stage(analysis.sample, ProcessingStage.CURATED)
        updated.state.curation_decision = decision
        updated.state.decision_reasons = reasons
        report = SpeechCurationReport(
            sample_id=analysis.sample.sample_id,
            policy_version=self.policy.policy_version,
            decision=decision,
            audio_duration_sec=analysis.duration_sec,
            transcript_chars=analysis.transcript_chars,
            characters_per_second=analysis.characters_per_second,
            japanese_script_ratio=analysis.japanese_script_ratio,
            audio_sha256=analysis.audio_sha256,
            issues=analysis.issues,
            decision_reasons=reasons,
        )
        return updated, report

    def _skipped_report(self, sample: SpeechSample) -> SpeechCurationReport:
        text = _analysis_text(sample.text.asr_text)
        return SpeechCurationReport(
            sample_id=sample.sample_id,
            policy_version="existing-curation-policy-unknown",
            decision=sample.state.curation_decision,
            audio_duration_sec=sample.audio.duration_sec,
            transcript_chars=len(text),
            issues=[],
            decision_reasons=list(sample.state.decision_reasons),
            skipped_existing=True,
        )

    @staticmethod
    def _summarize(
        curated: list[SpeechSample], reports: list[SpeechCurationReport]
    ) -> CurationBatchSummary:
        durations = {
            report.sample_id: report.audio_duration_sec or 0.0 for report in reports
        }
        decision_durations = Counter()
        for sample in curated:
            decision_durations[sample.state.curation_decision.value] += durations.get(
                sample.sample_id, 0.0
            )
        issue_counts = Counter(
            reason for report in reports for reason in report.decision_reasons
        )
        top_issues = [
            IssueCount(code=code, count=count)
            for code, count in sorted(
                issue_counts.items(), key=lambda item: (-item[1], item[0])
            )
        ]
        return CurationBatchSummary(
            total_samples=len(curated),
            pass_count=sum(
                sample.state.curation_decision is CurationDecision.PASS
                for sample in curated
            ),
            review_count=sum(
                sample.state.curation_decision is CurationDecision.REVIEW
                for sample in curated
            ),
            drop_count=sum(
                sample.state.curation_decision is CurationDecision.DROP
                for sample in curated
            ),
            total_audio_duration_sec=sum(durations.values()),
            pass_duration_sec=decision_durations[CurationDecision.PASS.value],
            review_duration_sec=decision_durations[CurationDecision.REVIEW.value],
            drop_duration_sec=decision_durations[CurationDecision.DROP.value],
            top_issue_codes=top_issues,
        )
