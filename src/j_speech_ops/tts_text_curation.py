"""Deterministic Stage 7 Japanese TTS text curation and risk routing."""

from __future__ import annotations

import hashlib
import re
import unicodedata
from collections import Counter
from collections.abc import Iterator
from dataclasses import dataclass
from enum import Enum
from os import PathLike
from pathlib import Path

from pydantic import BaseModel, ConfigDict, Field, JsonValue, model_validator

from .curation import CurationIssue, CurationIssueSeverity, IssueCount
from .japanese_normalization import NormalizationResult
from .japanese_reading import AppliedReadingOverride, ReadingProviderInfo
from .pronunciation_risks import (
    PronunciationRiskMatch,
    PronunciationRiskWatchlist,
)
from .schemas import CurationDecision, RightsStatus
from .text_manifest import iter_text_manifest, write_text_manifest
from .text_preparation import (
    TextPreparationIssue,
    TextPreparationReport,
    TextPreparationStatus,
    iter_text_preparation_reports,
)
from .tts_text import TTSTextSample, TTSTextStage, TTSTextState, transition_text_stage


TTS_TEXT_CURATION_POLICY_VERSION = "tts-text-curation-v1"

_LATIN_PATTERN = re.compile(r"[A-Za-zＡ-Ｚａ-ｚ]+")
_KANJI_PATTERN = re.compile(r"[\u3400-\u4dbf\u4e00-\u9fff\uf900-\ufaff]+")
_ALLOWED_READING_PUNCTUATION = frozenset(
    "、。！？!?，,．.：:；;・ー〜～…‥（）()「」『』【】［］[]〈〉《》〔〕〝〟\"'／/－-＿_"
)
_ALLOWED_READING_SYMBOLS = frozenset("%％+＋=＝@＠&＆#＃￥¥$＄℃°")
_EMOJI_RANGES = (
    (0x1F000, 0x1FAFF),
    (0x2600, 0x27BF),
    (0x2300, 0x23FF),
)
_ALLOWED_FORMAT_CONTROLS = frozenset({"\u200d", "\ufe0e", "\ufe0f"})


class TTSTextCurationPolicy(BaseModel):
    """Versioned routing switches, not a pronunciation-quality standard."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    policy_version: str = Field(
        default=TTS_TEXT_CURATION_POLICY_VERSION, min_length=1
    )
    review_unknown_rights: bool = True
    review_pronunciation_risk_terms: bool = True
    review_emoji: bool = True
    review_unresolved_latin: bool = True
    review_unresolved_digits: bool = True
    review_remaining_kanji: bool = True
    review_unexpected_symbols: bool = True
    drop_invalid_control_chars: bool = True


class ReadingScriptInspection(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    remaining_latin: tuple[str, ...] = ()
    remaining_digits: tuple[str, ...] = ()
    remaining_kanji: tuple[str, ...] = ()
    unexpected_symbols: tuple[str, ...] = ()


class ReadingScriptInspector:
    """Lightweight Unicode inspection; it does not judge pronunciation."""

    @staticmethod
    def _ordered_unique(values: list[str]) -> tuple[str, ...]:
        return tuple(dict.fromkeys(values))

    @staticmethod
    def _is_kana(char: str) -> bool:
        codepoint = ord(char)
        return 0x3040 <= codepoint <= 0x30FF or 0x31F0 <= codepoint <= 0x31FF

    @staticmethod
    def _decimal_runs(text: str) -> tuple[str, ...]:
        runs: list[str] = []
        current = ""
        for char in text:
            if unicodedata.category(char) == "Nd":
                current += char
            elif current:
                runs.append(current)
                current = ""
        if current:
            runs.append(current)
        return ReadingScriptInspector._ordered_unique(runs)

    @classmethod
    def inspect(cls, reading: str) -> ReadingScriptInspection:
        latin = cls._ordered_unique(_LATIN_PATTERN.findall(reading))
        digits = cls._decimal_runs(reading)
        kanji = cls._ordered_unique(_KANJI_PATTERN.findall(reading))
        unexpected: list[str] = []
        for char in reading:
            if (
                char.isspace()
                or cls._is_kana(char)
                or char in _ALLOWED_READING_PUNCTUATION
                or char in _ALLOWED_READING_SYMBOLS
                or _LATIN_PATTERN.fullmatch(char)
                or unicodedata.category(char) == "Nd"
                or _KANJI_PATTERN.fullmatch(char)
            ):
                continue
            unexpected.append(char)
        return ReadingScriptInspection(
            remaining_latin=latin,
            remaining_digits=digits,
            remaining_kanji=kanji,
            unexpected_symbols=cls._ordered_unique(unexpected),
        )


class TTSTextCurationStatus(str, Enum):
    CURATED = "curated"
    SKIPPED = "skipped"
    FAILED = "failed"


class TTSTextCurationReport(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    sample_id: str = Field(min_length=1)
    policy_version: str = Field(min_length=1)
    policy_fingerprint_sha256: str | None = Field(
        default=None, pattern=r"^[0-9a-f]{64}$"
    )
    status: TTSTextCurationStatus
    input_stage: TTSTextStage
    output_stage: TTSTextStage
    decision: CurationDecision
    issues: tuple[CurationIssue, ...] = ()
    decision_reasons: tuple[str, ...] = ()
    raw_text: str
    normalized_text: str | None = None
    reading_kana: str | None = None
    preparation_policy_version: str | None = None
    preparation_status: TextPreparationStatus | None = None
    preparation_warnings: tuple[TextPreparationIssue, ...] = ()
    preparation_errors: tuple[TextPreparationIssue, ...] = ()
    normalization: NormalizationResult | None = None
    reading_provider: ReadingProviderInfo | None = None
    reading_warnings: tuple[str, ...] = ()
    reading_errors: tuple[str, ...] = ()
    applied_overrides: tuple[AppliedReadingOverride, ...] = ()
    override_fingerprint_sha256: str | None = Field(
        default=None, pattern=r"^[0-9a-f]{64}$"
    )
    reading_script: ReadingScriptInspection | None = None
    risk_terms: tuple[PronunciationRiskMatch, ...] = ()
    override_resolved_risk_terms: tuple[PronunciationRiskMatch, ...] = ()
    watchlist_version: str
    watchlist_source: str
    watchlist_fingerprint_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    skipped_existing: bool = False

    @model_validator(mode="after")
    def status_and_decision_must_agree(self) -> "TTSTextCurationReport":
        if self.status is TTSTextCurationStatus.FAILED:
            if self.decision is not CurationDecision.UNDECIDED:
                raise ValueError("failed curation reports must remain undecided")
        elif self.decision is CurationDecision.UNDECIDED:
            raise ValueError("curated/skipped reports require a final decision")
        return self


class TTSTextCurationBatchSummary(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    total_samples: int = Field(ge=0)
    pass_count: int = Field(ge=0)
    review_count: int = Field(ge=0)
    drop_count: int = Field(ge=0)
    failed_count: int = Field(ge=0)
    top_issue_codes: tuple[IssueCount, ...] = ()
    override_applied_sample_count: int = Field(ge=0)
    pronunciation_risk_sample_count: int = Field(ge=0)
    watchlist_version: str
    watchlist_fingerprint_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")


@dataclass(frozen=True)
class TTSTextCurationBatchOutputs:
    curated_manifest: Path
    synthesis_ready_manifest: Path
    report_file: Path
    summary: TTSTextCurationBatchSummary


def _issue(
    code: str,
    severity: CurationIssueSeverity,
    message: str,
    **evidence: JsonValue,
) -> CurationIssue:
    return CurationIssue(
        code=code, severity=severity, message=message, evidence=evidence
    )


def _policy_fingerprint(policy: TTSTextCurationPolicy) -> str:
    return hashlib.sha256(policy.model_dump_json().encode("utf-8")).hexdigest()


def _decision_for(issues: list[CurationIssue]) -> CurationDecision:
    if any(issue.severity is CurationIssueSeverity.CRITICAL for issue in issues):
        return CurationDecision.DROP
    if any(issue.severity is CurationIssueSeverity.WARNING for issue in issues):
        return CurationDecision.REVIEW
    return CurationDecision.PASS


def _is_emoji(char: str) -> bool:
    codepoint = ord(char)
    return any(start <= codepoint <= end for start, end in _EMOJI_RANGES)


def _emoji_surfaces(*texts: str | None) -> tuple[str, ...]:
    return tuple(
        dict.fromkeys(
            char for text in texts if text for char in text if _is_emoji(char)
        )
    )


def _invalid_controls(*texts: str | None) -> tuple[str, ...]:
    found: list[str] = []
    for text in texts:
        if text is None:
            continue
        for char in text:
            category = unicodedata.category(char)
            if category not in {"Cc", "Cf"}:
                continue
            if char in {"\t", "\n", "\r"} or char in _ALLOWED_FORMAT_CONTROLS:
                continue
            found.append(f"U+{ord(char):04X}")
    return tuple(dict.fromkeys(found))


def write_tts_text_curation_reports(
    path: str | PathLike[str], reports: list[TTSTextCurationReport]
) -> int:
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    with destination.open("w", encoding="utf-8", newline="\n") as handle:
        for report in reports:
            handle.write(report.model_dump_json())
            handle.write("\n")
    return len(reports)


def iter_tts_text_curation_reports(
    path: str | PathLike[str],
) -> Iterator[TTSTextCurationReport]:
    report_path = Path(path)
    sample_ids: set[str] = set()
    with report_path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            try:
                report = TTSTextCurationReport.model_validate_json(line)
            except ValueError as exc:
                raise ValueError(
                    f"{report_path}:{line_number}: invalid TTS text curation report: {exc}"
                ) from exc
            if report.sample_id in sample_ids:
                raise ValueError(
                    f"{report_path}:{line_number}: duplicate curation report sample_id "
                    f"{report.sample_id}"
                )
            sample_ids.add(report.sample_id)
            yield report


class TTSTextCurationPipeline:
    def __init__(
        self,
        *,
        output_root: Path,
        watchlist: PronunciationRiskWatchlist,
        policy: TTSTextCurationPolicy | None = None,
    ) -> None:
        self.output_root = Path(output_root).resolve()
        self.watchlist = watchlist
        self.policy = policy or TTSTextCurationPolicy()
        self.policy_fingerprint_sha256 = _policy_fingerprint(self.policy)
        self.manifest_dir = self.output_root / "manifests"
        self.report_dir = self.output_root / "reports"

    def run(
        self,
        input_manifest: Path,
        preparation_report: Path,
        *,
        force: bool = False,
    ) -> TTSTextCurationBatchOutputs:
        samples = list(iter_text_manifest(input_manifest))
        preparation_reports = list(iter_text_preparation_reports(preparation_report))
        preparation_by_id: dict[str, TextPreparationReport] = {}
        for report in preparation_reports:
            if report.sample_id in preparation_by_id:
                raise ValueError(
                    f"preparation report contains duplicate sample_id {report.sample_id}"
                )
            preparation_by_id[report.sample_id] = report

        manifest_ids = {sample.sample_id for sample in samples}
        preparation_ids = set(preparation_by_id)
        if manifest_ids and preparation_ids and manifest_ids.isdisjoint(preparation_ids):
            raise ValueError(
                "manifest and preparation report have no matching sample_id values"
            )

        self.manifest_dir.mkdir(parents=True, exist_ok=True)
        self.report_dir.mkdir(parents=True, exist_ok=True)
        curated_manifest = self.manifest_dir / "curated_tts_text_samples.jsonl"
        synthesis_ready_manifest = (
            self.manifest_dir / "synthesis_ready_text_samples.jsonl"
        )
        report_file = self.report_dir / "tts_text_curation_report.jsonl"
        previous_reports = self._load_previous_reports(report_file) if not force else {}

        curated: list[TTSTextSample] = []
        reports: list[TTSTextCurationReport] = []
        for sample in samples:
            preparation = preparation_by_id.get(sample.sample_id)
            if preparation is None:
                reports.append(
                    self._failed_report(
                        sample,
                        "missing_preparation_report",
                        "no Stage 6 companion report exists for this sample_id",
                    )
                )
                continue
            self._validate_companion(sample, preparation)

            if sample.state.stage is TTSTextStage.CURATED and not force:
                if sample.state.curation_decision is CurationDecision.UNDECIDED:
                    reports.append(
                        self._failed_report(
                            sample,
                            "invalid_curated_state",
                            "CURATED sample has an undecided curation decision",
                            preparation,
                        )
                    )
                    continue
                curated.append(sample.model_copy(deep=True))
                previous = previous_reports.get(sample.sample_id)
                if (
                    previous is not None
                    and previous.decision is sample.state.curation_decision
                ):
                    reports.append(
                        previous.model_copy(
                            update={
                                "status": TTSTextCurationStatus.SKIPPED,
                                "input_stage": TTSTextStage.CURATED,
                                "output_stage": TTSTextStage.CURATED,
                                "skipped_existing": True,
                            }
                        )
                    )
                else:
                    reports.append(self._skipped_report(sample, preparation))
                continue

            if sample.state.stage not in {
                TTSTextStage.READING_PREPARED,
                TTSTextStage.CURATED,
            }:
                reports.append(
                    self._failed_report(
                        sample,
                        "unexpected_prepared_state",
                        "Stage 7 requires a READING_PREPARED or CURATED sample",
                        preparation,
                    )
                )
                continue
            if sample.text.phonemes is not None:
                reports.append(
                    self._failed_report(
                        sample,
                        "phonemes_already_set",
                        "Stage 7 requires phonemes to remain unset",
                        preparation,
                    )
                )
                continue
            if (
                sample.state.stage is TTSTextStage.READING_PREPARED
                and sample.state.curation_decision is not CurationDecision.UNDECIDED
            ):
                reports.append(
                    self._failed_report(
                        sample,
                        "premature_curation_decision",
                        "READING_PREPARED sample already has a curation decision",
                        preparation,
                    )
                )
                continue

            updated, report = self._curate(sample, preparation)
            curated.append(updated)
            reports.append(report)

        synthesis_ready = [
            sample
            for sample in curated
            if sample.state.curation_decision is CurationDecision.PASS
        ]
        write_text_manifest(curated_manifest, curated)
        write_text_manifest(synthesis_ready_manifest, synthesis_ready)
        write_tts_text_curation_reports(report_file, reports)
        summary = self._summarize(samples, curated, reports)
        return TTSTextCurationBatchOutputs(
            curated_manifest=curated_manifest,
            synthesis_ready_manifest=synthesis_ready_manifest,
            report_file=report_file,
            summary=summary,
        )

    @staticmethod
    def _load_previous_reports(path: Path) -> dict[str, TTSTextCurationReport]:
        if not path.exists():
            return {}
        return {
            report.sample_id: report for report in iter_tts_text_curation_reports(path)
        }

    @staticmethod
    def _validate_companion(
        sample: TTSTextSample, report: TextPreparationReport
    ) -> None:
        if report.sample_id != sample.sample_id:
            raise ValueError("internal preparation report join mismatch")
        if sample.state.stage in {
            TTSTextStage.READING_PREPARED,
            TTSTextStage.CURATED,
        } and (
            report.output_stage is not TTSTextStage.READING_PREPARED
            or report.status
            not in {TextPreparationStatus.READING_PREPARED, TextPreparationStatus.SKIPPED}
        ):
            raise ValueError(
                f"sample {sample.sample_id} claims prepared state but its companion "
                "report does not record a prepared output"
            )
        if report.normalization is not None:
            if report.normalization.raw_text != sample.text.raw_text:
                raise ValueError(
                    f"sample {sample.sample_id} raw_text contradicts preparation report"
                )
            if (
                sample.text.normalized_text is not None
                and report.normalization.normalized_text
                != sample.text.normalized_text
            ):
                raise ValueError(
                    f"sample {sample.sample_id} normalized_text contradicts preparation report"
                )
        if report.reading is not None:
            if report.reading.sample_id != sample.sample_id:
                raise ValueError(
                    f"sample {sample.sample_id} is joined to a reading for another sample"
                )
            if (
                sample.text.normalized_text is not None
                and report.reading.input_text != sample.text.normalized_text
            ):
                raise ValueError(
                    f"sample {sample.sample_id} reading input contradicts normalized_text"
                )
            if (
                sample.text.reading_kana is not None
                and report.reading.reading_kana != sample.text.reading_kana
            ):
                raise ValueError(
                    f"sample {sample.sample_id} reading_kana contradicts preparation report"
                )

    def _curate(
        self, sample: TTSTextSample, preparation: TextPreparationReport
    ) -> tuple[TTSTextSample, TTSTextCurationReport]:
        issues: list[CurationIssue] = []
        normalized = sample.text.normalized_text
        reading = sample.text.reading_kana

        if normalized is None or not normalized.strip():
            issues.append(
                _issue(
                    "missing_normalized_text",
                    CurationIssueSeverity.CRITICAL,
                    "READING_PREPARED sample has no usable normalized_text",
                )
            )
        if reading is None or not reading.strip():
            issues.append(
                _issue(
                    "missing_reading",
                    CurationIssueSeverity.CRITICAL,
                    "READING_PREPARED sample has no usable reading_kana",
                )
            )

        if sample.rights_status is RightsStatus.RESTRICTED:
            issues.append(
                _issue(
                    "rights_restricted",
                    CurationIssueSeverity.CRITICAL,
                    "source rights prohibit synthesis-ready routing",
                    rights_status=sample.rights_status.value,
                )
            )
        elif (
            sample.rights_status is RightsStatus.UNKNOWN
            and self.policy.review_unknown_rights
        ):
            issues.append(
                _issue(
                    "rights_unknown",
                    CurationIssueSeverity.WARNING,
                    "source rights have not been cleared",
                    rights_status=sample.rights_status.value,
                )
            )

        invalid_controls = _invalid_controls(
            sample.text.raw_text, normalized, reading
        )
        if invalid_controls and self.policy.drop_invalid_control_chars:
            issues.append(
                _issue(
                    "invalid_control_character",
                    CurationIssueSeverity.CRITICAL,
                    "text contains a non-whitespace control or format character",
                    codepoints=list(invalid_controls),
                )
            )

        inspection = ReadingScriptInspector.inspect(reading) if reading else None
        if inspection is not None:
            if inspection.remaining_latin and self.policy.review_unresolved_latin:
                issues.append(
                    _issue(
                        "unresolved_reading_token",
                        CurationIssueSeverity.WARNING,
                        "reading_kana still contains Latin text",
                        remaining_latin=list(inspection.remaining_latin),
                    )
                )
            if inspection.remaining_digits and self.policy.review_unresolved_digits:
                issues.append(
                    _issue(
                        "unresolved_numeric_reading",
                        CurationIssueSeverity.WARNING,
                        "reading_kana still contains decimal digits",
                        remaining_digits=list(inspection.remaining_digits),
                    )
                )
            if inspection.remaining_kanji and self.policy.review_remaining_kanji:
                issues.append(
                    _issue(
                        "remaining_kanji_in_reading",
                        CurationIssueSeverity.WARNING,
                        "reading_kana still contains Kanji",
                        remaining_kanji=list(inspection.remaining_kanji),
                    )
                )
            if (
                inspection.unexpected_symbols
                and self.policy.review_unexpected_symbols
            ):
                issues.append(
                    _issue(
                        "unexpected_reading_symbol",
                        CurationIssueSeverity.WARNING,
                        "reading_kana contains characters outside the allowed baseline",
                        unexpected_symbols=list(inspection.unexpected_symbols),
                    )
                )

        emoji = _emoji_surfaces(sample.text.raw_text, normalized)
        if emoji and self.policy.review_emoji:
            issues.append(
                _issue(
                    "emoji_requires_review",
                    CurationIssueSeverity.WARNING,
                    "emoji spoken/omission behavior requires product review",
                    emoji=list(emoji),
                )
            )

        applied = preparation.reading.applied_overrides if preparation.reading else ()
        risk_terms, resolved_risk_terms = self.watchlist.match(
            raw_text=sample.text.raw_text,
            normalized_text=normalized,
            applied_overrides=applied,
        )
        if risk_terms and self.policy.review_pronunciation_risk_terms:
            issues.append(
                _issue(
                    "pronunciation_risk_term",
                    CurationIssueSeverity.WARNING,
                    "text contains a configured pronunciation-risk surface",
                    risk_terms=[item.model_dump(mode="json") for item in risk_terms],
                )
            )

        if preparation.normalization is not None and preparation.normalization.changed:
            issues.append(
                _issue(
                    "normalization_applied",
                    CurationIssueSeverity.INFO,
                    "Stage 6 normalization changed the written form",
                    transformations=list(preparation.normalization.transformations),
                )
            )
        if applied:
            issues.append(
                _issue(
                    "override_applied",
                    CurationIssueSeverity.INFO,
                    "Stage 6 applied confirmed pronunciation overrides",
                    surfaces=[item.surface for item in applied],
                    occurrence_count=sum(item.occurrences for item in applied),
                )
            )

        decision = _decision_for(issues)
        decision_reasons = tuple(
            dict.fromkeys(
                issue.code
                for issue in issues
                if issue.severity is not CurationIssueSeverity.INFO
            )
        )
        transitioned = transition_text_stage(sample, TTSTextStage.CURATED)
        updated = transitioned.model_copy(
            update={
                "state": TTSTextState(
                    stage=TTSTextStage.CURATED,
                    curation_decision=decision,
                    decision_reasons=decision_reasons,
                )
            },
            deep=True,
        )
        report = self._report(
            sample=sample,
            preparation=preparation,
            status=TTSTextCurationStatus.CURATED,
            output_stage=TTSTextStage.CURATED,
            decision=decision,
            issues=tuple(issues),
            decision_reasons=decision_reasons,
            reading_script=inspection,
            risk_terms=risk_terms,
            resolved_risk_terms=resolved_risk_terms,
        )
        return updated, report

    def _report(
        self,
        *,
        sample: TTSTextSample,
        preparation: TextPreparationReport,
        status: TTSTextCurationStatus,
        output_stage: TTSTextStage,
        decision: CurationDecision,
        issues: tuple[CurationIssue, ...] = (),
        decision_reasons: tuple[str, ...] = (),
        reading_script: ReadingScriptInspection | None = None,
        risk_terms: tuple[PronunciationRiskMatch, ...] = (),
        resolved_risk_terms: tuple[PronunciationRiskMatch, ...] = (),
        skipped_existing: bool = False,
        policy_version: str | None = None,
        policy_fingerprint: str | None = None,
    ) -> TTSTextCurationReport:
        reading_result = preparation.reading
        return TTSTextCurationReport(
            sample_id=sample.sample_id,
            policy_version=policy_version or self.policy.policy_version,
            policy_fingerprint_sha256=(
                self.policy_fingerprint_sha256
                if policy_fingerprint is None and policy_version is None
                else policy_fingerprint
            ),
            status=status,
            input_stage=sample.state.stage,
            output_stage=output_stage,
            decision=decision,
            issues=issues,
            decision_reasons=decision_reasons,
            raw_text=sample.text.raw_text,
            normalized_text=sample.text.normalized_text,
            reading_kana=sample.text.reading_kana,
            preparation_policy_version=preparation.policy_version,
            preparation_status=preparation.status,
            preparation_warnings=preparation.warnings,
            preparation_errors=preparation.errors,
            normalization=preparation.normalization,
            reading_provider=preparation.dependencies.reading_provider,
            reading_warnings=(reading_result.warnings if reading_result else ()),
            reading_errors=(reading_result.errors if reading_result else ()),
            applied_overrides=(
                reading_result.applied_overrides if reading_result else ()
            ),
            override_fingerprint_sha256=(
                reading_result.override_fingerprint_sha256 if reading_result else None
            ),
            reading_script=reading_script,
            risk_terms=risk_terms,
            override_resolved_risk_terms=resolved_risk_terms,
            watchlist_version=self.watchlist.config.version,
            watchlist_source=self.watchlist.config.source,
            watchlist_fingerprint_sha256=self.watchlist.fingerprint_sha256,
            skipped_existing=skipped_existing,
        )

    def _failed_report(
        self,
        sample: TTSTextSample,
        code: str,
        message: str,
        preparation: TextPreparationReport | None = None,
    ) -> TTSTextCurationReport:
        issue = _issue(code, CurationIssueSeverity.CRITICAL, message)
        if preparation is None:
            return TTSTextCurationReport(
                sample_id=sample.sample_id,
                policy_version=self.policy.policy_version,
                policy_fingerprint_sha256=self.policy_fingerprint_sha256,
                status=TTSTextCurationStatus.FAILED,
                input_stage=sample.state.stage,
                output_stage=sample.state.stage,
                decision=CurationDecision.UNDECIDED,
                issues=(issue,),
                decision_reasons=(code,),
                raw_text=sample.text.raw_text,
                normalized_text=sample.text.normalized_text,
                reading_kana=sample.text.reading_kana,
                watchlist_version=self.watchlist.config.version,
                watchlist_source=self.watchlist.config.source,
                watchlist_fingerprint_sha256=self.watchlist.fingerprint_sha256,
            )
        return self._report(
            sample=sample,
            preparation=preparation,
            status=TTSTextCurationStatus.FAILED,
            output_stage=sample.state.stage,
            decision=CurationDecision.UNDECIDED,
            issues=(issue,),
            decision_reasons=(code,),
        )

    def _skipped_report(
        self, sample: TTSTextSample, preparation: TextPreparationReport
    ) -> TTSTextCurationReport:
        return self._report(
            sample=sample,
            preparation=preparation,
            status=TTSTextCurationStatus.SKIPPED,
            output_stage=TTSTextStage.CURATED,
            decision=sample.state.curation_decision,
            decision_reasons=tuple(sample.state.decision_reasons),
            skipped_existing=True,
            policy_version="existing-curation-policy-unknown",
            policy_fingerprint=None,
        )

    def _summarize(
        self,
        input_samples: list[TTSTextSample],
        curated: list[TTSTextSample],
        reports: list[TTSTextCurationReport],
    ) -> TTSTextCurationBatchSummary:
        decision_counts = Counter(
            sample.state.curation_decision.value for sample in curated
        )
        issue_counts = Counter(issue.code for report in reports for issue in report.issues)
        top_issues = tuple(
            IssueCount(code=code, count=count)
            for code, count in sorted(
                issue_counts.items(), key=lambda item: (-item[1], item[0])
            )
        )
        return TTSTextCurationBatchSummary(
            total_samples=len(input_samples),
            pass_count=decision_counts[CurationDecision.PASS.value],
            review_count=decision_counts[CurationDecision.REVIEW.value],
            drop_count=decision_counts[CurationDecision.DROP.value],
            failed_count=sum(
                report.status is TTSTextCurationStatus.FAILED for report in reports
            ),
            top_issue_codes=top_issues,
            override_applied_sample_count=sum(
                bool(report.applied_overrides) for report in reports
            ),
            pronunciation_risk_sample_count=sum(
                bool(report.risk_terms) for report in reports
            ),
            watchlist_version=self.watchlist.config.version,
            watchlist_fingerprint_sha256=self.watchlist.fingerprint_sha256,
        )
