from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest
from pydantic import ValidationError

from j_speech_ops.japanese_normalization import NormalizationResult
from j_speech_ops.japanese_reading import (
    AppliedReadingOverride,
    ReadingProviderInfo,
    ReadingResult,
)
from j_speech_ops.pronunciation_risks import PronunciationRiskWatchlist
from j_speech_ops.schemas import CurationDecision, RightsStatus
from j_speech_ops.text_manifest import iter_text_manifest, write_text_manifest
from j_speech_ops.text_preparation import (
    TextPreparationDependencies,
    TextPreparationIssue,
    TextPreparationReport,
    TextPreparationStatus,
    write_text_preparation_reports,
)
from j_speech_ops.tts_text import (
    TTSTextContent,
    TTSTextSample,
    TTSTextSource,
    TTSTextStage,
)
from j_speech_ops.tts_text_curation import (
    TTSTextCurationPipeline,
    TTSTextCurationPolicy,
    TTSTextCurationStatus,
    iter_tts_text_curation_reports,
)


PROVIDER = ReadingProviderInfo(
    provider="synthetic-stage6-provider",
    provider_version="1",
    package_name="synthetic",
    package_version="1",
    dictionary_identity="synthetic test dictionary",
    settings={"kana": True},
)
OVERRIDE_FINGERPRINT = "a" * 64


def make_sample(
    sample_id: str,
    raw_text: str = "本日はよろしくお願いいたします。",
    *,
    normalized_text: str | None = "本日はよろしくお願いいたします。",
    reading_kana: str | None = "ホンジツワヨロシクオネガイイタシマス。",
    rights: RightsStatus = RightsStatus.CLEARED,
    stage: TTSTextStage = TTSTextStage.READING_PREPARED,
    decision: CurationDecision = CurationDecision.UNDECIDED,
    reasons: tuple[str, ...] = (),
    phonemes: tuple[str, ...] | None = None,
) -> TTSTextSample:
    return TTSTextSample(
        sample_id=sample_id,
        dataset_id="stage7-test",
        language="ja",
        locale="ja-JP",
        text=TTSTextContent(
            raw_text=raw_text,
            normalized_text=normalized_text,
            reading_kana=reading_kana,
            phonemes=phonemes,
        ),
        source=TTSTextSource(
            source_type="file", source_name="stage7.txt", source_id=sample_id
        ),
        rights_status=rights,
        intended_use="tts_production",
        domain="hospitality",
        state={
            "stage": stage.value,
            "curation_decision": decision.value,
            "decision_reasons": reasons,
        },
        metadata={"preserve": True},
    )


def make_preparation_report(
    sample: TTSTextSample,
    *,
    applied_overrides: tuple[AppliedReadingOverride, ...] = (),
    include_reading: bool = True,
) -> TextPreparationReport:
    normalized = sample.text.normalized_text
    reading = sample.text.reading_kana
    normalization = None
    if normalized is not None:
        normalization = NormalizationResult(
            policy_version="japanese-text-prep-v1.1",
            raw_text=sample.text.raw_text,
            normalized_text=normalized,
            changed=sample.text.raw_text != normalized,
            transformations=(
                ("jaconv_unicode_width_normalization",)
                if sample.text.raw_text != normalized
                else ()
            ),
            jaconv_version="0.5.0",
        )
    reading_result = None
    if include_reading and normalized is not None and reading is not None:
        reading_result = ReadingResult(
            sample_id=sample.sample_id,
            provider=PROVIDER.provider,
            provider_version=PROVIDER.provider_version,
            input_text=normalized,
            baseline_reading_kana=reading,
            reading_kana=reading,
            override_count=sum(item.occurrences for item in applied_overrides),
            applied_overrides=applied_overrides,
            override_fingerprint_sha256=OVERRIDE_FINGERPRINT,
        )
    return TextPreparationReport(
        sample_id=sample.sample_id,
        policy_version="japanese-text-prep-v1.1",
        status=(
            TextPreparationStatus.READING_PREPARED
            if reading_result is not None
            else TextPreparationStatus.SKIPPED
        ),
        input_stage=TTSTextStage.RAW,
        output_stage=TTSTextStage.READING_PREPARED,
        normalization=normalization,
        reading=reading_result,
        dependencies=TextPreparationDependencies(
            jaconv_version="0.5.0", reading_provider=PROVIDER
        ),
        skipped_existing=reading_result is None,
    )


def write_watchlist(path: Path, surfaces: list[str] | None = None) -> Path:
    entries = [
        {
            "surface": surface,
            "category": "proper_name",
            "reason": "multiple_readings_possible",
            "note": "test entry",
        }
        for surface in (surfaces or [])
    ]
    path.write_text(
        json.dumps(
            {"version": "test-watchlist-v1", "source": "tests", "entries": entries},
            ensure_ascii=False,
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    return path


def run_batch(
    tmp_path: Path,
    samples: list[TTSTextSample],
    *,
    reports: list[TextPreparationReport] | None = None,
    risk_surfaces: list[str] | None = None,
    force: bool = False,
):
    manifest = tmp_path / "reading_prepared_text_samples.jsonl"
    preparation = tmp_path / "text_preparation_report.jsonl"
    watchlist_path = tmp_path / "japanese_pronunciation_risks.json"
    write_text_manifest(manifest, samples)
    write_text_preparation_reports(
        preparation,
        reports
        if reports is not None
        else [make_preparation_report(sample) for sample in samples],
    )
    write_watchlist(watchlist_path, risk_surfaces)
    pipeline = TTSTextCurationPipeline(
        output_root=tmp_path / "data",
        watchlist=PronunciationRiskWatchlist.load(watchlist_path),
    )
    return pipeline.run(manifest, preparation, force=force)


def curated_decision_and_codes(outputs):
    sample = list(iter_text_manifest(outputs.curated_manifest))[0]
    report = list(iter_tts_text_curation_reports(outputs.report_file))[0]
    return sample.state.curation_decision, {issue.code for issue in report.issues}


def test_clean_japanese_passes_and_only_state_changes(tmp_path: Path) -> None:
    original = make_sample("clean")
    protected = original.model_dump(exclude={"state"})
    outputs = run_batch(tmp_path, [original])
    curated = list(iter_text_manifest(outputs.curated_manifest))[0]
    report = list(iter_tts_text_curation_reports(outputs.report_file))[0]

    assert curated.state.stage is TTSTextStage.CURATED
    assert curated.state.curation_decision is CurationDecision.PASS
    assert curated.state.decision_reasons == ()
    assert curated.model_dump(exclude={"state"}) == protected
    assert curated.text.phonemes is None
    assert report.decision is CurationDecision.PASS
    assert report.policy_version == "tts-text-curation-v1"
    assert len(report.watchlist_fingerprint_sha256) == 64


def test_stage61_amount_and_resolved_latin_source_pass(tmp_path: Path) -> None:
    amount = make_sample(
        "amount",
        "料金は３,５００円です。",
        normalized_text="料金は3500円です。",
        reading_kana="リョーキンワサンゼンゴヒャクエンデス。",
    )
    latin_source = make_sample(
        "latin-source",
        "JRをご利用ください。",
        normalized_text="JRをご利用ください。",
        reading_kana="ジェイアールヲゴリヨークダサイ。",
    )
    malformed_grouping = make_sample(
        "malformed-grouping",
        "12,34円",
        normalized_text="12,34円",
        reading_kana="ジューニ，サンジューヨエン",
    )
    outputs = run_batch(tmp_path, [amount, latin_source, malformed_grouping])
    curated = list(iter_text_manifest(outputs.curated_manifest))
    reports = list(iter_tts_text_curation_reports(outputs.report_file))
    assert [sample.state.curation_decision for sample in curated] == [
        CurationDecision.PASS,
        CurationDecision.PASS,
        CurationDecision.PASS,
    ]
    assert "normalization_applied" in {issue.code for issue in reports[0].issues}
    assert reports[0].decision_reasons == ()


@pytest.mark.parametrize(
    ("sample_id", "rights", "expected", "code"),
    [
        ("restricted", RightsStatus.RESTRICTED, CurationDecision.DROP, "rights_restricted"),
        ("unknown", RightsStatus.UNKNOWN, CurationDecision.REVIEW, "rights_unknown"),
    ],
)
def test_rights_routing(
    tmp_path: Path,
    sample_id: str,
    rights: RightsStatus,
    expected: CurationDecision,
    code: str,
) -> None:
    decision, codes = curated_decision_and_codes(
        run_batch(tmp_path, [make_sample(sample_id, rights=rights)])
    )
    assert decision is expected
    assert code in codes


@pytest.mark.parametrize(
    ("sample_id", "normalized", "reading", "code"),
    [
        ("missing-normalized", None, "テスト", "missing_normalized_text"),
        ("missing-reading", "テスト", None, "missing_reading"),
    ],
)
def test_missing_prepared_data_drops(
    tmp_path: Path,
    sample_id: str,
    normalized: str | None,
    reading: str | None,
    code: str,
) -> None:
    sample = make_sample(
        sample_id,
        "テスト",
        normalized_text=normalized,
        reading_kana=reading,
    )
    report = make_preparation_report(sample, include_reading=False)
    decision, codes = curated_decision_and_codes(
        run_batch(tmp_path, [sample], reports=[report])
    )
    assert decision is CurationDecision.DROP
    assert code in codes


@pytest.mark.parametrize(
    ("sample_id", "reading", "code", "inspection_field"),
    [
        ("latin", "ホテルXYZデス", "unresolved_reading_token", "remaining_latin"),
        ("digits", "2026ネンデス", "unresolved_numeric_reading", "remaining_digits"),
        ("kanji", "東京デス", "remaining_kanji_in_reading", "remaining_kanji"),
    ],
)
def test_remaining_reading_scripts_review(
    tmp_path: Path,
    sample_id: str,
    reading: str,
    code: str,
    inspection_field: str,
) -> None:
    sample = make_sample(
        sample_id, "入力", normalized_text="入力", reading_kana=reading
    )
    outputs = run_batch(tmp_path, [sample])
    decision, codes = curated_decision_and_codes(outputs)
    report = list(iter_tts_text_curation_reports(outputs.report_file))[0]
    assert decision is CurationDecision.REVIEW
    assert code in codes
    assert getattr(report.reading_script, inspection_field)


def test_watchlist_risk_reports_occurrences_and_routes_review(tmp_path: Path) -> None:
    sample = make_sample(
        "risk",
        "佐伯様と佐伯様です。",
        normalized_text="佐伯様と佐伯様です。",
        reading_kana="サエキサマトサエキサマデス。",
    )
    outputs = run_batch(tmp_path, [sample], risk_surfaces=["佐伯"])
    decision, codes = curated_decision_and_codes(outputs)
    report = list(iter_tts_text_curation_reports(outputs.report_file))[0]
    assert decision is CurationDecision.REVIEW
    assert "pronunciation_risk_term" in codes
    assert report.risk_terms[0].surface == "佐伯"
    assert report.risk_terms[0].occurrence_count == 2


def test_explicit_applied_override_suppresses_matching_watchlist_risk(
    tmp_path: Path,
) -> None:
    sample = make_sample(
        "resolved-risk",
        "佐伯様と佐伯様です。",
        normalized_text="佐伯様と佐伯様です。",
        reading_kana="サイキサマトサイキサマデス。",
    )
    applied = AppliedReadingOverride(
        surface="佐伯",
        reading_kana="サイキ",
        source="human-confirmed test override",
        occurrences=2,
    )
    preparation = make_preparation_report(sample, applied_overrides=(applied,))
    preparation = preparation.model_copy(
        update={
            "warnings": (
                TextPreparationIssue(
                    code="stage6_provenance_note", message="synthetic warning"
                ),
            ),
            "reading": preparation.reading.model_copy(
                update={"warnings": ("synthetic provider warning",)}
            ),
        }
    )
    outputs = run_batch(
        tmp_path,
        [sample],
        reports=[preparation],
        risk_surfaces=["佐伯"],
    )
    decision, codes = curated_decision_and_codes(outputs)
    report = list(iter_tts_text_curation_reports(outputs.report_file))[0]
    assert decision is CurationDecision.PASS
    assert "pronunciation_risk_term" not in codes
    assert "override_applied" in codes
    assert report.risk_terms == ()
    assert report.override_resolved_risk_terms[0].occurrence_count == 2
    assert report.applied_overrides == (applied,)
    assert report.override_fingerprint_sha256 == OVERRIDE_FINGERPRINT
    assert report.preparation_warnings[0].code == "stage6_provenance_note"
    assert report.reading_warnings == ("synthetic provider warning",)


def test_partial_override_proof_does_not_suppress_all_occurrences(
    tmp_path: Path,
) -> None:
    sample = make_sample(
        "partial",
        "佐伯佐伯",
        normalized_text="佐伯佐伯",
        reading_kana="サエキサエキ",
    )
    applied = AppliedReadingOverride(
        surface="佐伯", reading_kana="サエキ", source="partial", occurrences=1
    )
    outputs = run_batch(
        tmp_path,
        [sample],
        reports=[make_preparation_report(sample, applied_overrides=(applied,))],
        risk_surfaces=["佐伯"],
    )
    assert curated_decision_and_codes(outputs)[0] is CurationDecision.REVIEW


def test_emoji_reviews_and_invalid_control_drops(tmp_path: Path) -> None:
    emoji = make_sample(
        "emoji",
        "ありがとうございます😊",
        normalized_text="ありがとうございます😊",
        reading_kana="アリガトーゴザイマス",
    )
    control = make_sample(
        "control",
        "テスト\x00です",
        normalized_text="テスト\x00です",
        reading_kana="テストデス",
    )
    outputs = run_batch(tmp_path, [emoji, control])
    samples = list(iter_text_manifest(outputs.curated_manifest))
    reports = list(iter_tts_text_curation_reports(outputs.report_file))
    assert [sample.state.curation_decision for sample in samples] == [
        CurationDecision.REVIEW,
        CurationDecision.DROP,
    ]
    assert "emoji_requires_review" in reports[0].decision_reasons
    assert "invalid_control_character" in reports[1].decision_reasons


def test_drop_priority_over_pronunciation_review(tmp_path: Path) -> None:
    sample = make_sample(
        "priority",
        "佐伯様",
        normalized_text="佐伯様",
        reading_kana="サエキサマ",
        rights=RightsStatus.RESTRICTED,
    )
    outputs = run_batch(tmp_path, [sample], risk_surfaces=["佐伯"])
    decision, codes = curated_decision_and_codes(outputs)
    assert decision is CurationDecision.DROP
    assert {"rights_restricted", "pronunciation_risk_term"} <= codes


def test_manifests_are_reloadable_pass_only_and_deterministic(tmp_path: Path) -> None:
    samples = [
        make_sample("pass"),
        make_sample(
            "review",
            "ありがとうございます😊",
            normalized_text="ありがとうございます😊",
            reading_kana="アリガトーゴザイマス",
        ),
        make_sample("drop", rights=RightsStatus.RESTRICTED),
    ]
    first = run_batch(tmp_path, samples)
    first_bytes = {
        path.name: path.read_bytes()
        for path in (
            first.curated_manifest,
            first.synthesis_ready_manifest,
            first.report_file,
        )
    }
    second = run_batch(tmp_path, samples)
    second_bytes = {
        path.name: path.read_bytes()
        for path in (
            second.curated_manifest,
            second.synthesis_ready_manifest,
            second.report_file,
        )
    }
    curated = list(iter_text_manifest(second.curated_manifest))
    ready = list(iter_text_manifest(second.synthesis_ready_manifest))
    assert first_bytes == second_bytes
    assert [sample.sample_id for sample in curated] == ["pass", "review", "drop"]
    assert [sample.sample_id for sample in ready] == ["pass"]
    assert second.summary.total_samples == 3
    assert second.summary.pass_count == 1
    assert second.summary.review_count == 1
    assert second.summary.drop_count == 1
    assert second.summary.failed_count == 0


def test_existing_curated_skips_and_force_recurates(tmp_path: Path) -> None:
    prepared = make_sample("existing")
    preparation = make_preparation_report(prepared)
    first = run_batch(tmp_path, [prepared], reports=[preparation])
    curated = list(iter_text_manifest(first.curated_manifest))[0]

    skipped = run_batch(tmp_path, [curated], reports=[preparation])
    skipped_report = list(iter_tts_text_curation_reports(skipped.report_file))[0]
    forced = run_batch(tmp_path, [curated], reports=[preparation], force=True)
    forced_sample = list(iter_text_manifest(forced.curated_manifest))[0]

    assert skipped_report.status is TTSTextCurationStatus.SKIPPED
    assert skipped_report.skipped_existing is True
    assert forced_sample.state.stage is TTSTextStage.CURATED
    assert forced_sample.text == curated.text


def test_missing_companion_report_isolated_and_batch_continues(tmp_path: Path) -> None:
    missing = make_sample("missing-report")
    valid = make_sample("valid")
    outputs = run_batch(
        tmp_path,
        [missing, valid],
        reports=[make_preparation_report(valid)],
    )
    curated = list(iter_text_manifest(outputs.curated_manifest))
    reports = list(iter_tts_text_curation_reports(outputs.report_file))
    assert [sample.sample_id for sample in curated] == ["valid"]
    assert reports[0].status is TTSTextCurationStatus.FAILED
    assert reports[0].decision is CurationDecision.UNDECIDED
    assert reports[0].decision_reasons == ("missing_preparation_report",)
    assert outputs.summary.failed_count == 1


def test_unexpected_stage_and_existing_phonemes_are_isolated_failures(
    tmp_path: Path,
) -> None:
    raw = make_sample("raw", stage=TTSTextStage.RAW)
    phonemes = make_sample("phonemes", phonemes=("t", "e", "s", "u"))
    outputs = run_batch(tmp_path, [raw, phonemes])
    reports = list(iter_tts_text_curation_reports(outputs.report_file))
    assert list(iter_text_manifest(outputs.curated_manifest)) == []
    assert [report.status for report in reports] == [
        TTSTextCurationStatus.FAILED,
        TTSTextCurationStatus.FAILED,
    ]
    assert reports[0].decision_reasons == ("unexpected_prepared_state",)
    assert reports[1].decision_reasons == ("phonemes_already_set",)
    assert outputs.summary.failed_count == 2


def test_duplicate_and_fundamentally_mismatched_reports_fail_fast(
    tmp_path: Path,
) -> None:
    sample = make_sample("sample")
    manifest = tmp_path / "manifest.jsonl"
    report_path = tmp_path / "reports.jsonl"
    watchlist_path = write_watchlist(tmp_path / "watchlist.json")
    write_text_manifest(manifest, [sample])
    report = make_preparation_report(sample)
    report_path.write_text(
        f"{report.model_dump_json()}\n{report.model_dump_json()}\n",
        encoding="utf-8",
    )
    pipeline = TTSTextCurationPipeline(
        output_root=tmp_path / "data",
        watchlist=PronunciationRiskWatchlist.load(watchlist_path),
    )
    with pytest.raises(ValueError, match="duplicate sample_id"):
        pipeline.run(manifest, report_path)

    other = make_sample("other")
    write_text_preparation_reports(report_path, [make_preparation_report(other)])
    with pytest.raises(ValueError, match="no matching sample_id"):
        pipeline.run(manifest, report_path)


def test_contradictory_companion_report_fails_fast(tmp_path: Path) -> None:
    sample = make_sample("contradiction")
    report = make_preparation_report(sample)
    altered = report.model_copy(
        update={
            "reading": report.reading.model_copy(
                update={"reading_kana": "ベツノヨミ"}
            )
        }
    )
    with pytest.raises(ValueError, match="reading_kana contradicts"):
        run_batch(tmp_path, [sample], reports=[altered])


def test_watchlist_validation_and_fingerprint(tmp_path: Path) -> None:
    path = write_watchlist(tmp_path / "watchlist.json", ["佐伯"])
    watchlist = PronunciationRiskWatchlist.load(path)
    assert watchlist.fingerprint_sha256 == hashlib.sha256(path.read_bytes()).hexdigest()

    duplicate = {
        "version": "v1",
        "source": "test",
        "entries": [
            {"surface": "佐伯", "category": "name", "reason": "ambiguous"},
            {"surface": "佐伯", "category": "name", "reason": "ambiguous"},
        ],
    }
    path.write_text(json.dumps(duplicate, ensure_ascii=False), encoding="utf-8")
    with pytest.raises(ValueError, match="surfaces must be unique"):
        PronunciationRiskWatchlist.load(path)

    with pytest.raises(ValidationError):
        TTSTextCurationPolicy(policy_version="")
