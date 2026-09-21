from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest

from j_speech_ops.generated_audio_qa import (
    ASRRoundTripEvidence,
    AudioIntegrityEvidence,
    ContentComparisonEvidence,
    GeneratedAudioQAResult,
    QADecision,
    QAExecutionStatus,
    QAPassedAudioArtifact,
    write_qa_results,
)
from j_speech_ops.japanese_normalization import NormalizationResult
from j_speech_ops.japanese_reading import AppliedReadingOverride, ReadingProviderInfo
from j_speech_ops.pronunciation_regression import (
    ConfirmationStatus,
    ExpectationScope,
    ExpectationSource,
    ExpectedPronunciationReference,
    PronunciationEvidenceDecision,
    PronunciationEvidencePipeline,
    PronunciationEvidenceResult,
    PronunciationRiskEvidence,
    PronunciationRegressionBaseline,
    RegressionChangeKind,
    RegressionDecision,
    RiskAlignmentStatus,
    RiskEvidenceSource,
    capture_pronunciation_baseline,
    compare_pronunciation_regression,
    evaluate_expected_sequence,
    iter_pronunciation_baselines,
    iter_pronunciation_evidence,
    iter_pronunciation_regressions,
    pronunciation_decision_for_risks,
)
from j_speech_ops.pronunciation_regression_cli import build_parser
from j_speech_ops.pronunciation_risks import PronunciationRiskMatch
from j_speech_ops.schemas import CurationDecision
from j_speech_ops.text_preparation import TextPreparationStatus
from j_speech_ops.tts_synthesis import (
    GeneratedAudioArtifact,
    GenerationParameters,
    SynthesisRun,
    SynthesisRunStatus,
    write_generated_audio_manifest,
    write_synthesis_runs,
)
from j_speech_ops.tts_text import TTSTextStage
from j_speech_ops.tts_text_curation import (
    TTSTextCurationReport,
    TTSTextCurationStatus,
    write_tts_text_curation_reports,
)


HASH_A = "a" * 64
HASH_B = "b" * 64
HASH_C = "c" * 64
HASH_D = "d" * 64
PROVIDER = ReadingProviderInfo(
    provider="openjtalk-frontend",
    provider_version="test",
    package_name="pyopenjtalk-plus",
    package_version="test",
)


def write_one(path: Path, model) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(model.model_dump_json() + "\n", encoding="utf-8")


def make_inputs(
    tmp_path: Path,
    *,
    sample_id: str = "CASE_JR",
    source_text: str = "JR新宿駅までご案内いたします。",
    normalized_text: str = "JR新宿駅までご案内いたします。",
    synthesis_text: str = "ジェイアール新宿駅までご案内いたします。",
    expected_reading: str = "ジェイアールシンジュクエキマデゴアンナイイタシマス。",
    observed_reading: str = "ジェイアールシンジュクエキマデゴアンナイイタシマス",
    asr_transcript: str = "JR新宿駅までご案内いたします",
    override: AppliedReadingOverride | None = None,
    unresolved_risk: PronunciationRiskMatch | None = None,
    speaker: str = "Ono_Anna",
    audio_hash: str = HASH_A,
):
    if override is None and unresolved_risk is None and "JR" in normalized_text:
        override = AppliedReadingOverride(
            surface="JR",
            reading_kana="ジェイアール",
            source="test confirmed override fixture; not human listening",
            occurrences=1,
        )
    overrides = (override,) if override else ()
    run_id = hashlib.sha256(f"run:{sample_id}".encode()).hexdigest()
    qa_id = hashlib.sha256(f"qa:{sample_id}".encode()).hexdigest()
    canonical_expected = expected_reading.rstrip("。")
    canonical_observed = observed_reading.rstrip("。")
    artifact = GeneratedAudioArtifact(
        run_id=run_id,
        sample_id=sample_id,
        input_text=normalized_text,
        expected_reading_kana=expected_reading,
        synthesis_text=synthesis_text,
        applied_pronunciation_overrides=overrides,
        audio_path=str(tmp_path / f"{sample_id}.wav"),
        backend="qwen3-tts",
        backend_version="0.1.1",
        model_name="Qwen/Qwen3-TTS-12Hz-1.7B-CustomVoice",
        speaker=speaker,
        language="Japanese",
        sample_rate_hz=24_000,
        channels=1,
        duration_sec=1.0,
        audio_sha256=audio_hash,
    )
    run = SynthesisRun(
        run_id=run_id,
        sample_id=sample_id,
        backend=artifact.backend,
        backend_version=artifact.backend_version,
        model_name=artifact.model_name,
        model_path="D:/external/qwen",
        speaker=speaker,
        language="Japanese",
        device="cuda:0",
        dtype="bfloat16",
        attention_implementation="sdpa",
        generation_parameters=GenerationParameters(),
        config_fingerprint_sha256=HASH_B,
        seed=123,
        normalized_text=normalized_text,
        synthesis_text=synthesis_text,
        expected_reading_kana=expected_reading,
        applied_pronunciation_overrides=overrides,
        status=SynthesisRunStatus.SUCCESS,
        output_audio_path=artifact.audio_path,
        sample_rate_hz=24_000,
        channels=1,
        duration_sec=1.0,
        generation_time_sec=2.0,
        rtf=2.0,
        audio_sha256=audio_hash,
    )
    integrity = AudioIntegrityEvidence(
        audio_path=artifact.audio_path,
        file_size_bytes=100,
        expected_audio_sha256=audio_hash,
        observed_audio_sha256=audio_hash,
        expected_sample_rate_hz=24_000,
        observed_sample_rate_hz=24_000,
        expected_channels=1,
        observed_channels=1,
        expected_duration_sec=1.0,
        observed_duration_sec=1.0,
    )
    round_trip = ASRRoundTripEvidence(
        backend="fake-whisper",
        backend_version="1",
        model_name="large-v3",
        model_identity="fake-whisper:1:large-v3",
        config_fingerprint_sha256=HASH_C,
        transcript=asr_transcript,
        normalized_transcript=asr_transcript,
        observed_reading_kana=observed_reading,
        inference_time_sec=0.1,
        audio_duration_sec=1.0,
        rtf=0.1,
    )
    comparison = ContentComparisonEvidence(
        expected_reading_kana=expected_reading,
        observed_reading_kana=observed_reading,
        canonical_expected_reading=canonical_expected,
        canonical_observed_reading=canonical_observed,
        edit_distance=0,
        cer=0.0,
    )
    qa_result = GeneratedAudioQAResult(
        qa_id=qa_id,
        run_id=run_id,
        sample_id=sample_id,
        policy_version="generated-audio-content-qa-v1",
        policy_fingerprint_sha256=HASH_D,
        status=QAExecutionStatus.COMPLETED,
        decision=QADecision.PASS,
        audio_path=artifact.audio_path,
        actual_synthesis_text=synthesis_text,
        expected_audio_sha256=audio_hash,
        observed_audio_sha256=audio_hash,
        expected_reading_kana=expected_reading,
        asr_model_identity=round_trip.model_identity,
        asr_config_fingerprint_sha256=HASH_C,
        reading_provider_fingerprint_sha256=HASH_B,
        asr_transcript=asr_transcript,
        normalized_asr_transcript=asr_transcript,
        observed_reading_kana=observed_reading,
        canonical_expected_reading=canonical_expected,
        canonical_observed_reading=canonical_observed,
        edit_distance=0,
        cer=0.0,
        asr_inference_time_sec=0.1,
        audio_duration_sec=1.0,
        asr_rtf=0.1,
        audio_integrity=integrity,
        asr_round_trip=round_trip,
        content_comparison=comparison,
    )
    qa_pass = QAPassedAudioArtifact(
        qa_id=qa_id,
        run_id=run_id,
        sample_id=sample_id,
        audio_path=artifact.audio_path,
        audio_sha256=audio_hash,
        input_text=normalized_text,
        synthesis_text=synthesis_text,
        expected_reading_kana=expected_reading,
        asr_transcript=asr_transcript,
        observed_reading_kana=observed_reading,
        canonical_expected_reading=canonical_expected,
        canonical_observed_reading=canonical_observed,
        policy_version="generated-audio-content-qa-v1",
        policy_fingerprint_sha256=HASH_D,
        asr_model_identity=round_trip.model_identity,
        asr_config_fingerprint_sha256=HASH_C,
        reading_provider_fingerprint_sha256=HASH_B,
        tts_backend=artifact.backend,
        tts_model_name=artifact.model_name,
        speaker=speaker,
    )
    curation = TTSTextCurationReport(
        sample_id=sample_id,
        policy_version="tts-text-curation-v1",
        policy_fingerprint_sha256=HASH_A,
        status=TTSTextCurationStatus.CURATED,
        input_stage=TTSTextStage.READING_PREPARED,
        output_stage=TTSTextStage.CURATED,
        decision=CurationDecision.PASS,
        raw_text=source_text,
        normalized_text=normalized_text,
        reading_kana=expected_reading,
        preparation_policy_version="japanese-text-prep-v1.1",
        preparation_status=TextPreparationStatus.READING_PREPARED,
        normalization=NormalizationResult(
            policy_version="japanese-text-prep-v1.1",
            raw_text=source_text,
            normalized_text=normalized_text,
            changed=source_text != normalized_text,
            transformations=("fixture_normalization",)
            if source_text != normalized_text
            else (),
            jaconv_version="0.5.0",
        ),
        reading_provider=PROVIDER,
        applied_overrides=overrides,
        override_fingerprint_sha256=HASH_B,
        risk_terms=(unresolved_risk,) if unresolved_risk else (),
        watchlist_version="test-watchlist-v1",
        watchlist_source="test fixture",
        watchlist_fingerprint_sha256=HASH_C,
    )

    paths = {
        "artifacts": tmp_path / "generated_audio_manifest.jsonl",
        "runs": tmp_path / "synthesis_runs.jsonl",
        "qa_results": tmp_path / "generated_audio_qa_results.jsonl",
        "qa_pass": tmp_path / "qa_pass_audio_manifest.jsonl",
        "curation": tmp_path / "tts_text_curation_report.jsonl",
    }
    write_generated_audio_manifest(paths["artifacts"], [artifact])
    write_synthesis_runs(paths["runs"], [run])
    write_qa_results(paths["qa_results"], [qa_result])
    write_one(paths["qa_pass"], qa_pass)
    write_tts_text_curation_reports(paths["curation"], [curation])
    return paths


def build_evidence(tmp_path: Path, **kwargs):
    paths = make_inputs(tmp_path, **kwargs)
    outputs = PronunciationEvidencePipeline(output_root=tmp_path / "stage10a").build(
        artifact_manifest=paths["artifacts"],
        synthesis_runs_manifest=paths["runs"],
        qa_results_manifest=paths["qa_results"],
        qa_pass_manifest=paths["qa_pass"],
        curation_report=paths["curation"],
    )
    return outputs, list(iter_pronunciation_evidence(outputs.evidence_manifest))[0]


def write_evidence(path: Path, evidence: PronunciationEvidenceResult) -> None:
    write_one(path, evidence)


def validated_copy(evidence: PronunciationEvidenceResult, **updates) -> PronunciationEvidenceResult:
    payload = evidence.model_dump(mode="json")
    payload.update(updates)
    return PronunciationEvidenceResult.model_validate(payload)


def test_stage7_to_stage10a_provenance_is_traceable(tmp_path: Path) -> None:
    outputs, evidence = build_evidence(tmp_path)
    assert outputs.total == 1
    assert evidence.sample_id == "CASE_JR"
    assert evidence.artifact_run_id == evidence.synthesis_run_id
    assert evidence.stage9_qa_id
    assert evidence.stage7_policy_version == "tts-text-curation-v1"
    assert evidence.stage9_policy_version == "generated-audio-content-qa-v1"
    assert evidence.audio_sha256 == HASH_A
    assert evidence.tts_model_name.endswith("CustomVoice")
    assert evidence.speaker == "Ono_Anna"


def test_confirmed_override_has_explicit_source_and_supporting_evidence(
    tmp_path: Path,
) -> None:
    _, evidence = build_evidence(tmp_path)
    reference = next(
        item
        for item in evidence.expected_references
        if item.scope is ExpectationScope.RISK_SPAN
    )
    risk = evidence.risk_evidence[0]
    assert reference.surface == "JR"
    assert reference.expected_reading == "ジェイアール"
    assert reference.expectation_source is ExpectationSource.CONFIRMED_OVERRIDE
    assert reference.confirmation_status is ConfirmationStatus.CONFIG_CONFIRMED
    assert risk.expected_reference_id == reference.reference_id
    assert risk.alignment_status is RiskAlignmentStatus.EXPECTED_SEQUENCE_OBSERVED
    assert evidence.decision is PronunciationEvidenceDecision.NO_REVIEW_NEEDED
    assert any(span.text_form == "normalized" for span in risk.spans)


def test_reading_provider_expectation_is_not_human_confirmed(tmp_path: Path) -> None:
    _, evidence = build_evidence(tmp_path)
    full = next(
        item
        for item in evidence.expected_references
        if item.scope is ExpectationScope.FULL_TEXT
    )
    assert full.expectation_source is ExpectationSource.READING_PROVIDER
    assert full.confirmation_status is ConfirmationStatus.DERIVED


def test_unconfirmed_watchlist_risk_stays_unresolved_and_routes_review(
    tmp_path: Path,
) -> None:
    risk = PronunciationRiskMatch(
        surface="佐伯",
        category="proper_name",
        reason="multiple_readings_possible",
        occurrence_count=1,
    )
    outputs, evidence = build_evidence(
        tmp_path,
        sample_id="CASE_SAEKI",
        source_text="佐伯様をご案内します。",
        normalized_text="佐伯様をご案内します。",
        synthesis_text="佐伯様をご案内します。",
        expected_reading="サエキサマヲゴアンナイシマス。",
        observed_reading="サエキサマヲゴアンナイシマス",
        asr_transcript="佐伯様をご案内します",
        override=None,
        unresolved_risk=risk,
    )
    item = evidence.risk_evidence[0]
    assert item.expected_reference_id is None
    assert item.alignment_status is RiskAlignmentStatus.ALIGNMENT_UNRESOLVED
    assert evidence.decision is PronunciationEvidenceDecision.REVIEW
    assert outputs.review_count == 1


def test_expected_sequence_missing_is_review_evidence_not_drop() -> None:
    reference = ExpectedPronunciationReference(
        reference_id=HASH_A,
        scope=ExpectationScope.RISK_SPAN,
        surface="JR",
        expected_reading="ジェイアール",
        canonical_expected_reading="ジェイアール",
        expectation_source=ExpectationSource.CONFIRMED_OVERRIDE,
        confirmation_status=ConfirmationStatus.CONFIG_CONFIRMED,
        provenance_reference="test override",
    )
    status = evaluate_expected_sequence(reference, "ジェーアールシンジュク")
    assert status is RiskAlignmentStatus.EXPECTED_SEQUENCE_NOT_OBSERVED
    risk = PronunciationRiskEvidence(
        risk_id=HASH_B,
        surface="JR",
        risk_type="confirmed_override_surface",
        reason="confirmed_override_applied",
        occurrence_count=1,
        evidence_source=RiskEvidenceSource.CONFIRMED_OVERRIDE_PROVENANCE,
        spans=(),
        expected_reference_id=reference.reference_id,
        alignment_status=status,
        evidence_statement="Expected sequence was not observed; review required.",
    )
    assert pronunciation_decision_for_risks([risk]) is PronunciationEvidenceDecision.REVIEW
    assert evaluate_expected_sequence(None, "サエキ") is RiskAlignmentStatus.ALIGNMENT_UNRESOLVED


def test_expectation_source_types_remain_distinct() -> None:
    assert {item.value for item in ExpectationSource} == {
        "normalization_rule",
        "reading_provider",
        "confirmed_override",
        "human_confirmed",
    }


def test_capture_baseline_is_deterministic(tmp_path: Path) -> None:
    outputs, evidence = build_evidence(tmp_path)
    first = tmp_path / "baseline-1.jsonl"
    second = tmp_path / "baseline-2.jsonl"
    assert capture_pronunciation_baseline(
        evidence_manifest=outputs.evidence_manifest, output_path=first
    ) == 1
    assert capture_pronunciation_baseline(
        evidence_manifest=outputs.evidence_manifest, output_path=second
    ) == 1
    assert first.read_bytes() == second.read_bytes()
    baseline = list(iter_pronunciation_baselines(first))[0]
    assert baseline.logical_case_id == evidence.logical_case_id
    assert baseline.observed_canonical_reading == evidence.observed_evidence.canonical_observed_reading


def prepare_regression(tmp_path: Path):
    outputs, evidence = build_evidence(tmp_path)
    baseline = tmp_path / "pronunciation_regression_baseline.jsonl"
    capture_pronunciation_baseline(
        evidence_manifest=outputs.evidence_manifest, output_path=baseline
    )
    return evidence, baseline


def run_modified_regression(
    tmp_path: Path,
    evidence: PronunciationEvidenceResult,
    baseline: Path,
    name: str,
):
    current = tmp_path / f"current-{name}.jsonl"
    write_evidence(current, evidence)
    outputs = compare_pronunciation_regression(
        evidence_manifest=current,
        baseline_manifest=baseline,
        output_root=tmp_path / f"compare-{name}",
    )
    result = list(iter_pronunciation_regressions(outputs.results_manifest))[0]
    return outputs, result


def test_expected_reading_change_triggers_regression_review(tmp_path: Path) -> None:
    evidence, baseline = prepare_regression(tmp_path)
    payload = evidence.model_dump(mode="json")
    full = payload["expected_references"][0]
    full["expected_reading"] = "ジェーアールシンジュク"
    full["canonical_expected_reading"] = "ジェーアールシンジュク"
    full["reference_id"] = HASH_D
    payload["evidence_id"] = HASH_C
    current = PronunciationEvidenceResult.model_validate(payload)
    _, result = run_modified_regression(tmp_path, current, baseline, "expected")
    assert result.decision is RegressionDecision.REVIEW
    assert RegressionChangeKind.EXPECTED_READING_CHANGED in {
        change.kind for change in result.changes
    }


def test_observed_reading_change_triggers_regression_review(tmp_path: Path) -> None:
    evidence, baseline = prepare_regression(tmp_path)
    observed = evidence.observed_evidence.model_copy(
        update={"canonical_observed_reading": "ジェーアールシンジュク"}
    )
    current = validated_copy(evidence, evidence_id=HASH_C, observed_evidence=observed)
    _, result = run_modified_regression(tmp_path, current, baseline, "observed")
    assert result.decision is RegressionDecision.REVIEW
    assert RegressionChangeKind.OBSERVED_READING_CHANGED in {
        change.kind for change in result.changes
    }


def test_risk_status_change_triggers_review(tmp_path: Path) -> None:
    evidence, baseline = prepare_regression(tmp_path)
    risk = evidence.risk_evidence[0].model_copy(
        update={
            "alignment_status": RiskAlignmentStatus.EXPECTED_SEQUENCE_NOT_OBSERVED,
            "evidence_statement": "Synthetic missing-sequence regression fixture.",
        }
    )
    current = validated_copy(
        evidence,
        evidence_id=HASH_C,
        risk_evidence=(risk,),
        decision=PronunciationEvidenceDecision.REVIEW,
        review_reasons=("JR:expected_sequence_not_observed",),
    )
    _, result = run_modified_regression(tmp_path, current, baseline, "risk")
    assert result.decision is RegressionDecision.REVIEW
    assert RegressionChangeKind.RISK_STATUS_CHANGED in {
        change.kind for change in result.changes
    }


@pytest.mark.parametrize("field", ["speaker", "tts_model_name", "tts_backend"])
def test_config_change_is_recorded_but_not_regression_by_itself(
    tmp_path: Path, field: str
) -> None:
    evidence, baseline = prepare_regression(tmp_path)
    current = validated_copy(evidence, evidence_id=HASH_C, **{field: f"changed-{field}"})
    outputs, result = run_modified_regression(tmp_path, current, baseline, field)
    assert result.decision is RegressionDecision.NO_REGRESSION_DETECTED
    assert outputs.summary.provenance_only_changes == 1
    assert any(
        change.kind is RegressionChangeKind.PROVENANCE_CONFIG_CHANGED
        and change.field == field
        and not change.material_pronunciation_change
        for change in result.changes
    )


def test_audio_hash_change_is_identity_not_pronunciation_regression(
    tmp_path: Path,
) -> None:
    evidence, baseline = prepare_regression(tmp_path)
    current = validated_copy(evidence, evidence_id=HASH_C, audio_sha256=HASH_D)
    _, result = run_modified_regression(tmp_path, current, baseline, "audio-hash")
    assert result.decision is RegressionDecision.NO_REGRESSION_DETECTED
    assert result.changes[0].kind is RegressionChangeKind.ARTIFACT_IDENTITY_CHANGED
    assert result.changes[0].material_pronunciation_change is False


def test_identical_evidence_has_no_regression(tmp_path: Path) -> None:
    evidence, baseline = prepare_regression(tmp_path)
    outputs, result = run_modified_regression(tmp_path, evidence, baseline, "same")
    assert result.decision is RegressionDecision.NO_REGRESSION_DETECTED
    assert result.changes == ()
    assert outputs.summary.no_regression_detected == 1
    assert outputs.summary.review_candidates == 0


def test_review_manifest_contains_review_only(tmp_path: Path) -> None:
    risk = PronunciationRiskMatch(
        surface="佐伯",
        category="proper_name",
        reason="multiple_readings_possible",
        occurrence_count=1,
    )
    outputs, _ = build_evidence(
        tmp_path,
        sample_id="CASE_REVIEW",
        source_text="佐伯様です。",
        normalized_text="佐伯様です。",
        synthesis_text="佐伯様です。",
        expected_reading="サエキサマデス。",
        observed_reading="サエキサマデス",
        asr_transcript="佐伯様です",
        override=None,
        unresolved_risk=risk,
    )
    review = list(iter_pronunciation_evidence(outputs.review_candidates))
    assert len(review) == 1
    assert review[0].decision is PronunciationEvidenceDecision.REVIEW


def test_fixture_sources_are_explicit_and_not_fake_human_history() -> None:
    fixture = Path(__file__).parent / "fixtures" / "pronunciation_regression_cases.json"
    payload = json.loads(fixture.read_text(encoding="utf-8"))
    assert len(payload["cases"]) == 6
    sources = {item["expectation_source"] for item in payload["cases"]}
    assert sources == {"reading_provider", "confirmed_override"}
    assert all(item["confirmation_status"] != "human_confirmed" for item in payload["cases"])


def test_contracts_have_no_unjustified_quality_score_fields() -> None:
    fields = set(PronunciationEvidenceResult.model_fields)
    assert "pronunciation_score" not in fields
    assert "pitch_accent_accuracy" not in fields
    assert "mos" not in fields
    assert "naturalness_score" not in fields


def test_cli_supports_build_capture_and_compare() -> None:
    parser = build_parser()
    build = parser.parse_args(
        [
            "build",
            "--artifact-manifest",
            "artifacts.jsonl",
            "--synthesis-runs",
            "runs.jsonl",
            "--qa-results",
            "qa.jsonl",
            "--qa-pass-manifest",
            "pass.jsonl",
            "--curation-report",
            "curation.jsonl",
        ]
    )
    capture = parser.parse_args(
        ["capture-baseline", "--evidence", "evidence.jsonl", "--output", "base.jsonl"]
    )
    compare = parser.parse_args(
        ["compare", "--evidence", "evidence.jsonl", "--baseline", "base.jsonl"]
    )
    assert build.command == "build"
    assert capture.command == "capture-baseline"
    assert compare.command == "compare"
