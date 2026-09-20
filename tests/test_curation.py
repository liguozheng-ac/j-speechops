from __future__ import annotations

import hashlib
import wave
from pathlib import Path

import pytest
from pydantic import ValidationError

from j_speech_ops.curation import (
    CurationPolicy,
    SpeechCurationPipeline,
    SpeechCurationReport,
    iter_speech_curation_reports,
)
from j_speech_ops.manifest import iter_manifest, write_manifest
from j_speech_ops.schemas import (
    CurationDecision,
    ProcessingStage,
    RightsStatus,
    SpeechSample,
)


def make_wav(path: Path, *, sample_value: int = 0) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with wave.open(str(path), "wb") as output:
        output.setnchannels(1)
        output.setsampwidth(2)
        output.setframerate(16_000)
        output.writeframes(sample_value.to_bytes(2, "little", signed=True) * 160)


def make_sample(
    sample_id: str,
    audio_path: str,
    *,
    text: str | None = "今日は良い天気です",
    duration: float | None = 2.0,
    rights: RightsStatus = RightsStatus.CLEARED,
    stage: ProcessingStage = ProcessingStage.TRANSCRIBED,
    decision: CurationDecision = CurationDecision.UNDECIDED,
    reasons: list[str] | None = None,
) -> SpeechSample:
    return SpeechSample.model_validate(
        {
            "sample_id": sample_id,
            "dataset_id": "stage4-test",
            "audio": {
                "path": audio_path,
                "format": "wav",
                "duration_sec": duration,
                "sample_rate_hz": 16000,
                "channels": 1,
            },
            "text": {
                "reference_text": "参照テキスト",
                "asr_text": text,
                "normalized_text": "既存の正規化テキスト",
                "reading_kana": "キソンノセイキカテキスト",
                "phonemes": ["k", "i"],
            },
            "source": {
                "source_name": "test-source",
                "source_record_id": f"source-{sample_id}",
                "license": "CC0-1.0",
                "rights_status": rights.value,
            },
            "lineage": {
                "parent_sample_id": f"parent-{sample_id}",
                "segment_start_sec": 1.0,
                "segment_end_sec": 3.0,
            },
            "state": {
                "stage": stage.value,
                "curation_decision": decision.value,
                "decision_reasons": reasons or [],
            },
            "metadata": {"preserve": True},
        }
    )


def run_batch(tmp_path: Path, samples: list[SpeechSample], *, force: bool = False):
    input_manifest = tmp_path / "transcribed_segments.jsonl"
    write_manifest(input_manifest, samples)
    return SpeechCurationPipeline(
        dataset_root=tmp_path,
        output_root=tmp_path / "data",
    ).run(input_manifest, force=force)


def decision_and_codes(outputs) -> tuple[CurationDecision, set[str]]:
    sample = list(iter_manifest(outputs.curated_manifest))[0]
    report = list(iter_speech_curation_reports(outputs.report_file))[0]
    return sample.state.curation_decision, {issue.code for issue in report.issues}


def test_clean_sample_passes_and_preserves_canonical_fields(tmp_path: Path) -> None:
    make_wav(tmp_path / "audio" / "clean.wav", sample_value=1)
    original = make_sample("clean", "audio/clean.wav")
    protected_before = original.model_dump(exclude={"state"})

    outputs = run_batch(tmp_path, [original])
    curated = list(iter_manifest(outputs.curated_manifest))[0]
    report = list(iter_speech_curation_reports(outputs.report_file))[0]

    assert curated.state.stage is ProcessingStage.CURATED
    assert curated.state.curation_decision is CurationDecision.PASS
    assert curated.state.decision_reasons == []
    assert curated.model_dump(exclude={"state"}) == protected_before
    assert report.issues == []
    assert report.policy_version == "speech-curation-v1"
    assert len(report.audio_sha256 or "") == 64


@pytest.mark.parametrize(
    ("sample_id", "text", "audio_exists", "rights", "expected_code"),
    [
        ("empty", "  ", True, RightsStatus.CLEARED, "empty_transcript"),
        ("missing", "今日は晴れ", False, RightsStatus.CLEARED, "audio_missing"),
        ("restricted", "今日は晴れ", True, RightsStatus.RESTRICTED, "rights_restricted"),
    ],
)
def test_hard_drop_rules(
    tmp_path: Path,
    sample_id: str,
    text: str,
    audio_exists: bool,
    rights: RightsStatus,
    expected_code: str,
) -> None:
    if audio_exists:
        make_wav(tmp_path / f"{sample_id}.wav", sample_value=2)
    sample = make_sample(sample_id, f"{sample_id}.wav", text=text, rights=rights)
    outputs = run_batch(tmp_path, [sample])
    decision, codes = decision_and_codes(outputs)
    assert decision is CurationDecision.DROP
    assert expected_code in codes


@pytest.mark.parametrize(
    ("sample_id", "text", "duration", "rights", "expected_code"),
    [
        ("unknown", "今日は晴れです", 2.0, RightsStatus.UNKNOWN, "rights_unknown"),
        ("short", "はい", 0.2, RightsStatus.CLEARED, "duration_too_short"),
        ("long", "長い発話です", 31.0, RightsStatus.CLEARED, "duration_too_long"),
        ("density", "あ" * 50, 1.0, RightsStatus.CLEARED, "transcript_density_anomaly"),
        ("latin", "OpenAI GPT 2026", 2.0, RightsStatus.CLEARED, "low_japanese_script_ratio"),
        (
            "repeat",
            "ありがとうございました" * 3,
            5.0,
            RightsStatus.CLEARED,
            "excessive_text_repetition",
        ),
    ],
)
def test_review_rules(
    tmp_path: Path,
    sample_id: str,
    text: str,
    duration: float,
    rights: RightsStatus,
    expected_code: str,
) -> None:
    make_wav(tmp_path / f"{sample_id}.wav", sample_value=3)
    outputs = run_batch(
        tmp_path,
        [make_sample(sample_id, f"{sample_id}.wav", text=text, duration=duration, rights=rights)],
    )
    decision, codes = decision_and_codes(outputs)
    assert decision is CurationDecision.REVIEW
    assert expected_code in codes


def test_exact_audio_duplicates_route_every_member_to_review(tmp_path: Path) -> None:
    make_wav(tmp_path / "a.wav", sample_value=42)
    (tmp_path / "b.wav").write_bytes((tmp_path / "a.wav").read_bytes())
    outputs = run_batch(
        tmp_path,
        [make_sample("a", "a.wav"), make_sample("b", "b.wav")],
    )
    samples = list(iter_manifest(outputs.curated_manifest))
    reports = list(iter_speech_curation_reports(outputs.report_file))
    assert [sample.state.curation_decision for sample in samples] == [
        CurationDecision.REVIEW,
        CurationDecision.REVIEW,
    ]
    assert all("duplicate_audio" in report.decision_reasons for report in reports)
    assert reports[0].audio_sha256 == reports[1].audio_sha256


def test_hard_drop_takes_priority_over_review(tmp_path: Path) -> None:
    make_wav(tmp_path / "combined.wav")
    outputs = run_batch(
        tmp_path,
        [
            make_sample(
                "combined",
                "combined.wav",
                duration=0.2,
                rights=RightsStatus.RESTRICTED,
            )
        ],
    )
    decision, codes = decision_and_codes(outputs)
    assert decision is CurationDecision.DROP
    assert {"rights_restricted", "duration_too_short"} <= codes


def test_manifest_outputs_summary_and_deterministic_rerun(tmp_path: Path) -> None:
    make_wav(tmp_path / "pass.wav", sample_value=10)
    make_wav(tmp_path / "drop.wav", sample_value=11)
    samples = [
        make_sample("pass", "pass.wav"),
        make_sample("drop", "drop.wav", text=""),
    ]
    first = run_batch(tmp_path, samples)
    first_bytes = {
        path.name: path.read_bytes()
        for path in (first.curated_manifest, first.usable_manifest, first.report_file)
    }
    second = run_batch(tmp_path, samples)
    usable = list(iter_manifest(second.usable_manifest))

    assert first_bytes == {
        path.name: path.read_bytes()
        for path in (second.curated_manifest, second.usable_manifest, second.report_file)
    }
    assert [sample.sample_id for sample in usable] == ["pass"]
    assert second.summary.total_samples == 2
    assert second.summary.pass_count == 1
    assert second.summary.review_count == 0
    assert second.summary.drop_count == 1
    assert second.summary.total_audio_duration_sec == 4.0


def test_existing_curated_is_skipped_unless_forced(tmp_path: Path) -> None:
    curated = make_sample(
        "existing",
        "now-missing.wav",
        stage=ProcessingStage.CURATED,
        decision=CurationDecision.PASS,
    )
    input_manifest = tmp_path / "input.jsonl"
    write_manifest(input_manifest, [curated])
    pipeline = SpeechCurationPipeline(
        dataset_root=tmp_path,
        output_root=tmp_path / "data",
    )

    skipped = pipeline.run(input_manifest)
    skipped_report = list(iter_speech_curation_reports(skipped.report_file))[0]
    forced = pipeline.run(input_manifest, force=True)

    assert skipped_report.skipped_existing is True
    assert skipped_report.policy_version == "existing-curation-policy-unknown"
    assert (
        list(iter_manifest(forced.curated_manifest))[0].state.curation_decision
        is CurationDecision.DROP
    )


def test_policy_validation_and_final_report_validation() -> None:
    with pytest.raises(ValidationError):
        CurationPolicy(min_duration_sec=2, max_duration_sec=1)
    with pytest.raises(ValidationError):
        SpeechCurationReport(
            sample_id="bad",
            policy_version="v1",
            decision=CurationDecision.UNDECIDED,
            transcript_chars=0,
        )


def test_duplicate_sample_ids_are_a_fundamentally_broken_manifest(tmp_path: Path) -> None:
    make_wav(tmp_path / "same.wav")
    with pytest.raises(ValueError, match="duplicate sample_id"):
        run_batch(
            tmp_path,
            [make_sample("same", "same.wav"), make_sample("same", "same.wav")],
        )


def test_audio_hash_is_standard_sha256(tmp_path: Path) -> None:
    make_wav(tmp_path / "hash.wav", sample_value=99)
    outputs = run_batch(tmp_path, [make_sample("hash", "hash.wav")])
    report = list(iter_speech_curation_reports(outputs.report_file))[0]
    assert report.audio_sha256 == hashlib.sha256((tmp_path / "hash.wav").read_bytes()).hexdigest()
