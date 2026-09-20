from __future__ import annotations

import hashlib
from pathlib import Path

from j_speech_ops.japanese_normalization import JapaneseNormalizer
from j_speech_ops.japanese_reading import (
    OpenJTalkReadingProvider,
    ReadingOverrides,
    ReadingProviderInfo,
    ReadingResult,
    ReadingSampleError,
)
from j_speech_ops.schemas import CurationDecision
from j_speech_ops.text_manifest import iter_text_manifest, write_text_manifest
from j_speech_ops.text_preparation import (
    TextPreparationPipeline,
    TextPreparationStatus,
    iter_text_preparation_reports,
)
from j_speech_ops.tts_text import (
    TTSTextContent,
    TTSTextSample,
    TTSTextSource,
    TTSTextStage,
)


ROOT = Path(__file__).resolve().parents[1]


def make_sample(
    sample_id: str,
    raw_text: str,
    *,
    stage: TTSTextStage = TTSTextStage.RAW,
    normalized_text: str | None = None,
    reading_kana: str | None = None,
) -> TTSTextSample:
    return TTSTextSample(
        sample_id=sample_id,
        dataset_id="stage6-test",
        language="ja",
        locale="ja-JP",
        text=TTSTextContent(
            raw_text=raw_text,
            normalized_text=normalized_text,
            reading_kana=reading_kana,
        ),
        source=TTSTextSource(
            source_type="file",
            source_name="stage6.txt",
            source_id=sample_id,
        ),
        rights_status="cleared",
        intended_use="tts_production",
        domain="hospitality",
        state={"stage": stage.value},
        metadata={"preserve": True},
    )


def pipeline(tmp_path: Path, provider=None) -> TextPreparationPipeline:
    return TextPreparationPipeline(
        output_root=tmp_path / "data",
        normalizer=JapaneseNormalizer(),
        reading_provider=provider or OpenJTalkReadingProvider(),
        overrides=ReadingOverrides.load(
            ROOT / "config" / "japanese_reading_overrides.json"
        ),
    )


def test_pipeline_outputs_both_layers_and_protects_fields(tmp_path: Path) -> None:
    original = make_sample("JP_1", "料金は３,５００円です。")
    protected = original.model_dump(exclude={"text", "state"})
    manifest = tmp_path / "tts_text_samples.jsonl"
    write_text_manifest(manifest, [original])

    outputs = pipeline(tmp_path).run(manifest)
    normalized = list(iter_text_manifest(outputs.normalized_manifest))[0]
    prepared = list(iter_text_manifest(outputs.reading_prepared_manifest))[0]
    report = list(iter_text_preparation_reports(outputs.report_file))[0]

    assert normalized.state.stage is TTSTextStage.NORMALIZED
    assert normalized.text.raw_text == original.text.raw_text
    assert normalized.text.normalized_text == "料金は3500円です。"
    assert normalized.text.reading_kana is None
    assert prepared.state.stage is TTSTextStage.READING_PREPARED
    assert prepared.text.raw_text == original.text.raw_text
    assert prepared.text.reading_kana == "リョーキンワサンゼンゴヒャクエンデス。"
    assert prepared.text.phonemes is None
    assert prepared.state.curation_decision is CurationDecision.UNDECIDED
    assert prepared.model_dump(exclude={"text", "state"}) == protected
    assert report.status is TextPreparationStatus.READING_PREPARED
    assert report.normalization is not None
    assert "numeric_thousands_separator_removed" in report.normalization.transformations
    assert report.reading is not None
    assert report.dependencies.jaconv_version == "0.5.0"


class FailingProvider:
    info = ReadingProviderInfo(
        provider="fake",
        provider_version="1",
        package_name="fake",
        package_version="1",
    )

    def prepare(self, sample_id: str, text: str, overrides) -> ReadingResult:
        if sample_id == "BAD":
            raise ReadingSampleError("synthetic unresolved reading")
        return ReadingResult(
            sample_id=sample_id,
            provider="fake",
            provider_version="1",
            input_text=text,
            baseline_reading_kana="テスト",
            reading_kana="テスト",
            override_count=0,
        )


def test_reading_failure_stays_normalized_and_batch_continues(tmp_path: Path) -> None:
    manifest = tmp_path / "input.jsonl"
    write_text_manifest(
        manifest,
        [make_sample("BAD", "未解決"), make_sample("GOOD", "有効")],
    )
    outputs = pipeline(tmp_path, FailingProvider()).run(manifest)
    normalized = list(iter_text_manifest(outputs.normalized_manifest))
    prepared = list(iter_text_manifest(outputs.reading_prepared_manifest))
    reports = list(iter_text_preparation_reports(outputs.report_file))
    assert [sample.sample_id for sample in normalized] == ["BAD", "GOOD"]
    assert [sample.sample_id for sample in prepared] == ["GOOD"]
    assert reports[0].status is TextPreparationStatus.NORMALIZED_ONLY
    assert reports[0].output_stage is TTSTextStage.NORMALIZED
    assert reports[1].status is TextPreparationStatus.READING_PREPARED


def test_existing_reading_is_skipped_unless_forced(tmp_path: Path) -> None:
    sample = make_sample(
        "DONE",
        "テスト",
        stage=TTSTextStage.READING_PREPARED,
        normalized_text="テスト",
        reading_kana="テスト",
    )
    manifest = tmp_path / "input.jsonl"
    write_text_manifest(manifest, [sample])
    processor = pipeline(tmp_path)
    skipped = processor.run(manifest)
    skipped_report = list(iter_text_preparation_reports(skipped.report_file))[0]
    forced = processor.run(manifest, force=True)
    forced_sample = list(iter_text_manifest(forced.reading_prepared_manifest))[0]
    assert skipped_report.status is TextPreparationStatus.SKIPPED
    assert skipped_report.skipped_existing is True
    assert forced_sample.state.stage is TTSTextStage.READING_PREPARED
    assert forced_sample.text.raw_text == sample.text.raw_text


def test_pipeline_outputs_are_deterministic(tmp_path: Path) -> None:
    manifest = tmp_path / "input.jsonl"
    write_text_manifest(
        manifest,
        [
            make_sample("A", "JR新宿駅までご案内いたします。"),
            make_sample("B", "Wi-Fiをご確認ください。"),
        ],
    )
    processor = pipeline(tmp_path)
    first = processor.run(manifest)
    first_hashes = [
        hashlib.sha256(path.read_bytes()).hexdigest()
        for path in (
            first.normalized_manifest,
            first.reading_prepared_manifest,
            first.report_file,
        )
    ]
    second = processor.run(manifest)
    second_hashes = [
        hashlib.sha256(path.read_bytes()).hexdigest()
        for path in (
            second.normalized_manifest,
            second.reading_prepared_manifest,
            second.report_file,
        )
    ]
    assert first_hashes == second_hashes


def test_stage5_and_speech_schemas_remain_unchanged() -> None:
    expected = {
        "speech_sample.schema.json": "dec2c98402e4433df64e0b0f2280a73c7ee8cf3ee03232d7bcc8ec2ef7b70435",
        "tts_text_sample.schema.json": "169e87dfa910a64bef2e1200eb94927c96857065c76629d7fb07cb107de74ede",
    }
    for filename, expected_hash in expected.items():
        actual = hashlib.sha256((ROOT / "schemas" / filename).read_bytes()).hexdigest()
        assert actual == expected_hash
