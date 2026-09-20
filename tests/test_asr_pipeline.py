from __future__ import annotations

from pathlib import Path

import numpy as np
import soundfile as sf

from j_speech_ops.asr import ASRConfig, ASRResult, ASRRuntimeInfo, ASRSegment
from j_speech_ops.asr_pipeline import ASRBatchPipeline, ASRRunStatus, iter_asr_run_reports
from j_speech_ops.manifest import iter_manifest, write_manifest
from j_speech_ops.schemas import CurationDecision, ProcessingStage, SpeechSample


class FakeASR:
    config = ASRConfig()

    @property
    def runtime_info(self) -> ASRRuntimeInfo:
        return ASRRuntimeInfo(
            backend_name="fake-asr",
            backend_version="test",
            model_name=self.config.model_name,
            device=self.config.device,
            compute_type=self.config.compute_type,
            language=self.config.language,
            effective_config=self.config.model_dump(mode="json"),
        )

    def __init__(self, text: str = "今日は晴れです。") -> None:
        self.text = text
        self.calls: list[Path] = []

    def transcribe(self, audio_path: Path) -> ASRResult:
        self.calls.append(audio_path)
        segments = []
        if self.text:
            segments = [ASRSegment(start_sec=0, end_sec=0.5, text=self.text)]
        return ASRResult(
            text=self.text,
            language="ja",
            language_probability=0.98,
            duration_sec=0.5,
            segments=segments,
        )


def make_wav(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    sf.write(path, np.zeros(8_000, dtype=np.float32), 16_000, subtype="PCM_16")


def make_sample(
    sample_id: str,
    audio_path: str,
    stage: ProcessingStage = ProcessingStage.AUDIO_PREPARED,
    asr_text: str | None = None,
) -> SpeechSample:
    return SpeechSample.model_validate(
        {
            "sample_id": sample_id,
            "dataset_id": "stage3-test",
            "audio": {
                "path": audio_path,
                "format": "wav",
                "duration_sec": 0.5,
                "sample_rate_hz": 16000,
                "channels": 1,
            },
            "text": {
                "reference_text": "参照文です。",
                "asr_text": asr_text,
                "normalized_text": "参照文です。",
                "reading_kana": "サンショウブンデス。",
                "phonemes": ["s", "a"],
            },
            "source": {
                "source_name": "test-source",
                "license": "test-license",
                "rights_status": "cleared",
            },
            "state": {"stage": stage.value},
        }
    )


def pipeline(tmp_path: Path, adapter: FakeASR) -> ASRBatchPipeline:
    return ASRBatchPipeline(
        dataset_root=tmp_path,
        output_root=tmp_path / "data",
        adapter=adapter,
    )


def test_success_writes_only_asr_text_and_advances_lifecycle(tmp_path: Path) -> None:
    make_wav(tmp_path / "segments" / "JP_1.wav")
    original = make_sample("JP_1", "segments/JP_1.wav")
    input_manifest = tmp_path / "audio_segments.jsonl"
    write_manifest(input_manifest, [original])
    adapter = FakeASR()

    outputs = pipeline(tmp_path, adapter).run(input_manifest)
    transcribed = list(iter_manifest(outputs.transcribed_manifest))[0]
    report = list(iter_asr_run_reports(outputs.report_file))[0]

    assert transcribed.state.stage is ProcessingStage.TRANSCRIBED
    assert transcribed.text.asr_text == "今日は晴れです。"
    assert transcribed.text.reference_text == original.text.reference_text
    assert transcribed.text.normalized_text == original.text.normalized_text
    assert transcribed.text.reading_kana == original.text.reading_kana
    assert transcribed.text.phonemes == original.text.phonemes
    assert report.status is ASRRunStatus.TRANSCRIBED
    assert report.requested_language == "ja"
    assert report.effective_config["vad_filter"] is False
    assert report.transcription_segment_count == 1
    assert outputs.summary.transcribed == 1


def test_rerun_reuses_existing_transcript_unless_forced(tmp_path: Path) -> None:
    make_wav(tmp_path / "segment.wav")
    input_manifest = tmp_path / "audio_segments.jsonl"
    write_manifest(input_manifest, [make_sample("JP_2", "segment.wav")])
    adapter = FakeASR()
    batch = pipeline(tmp_path, adapter)

    first = batch.run(input_manifest)
    second = batch.run(input_manifest)
    forced = batch.run(input_manifest, force=True)

    assert len(adapter.calls) == 2
    assert first.summary.transcribed == 1
    assert second.summary.skipped == 1
    assert forced.summary.transcribed == 1
    assert len(list(iter_manifest(second.transcribed_manifest))) == 1


def test_transcribed_input_is_skipped(tmp_path: Path) -> None:
    make_wav(tmp_path / "segment.wav")
    input_manifest = tmp_path / "input.jsonl"
    write_manifest(
        input_manifest,
        [
            make_sample(
                "JP_DONE",
                "segment.wav",
                stage=ProcessingStage.TRANSCRIBED,
                asr_text="既存の文字起こし。",
            )
        ],
    )
    adapter = FakeASR()

    outputs = pipeline(tmp_path, adapter).run(input_manifest)

    assert adapter.calls == []
    assert outputs.summary.skipped == 1
    assert list(iter_manifest(outputs.transcribed_manifest))[0].text.asr_text == "既存の文字起こし。"


def test_missing_audio_fails_one_sample_and_batch_continues(tmp_path: Path) -> None:
    make_wav(tmp_path / "good.wav")
    input_manifest = tmp_path / "input.jsonl"
    write_manifest(
        input_manifest,
        [
            make_sample("JP_MISSING", "missing.wav"),
            make_sample("JP_GOOD", "good.wav"),
        ],
    )
    adapter = FakeASR()

    outputs = pipeline(tmp_path, adapter).run(input_manifest)
    samples = list(iter_manifest(outputs.transcribed_manifest))
    reports = list(iter_asr_run_reports(outputs.report_file))

    assert samples[0].state.stage is ProcessingStage.AUDIO_PREPARED
    assert samples[0].state.curation_decision is CurationDecision.DROP
    assert samples[0].state.decision_reasons == ["audio_missing"]
    assert samples[1].state.stage is ProcessingStage.TRANSCRIBED
    assert reports[0].status is ASRRunStatus.FAILED
    assert reports[0].errors[0].code == "audio_missing"
    assert reports[1].status is ASRRunStatus.TRANSCRIBED
    assert outputs.summary.failed == 1
    assert outputs.summary.transcribed == 1
    assert len(adapter.calls) == 1


def test_empty_transcript_is_sample_failure(tmp_path: Path) -> None:
    make_wav(tmp_path / "segment.wav")
    input_manifest = tmp_path / "input.jsonl"
    write_manifest(input_manifest, [make_sample("JP_EMPTY", "segment.wav")])

    outputs = pipeline(tmp_path, FakeASR(text="  ")).run(input_manifest)
    sample = list(iter_manifest(outputs.transcribed_manifest))[0]
    report = list(iter_asr_run_reports(outputs.report_file))[0]

    assert sample.state.stage is ProcessingStage.AUDIO_PREPARED
    assert sample.state.curation_decision is CurationDecision.DROP
    assert "asr_empty_transcript" in sample.state.decision_reasons
    assert report.errors[0].code == "asr_empty_transcript"
