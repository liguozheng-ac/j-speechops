from __future__ import annotations

import hashlib
from pathlib import Path

import numpy as np
import pytest
import soundfile as sf
from torch import Tensor

from j_speech_ops.audio_preparation import (
    AudioPreparationPipeline,
    AudioPreparationStatus,
    iter_audio_preparation_reports,
)
from j_speech_ops.manifest import iter_manifest, write_manifest
from j_speech_ops.schemas import CurationDecision, ProcessingStage, SpeechSample
from j_speech_ops.vad import SpeechRegion


class FakeVAD:
    backend_name = "fake-vad"
    backend_version = "test"

    def __init__(self, regions: list[SpeechRegion]) -> None:
        self.regions = regions
        self.calls = 0

    def detect(self, waveform: Tensor, sample_rate_hz: int) -> list[SpeechRegion]:
        assert waveform.ndim == 1
        assert sample_rate_hz == 16_000
        self.calls += 1
        return list(self.regions)


def make_wav(path: Path, *, sample_rate: int = 44_100, channels: int = 2) -> None:
    frames = sample_rate
    mono = (0.1 * np.sin(2 * np.pi * 220 * np.arange(frames) / sample_rate)).astype(np.float32)
    data = mono if channels == 1 else np.column_stack([mono, mono * 0.5])
    path.parent.mkdir(parents=True, exist_ok=True)
    sf.write(path, data, sample_rate, subtype="PCM_16")


def make_parent(sample_id: str, audio_path: str) -> SpeechSample:
    return SpeechSample.model_validate(
        {
            "sample_id": sample_id,
            "dataset_id": "stage2-test",
            "audio": {"path": audio_path},
            "text": {"reference_text": "親サンプルの全文です。"},
            "speaker": {"speaker_id": "speaker-1"},
            "source": {
                "source_name": "test-source",
                "source_record_id": "record-1",
                "license": "test-license",
                "rights_status": "cleared",
            },
        }
    )


def build_pipeline(tmp_path: Path, vad: FakeVAD) -> AudioPreparationPipeline:
    return AudioPreparationPipeline(
        dataset_root=tmp_path,
        output_root=tmp_path / "data",
        vad=vad,
    )


def test_one_region_creates_deterministic_child_with_lineage(tmp_path: Path) -> None:
    raw = tmp_path / "raw" / "JP_0001.wav"
    make_wav(raw)
    before = hashlib.sha256(raw.read_bytes()).hexdigest()
    input_manifest = tmp_path / "input.jsonl"
    write_manifest(input_manifest, [make_parent("JP_0001", "raw/JP_0001.wav")])
    pipeline = build_pipeline(tmp_path, FakeVAD([SpeechRegion(1_600, 8_000)]))

    outputs = pipeline.run(input_manifest)
    parents = list(iter_manifest(outputs.source_manifest))
    children = list(iter_manifest(outputs.segment_manifest))
    reports = list(iter_audio_preparation_reports(outputs.report_file))

    assert hashlib.sha256(raw.read_bytes()).hexdigest() == before
    assert parents[0].state.stage is ProcessingStage.AUDIO_PREPARED
    assert children[0].sample_id == "JP_0001__seg0001"
    assert children[0].audio.path == "data/interim/audio_segments/JP_0001__seg0001.wav"
    assert children[0].audio.duration_sec == pytest.approx(0.4)
    assert children[0].lineage.parent_sample_id == "JP_0001"
    assert children[0].lineage.segment_start_sec == pytest.approx(0.1)
    assert children[0].lineage.segment_end_sec == pytest.approx(0.5)
    assert children[0].source == parents[0].source
    assert children[0].text.reference_text is None
    assert children[0].state.stage is ProcessingStage.AUDIO_PREPARED
    assert children[0].state.curation_decision is CurationDecision.UNDECIDED
    assert reports[0].status is AudioPreparationStatus.PREPARED
    assert reports[0].speech_ratio == pytest.approx(0.4)
    assert outputs.summary.segments_created == 1

    segment_info = sf.info(tmp_path / children[0].audio.path)
    assert segment_info.samplerate == 16_000
    assert segment_info.channels == 1
    assert segment_info.subtype == "PCM_16"


def test_multiple_regions_and_rerun_do_not_duplicate_manifest_rows(tmp_path: Path) -> None:
    make_wav(tmp_path / "raw.wav", sample_rate=16_000, channels=1)
    input_manifest = tmp_path / "input.jsonl"
    write_manifest(input_manifest, [make_parent("JP_0002", "raw.wav")])
    pipeline = build_pipeline(
        tmp_path,
        FakeVAD([SpeechRegion(0, 4_000), SpeechRegion(8_000, 16_000)]),
    )

    first = pipeline.run(input_manifest)
    second = pipeline.run(input_manifest)

    children = list(iter_manifest(second.segment_manifest))
    assert [child.sample_id for child in children] == [
        "JP_0002__seg0001",
        "JP_0002__seg0002",
    ]
    assert len(list(iter_manifest(first.segment_manifest))) == 2
    assert len(list(iter_audio_preparation_reports(second.report_file))) == 1


def test_no_speech_marks_prepared_parent_dropped(tmp_path: Path) -> None:
    make_wav(tmp_path / "silence.wav", sample_rate=16_000, channels=1)
    input_manifest = tmp_path / "input.jsonl"
    write_manifest(input_manifest, [make_parent("JP_SILENCE", "silence.wav")])

    outputs = build_pipeline(tmp_path, FakeVAD([])).run(input_manifest)
    parent = list(iter_manifest(outputs.source_manifest))[0]
    report = list(iter_audio_preparation_reports(outputs.report_file))[0]

    assert parent.state.stage is ProcessingStage.AUDIO_PREPARED
    assert parent.state.curation_decision is CurationDecision.DROP
    assert "no_speech_detected" in parent.state.decision_reasons
    assert list(iter_manifest(outputs.segment_manifest)) == []
    assert report.status is AudioPreparationStatus.DROPPED
    assert report.original is not None
    assert report.prepared is not None
    assert report.warnings[0].code == "no_speech_detected"
    assert outputs.summary.errors == 0


def test_bad_audio_drops_one_sample_and_batch_continues(tmp_path: Path) -> None:
    corrupt = tmp_path / "corrupt.wav"
    corrupt.write_bytes(b"broken")
    make_wav(tmp_path / "good.wav", sample_rate=16_000, channels=1)
    input_manifest = tmp_path / "input.jsonl"
    write_manifest(
        input_manifest,
        [
            make_parent("JP_BAD", "corrupt.wav"),
            make_parent("JP_GOOD", "good.wav"),
        ],
    )
    pipeline = build_pipeline(tmp_path, FakeVAD([SpeechRegion(1_600, 8_000)]))

    outputs = pipeline.run(input_manifest)
    parents = list(iter_manifest(outputs.source_manifest))
    children = list(iter_manifest(outputs.segment_manifest))
    reports = list(iter_audio_preparation_reports(outputs.report_file))

    assert parents[0].state.stage is ProcessingStage.INGESTED
    assert parents[0].state.curation_decision is CurationDecision.DROP
    assert parents[0].state.decision_reasons == ["audio_decode_failed"]
    assert parents[1].state.stage is ProcessingStage.AUDIO_PREPARED
    assert [child.sample_id for child in children] == ["JP_GOOD__seg0001"]
    assert reports[0].errors[0].code == "audio_decode_failed"
    assert reports[1].status is AudioPreparationStatus.PREPARED
    assert outputs.summary.input_samples == 2
    assert outputs.summary.prepared == 1
    assert outputs.summary.dropped == 1
    assert outputs.summary.errors == 1


def test_missing_audio_preserves_ingested_stage(tmp_path: Path) -> None:
    input_manifest = tmp_path / "input.jsonl"
    write_manifest(input_manifest, [make_parent("JP_MISSING", "not-there.wav")])

    outputs = build_pipeline(tmp_path, FakeVAD([])).run(input_manifest)
    parent = list(iter_manifest(outputs.source_manifest))[0]
    report = list(iter_audio_preparation_reports(outputs.report_file))[0]

    assert parent.state.stage is ProcessingStage.INGESTED
    assert parent.state.curation_decision is CurationDecision.DROP
    assert parent.state.decision_reasons == ["audio_missing"]
    assert report.errors[0].code == "audio_missing"

