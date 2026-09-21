from __future__ import annotations

import os
import time
from pathlib import Path

import pytest

from j_speech_ops.asr import ASRConfig, FasterWhisperAdapter
from j_speech_ops.generated_audio_qa import (
    GeneratedAudioQAPipeline,
    detect_nvidia_gpu_name,
)
from j_speech_ops.japanese_normalization import JapaneseNormalizer
from j_speech_ops.japanese_reading import OpenJTalkReadingProvider


@pytest.mark.stage9_gpu_integration
@pytest.mark.skipif(
    os.environ.get("J_SPEECH_OPS_RUN_STAGE9_GPU") != "1",
    reason="set J_SPEECH_OPS_RUN_STAGE9_GPU=1 for Stage 9 GPU integration",
)
def test_stage9_round_trip_on_existing_generated_audio(tmp_path: Path) -> None:
    model_value = os.environ.get("J_SPEECH_OPS_WHISPER_MODEL_PATH")
    manifest_value = os.environ.get("J_SPEECH_OPS_STAGE8_AUDIO_MANIFEST")
    if not model_value or not Path(model_value).is_dir():
        pytest.skip("local Whisper model path is unavailable")
    if not manifest_value or not Path(manifest_value).is_file():
        pytest.skip("Stage 8 generated-audio manifest is unavailable")

    config = ASRConfig(model_path=model_value, local_files_only=True)
    model_started = time.perf_counter()
    adapter = FasterWhisperAdapter(config)
    load_time = time.perf_counter() - model_started
    outputs = GeneratedAudioQAPipeline(
        output_root=tmp_path / "stage9",
        adapter=adapter,
        normalizer=JapaneseNormalizer(),
        reading_provider=OpenJTalkReadingProvider(),
        asr_model_load_time_sec=load_time,
        gpu_name=detect_nvidia_gpu_name(config.device_index),
    ).run(Path(manifest_value), force=True)

    assert outputs.summary.total_samples > 0
    assert outputs.summary.completed_samples == outputs.summary.total_samples
    assert outputs.summary.failed_samples == 0
    assert outputs.summary.dropped_samples == 0
    assert outputs.summary.passed_samples + outputs.summary.review_samples == outputs.summary.total_samples
    assert outputs.summary.asr_model_load_time_sec > 0
