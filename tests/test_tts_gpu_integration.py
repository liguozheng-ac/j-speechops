from __future__ import annotations

import os
from pathlib import Path

import pytest

from j_speech_ops.tts_synthesis import (
    Qwen3TTSRuntimeAdapter,
    RuntimeBatchJob,
    RuntimeSynthesisRequest,
    validate_generated_wav,
)


@pytest.mark.stage8_gpu_integration
@pytest.mark.skipif(
    os.environ.get("J_SPEECH_OPS_RUN_STAGE8_GPU") != "1",
    reason="set J_SPEECH_OPS_RUN_STAGE8_GPU=1 for the Stage 8 Qwen GPU test",
)
def test_qwen3_tts_stage8_worker_generates_japanese_wav(tmp_path: Path) -> None:
    project_root = Path(__file__).resolve().parents[1]
    tts_python = project_root / ".venv-tts" / "Scripts" / "python.exe"
    model_path = Path(
        os.environ.get(
            "J_SPEECH_OPS_QWEN_MODEL_PATH",
            r"D:\AI-Models\Qwen3-TTS\Qwen3-TTS-12Hz-1.7B-CustomVoice",
        )
    )
    output = tmp_path / "stage8_gpu.wav"
    adapter = Qwen3TTSRuntimeAdapter(
        python_executable=tts_python,
        worker_script=project_root / "runtime" / "qwen3_tts_worker.py",
    )
    result = adapter.synthesize(
        RuntimeBatchJob(
            backend="qwen3-tts",
            model_name="Qwen/Qwen3-TTS-12Hz-1.7B-CustomVoice",
            model_path=str(model_path),
            speaker="Ono_Anna",
            language="Japanese",
            device="cuda:0",
            dtype="bfloat16",
            attention_implementation="sdpa",
            generation_parameters={},
            requests=(
                RuntimeSynthesisRequest(
                    run_id="a" * 64,
                    sample_id="stage8-gpu",
                    synthesis_text="動作確認です。",
                    seed=20260921,
                    output_audio_path=str(output),
                ),
            ),
        ),
        output_root=tmp_path,
    )
    assert result.results[0].status.value == "success"
    validation = validate_generated_wav(output)
    assert validation.sample_rate_hz > 0
    assert validation.duration_sec > 0
