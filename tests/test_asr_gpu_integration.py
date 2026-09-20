import os
from pathlib import Path

import pytest

from j_speech_ops.asr import ASRConfig, FasterWhisperAdapter


@pytest.mark.gpu_integration
def test_large_v3_cuda_on_real_japanese_speech() -> None:
    audio_value = os.environ.get("J_SPEECH_OPS_JA_SMOKE_WAV")
    if not audio_value:
        pytest.skip("set J_SPEECH_OPS_JA_SMOKE_WAV to a real Japanese WAV")
    audio_path = Path(audio_value)
    if not audio_path.is_file():
        pytest.skip(f"Japanese smoke WAV is unavailable: {audio_path}")
    model_value = os.environ.get("J_SPEECH_OPS_WHISPER_MODEL_PATH")
    if not model_value:
        pytest.skip(
            "set J_SPEECH_OPS_WHISPER_MODEL_PATH to a local faster-whisper model"
        )
    model_path = Path(model_value)
    if not model_path.is_dir():
        pytest.skip(f"local Whisper model is unavailable: {model_path}")

    config = ASRConfig(
        model_path=str(model_path),
        local_files_only=True,
    )
    adapter = FasterWhisperAdapter(config)
    result = adapter.transcribe(audio_path)

    assert result.text.strip()
    assert result.language == "ja"
    assert result.segments
