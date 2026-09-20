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

    config = ASRConfig(
        download_root=os.environ.get("J_SPEECH_OPS_MODEL_CACHE"),
        local_files_only=os.environ.get("J_SPEECH_OPS_LOCAL_MODELS_ONLY") == "1",
    )
    adapter = FasterWhisperAdapter(config)
    result = adapter.transcribe(audio_path)

    assert result.text.strip()
    assert result.language == "ja"
    assert result.segments

