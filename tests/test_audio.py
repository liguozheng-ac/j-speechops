from pathlib import Path

import numpy as np
import pytest
import soundfile as sf
import torch

from j_speech_ops.audio import (
    AudioValidationError,
    decode_audio,
    prepare_audio,
    write_pcm16_wav,
)


def test_valid_mono_wav_load_and_duration(tmp_path: Path) -> None:
    path = tmp_path / "mono.wav"
    sf.write(path, np.zeros(8_000, dtype=np.float32), 16_000, subtype="PCM_16")

    decoded = decode_audio(path)

    assert decoded.waveform.shape == (1, 8_000)
    assert decoded.waveform.dtype is torch.float32
    assert decoded.properties.sample_rate_hz == 16_000
    assert decoded.properties.channels == 1
    assert decoded.properties.duration_sec == pytest.approx(0.5)


def test_missing_file_is_structured_failure(tmp_path: Path) -> None:
    with pytest.raises(AudioValidationError) as captured:
        decode_audio(tmp_path / "missing.wav")
    assert captured.value.code == "audio_missing"


def test_corrupted_wav_is_structured_failure(tmp_path: Path) -> None:
    path = tmp_path / "corrupt.wav"
    path.write_bytes(b"not a wav")
    with pytest.raises(AudioValidationError) as captured:
        decode_audio(path)
    assert captured.value.code == "audio_decode_failed"


def test_empty_wav_is_structured_failure(tmp_path: Path) -> None:
    path = tmp_path / "empty.wav"
    sf.write(path, np.empty(0, dtype=np.float32), 16_000, subtype="PCM_16")
    with pytest.raises(AudioValidationError) as captured:
        decode_audio(path)
    assert captured.value.code == "audio_empty"


def test_non_finite_audio_is_structured_failure(tmp_path: Path) -> None:
    path = tmp_path / "non-finite.wav"
    sf.write(path, np.array([0.0, np.nan, 0.0], dtype=np.float32), 16_000, subtype="FLOAT")
    with pytest.raises(AudioValidationError) as captured:
        decode_audio(path)
    assert captured.value.code == "audio_non_finite"


def test_stereo_is_averaged_to_mono(tmp_path: Path) -> None:
    path = tmp_path / "stereo.wav"
    stereo = np.column_stack(
        [np.full(1_600, 0.5, dtype=np.float32), np.full(1_600, -0.5, dtype=np.float32)]
    )
    sf.write(path, stereo, 16_000, subtype="FLOAT")

    prepared = prepare_audio(decode_audio(path))

    assert prepared.waveform.shape == (1_600,)
    assert torch.allclose(prepared.waveform, torch.zeros_like(prepared.waveform))
    assert prepared.properties.channels == 1


def test_44100_hz_is_resampled_to_16000_hz(tmp_path: Path) -> None:
    path = tmp_path / "source.wav"
    sf.write(path, np.zeros(44_100, dtype=np.float32), 44_100, subtype="PCM_16")

    prepared = prepare_audio(decode_audio(path))

    assert prepared.waveform.shape == (16_000,)
    assert prepared.properties.sample_rate_hz == 16_000
    assert prepared.properties.duration_sec == pytest.approx(1.0)


def test_output_is_pcm16_mono_wav(tmp_path: Path) -> None:
    path = tmp_path / "prepared.wav"
    waveform = torch.linspace(-0.5, 0.5, 1_600)
    write_pcm16_wav(path, waveform, 16_000)

    info = sf.info(path)
    assert info.format == "WAV"
    assert info.subtype == "PCM_16"
    assert info.samplerate == 16_000
    assert info.channels == 1
    assert info.frames == 1_600
