"""Stage 2 audio decoding, validation, channel conversion, resampling, and WAV output."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import soundfile as sf
import torch
import torchaudio.functional as audio_functional
from torch import Tensor


@dataclass(frozen=True)
class AudioProperties:
    sample_rate_hz: int
    channels: int
    duration_sec: float


@dataclass(frozen=True)
class DecodedAudio:
    """A decoded channel-first float32 waveform and its source properties."""

    waveform: Tensor
    properties: AudioProperties


@dataclass(frozen=True)
class PreparedAudio:
    """A canonical mono float32 waveform ready for VAD."""

    waveform: Tensor
    properties: AudioProperties


class AudioValidationError(ValueError):
    """An expected per-sample audio failure with a stable reason code."""

    def __init__(self, code: str, message: str) -> None:
        self.code = code
        super().__init__(message)


def decode_audio(path: Path) -> DecodedAudio:
    """Decode an audio file through SoundFile and perform data-level validation."""
    if not path.exists():
        raise AudioValidationError("audio_missing", f"audio file does not exist: {path}")
    if not path.is_file():
        raise AudioValidationError("audio_unreadable", f"audio path is not a file: {path}")

    try:
        data, sample_rate_hz = sf.read(path, dtype="float32", always_2d=True)
    except (OSError, RuntimeError, sf.SoundFileError) as exc:
        raise AudioValidationError("audio_decode_failed", f"SoundFile could not decode {path}: {exc}") from exc

    if sample_rate_hz <= 0:
        raise AudioValidationError("audio_invalid_sample_rate", "sample rate must be greater than zero")
    if data.ndim != 2 or data.shape[1] <= 0:
        raise AudioValidationError("audio_invalid_channels", "channel count must be greater than zero")
    if data.shape[0] <= 0:
        raise AudioValidationError("audio_empty", "decoded audio contains no frames")

    waveform = torch.from_numpy(data).transpose(0, 1).contiguous().to(dtype=torch.float32)
    if not bool(torch.isfinite(waveform).all()):
        raise AudioValidationError("audio_non_finite", "decoded audio contains NaN or infinity")

    duration_sec = waveform.shape[1] / sample_rate_hz
    if duration_sec <= 0:
        raise AudioValidationError("audio_empty", "decoded audio duration must be greater than zero")

    return DecodedAudio(
        waveform=waveform,
        properties=AudioProperties(
            sample_rate_hz=sample_rate_hz,
            channels=waveform.shape[0],
            duration_sec=duration_sec,
        ),
    )


def prepare_audio(decoded: DecodedAudio, target_sample_rate_hz: int = 16_000) -> PreparedAudio:
    """Average channels to mono, then resample to the canonical sample rate."""
    if target_sample_rate_hz <= 0:
        raise ValueError("target_sample_rate_hz must be greater than zero")

    mono = decoded.waveform.mean(dim=0)
    if decoded.properties.sample_rate_hz != target_sample_rate_hz:
        mono = audio_functional.resample(
            mono,
            orig_freq=decoded.properties.sample_rate_hz,
            new_freq=target_sample_rate_hz,
        )
    mono = mono.contiguous().to(dtype=torch.float32, device="cpu")
    if mono.numel() == 0:
        raise AudioValidationError("audio_empty", "audio became empty during preparation")
    if not bool(torch.isfinite(mono).all()):
        raise AudioValidationError("audio_non_finite", "prepared audio contains NaN or infinity")

    return PreparedAudio(
        waveform=mono,
        properties=AudioProperties(
            sample_rate_hz=target_sample_rate_hz,
            channels=1,
            duration_sec=mono.numel() / target_sample_rate_hz,
        ),
    )


def write_pcm16_wav(path: Path, waveform: Tensor, sample_rate_hz: int) -> None:
    """Write a mono waveform as PCM_16 WAV through SoundFile."""
    if waveform.ndim != 1 or waveform.numel() == 0:
        raise ValueError("output waveform must be a non-empty mono tensor")
    if sample_rate_hz <= 0:
        raise ValueError("sample_rate_hz must be greater than zero")
    path.parent.mkdir(parents=True, exist_ok=True)
    sf.write(
        path,
        waveform.detach().to(dtype=torch.float32, device="cpu").numpy(),
        sample_rate_hz,
        format="WAV",
        subtype="PCM_16",
    )

