"""A narrow Silero VAD boundary for Stage 2."""

from __future__ import annotations

from dataclasses import dataclass
from importlib.metadata import version
from typing import Protocol

from pydantic import BaseModel, ConfigDict, Field
from torch import Tensor


class VADConfig(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    threshold: float = Field(default=0.5, ge=0, le=1)
    min_speech_duration_ms: int = Field(default=250, gt=0)
    min_silence_duration_ms: int = Field(default=100, gt=0)
    speech_pad_ms: int = Field(default=30, ge=0)
    max_speech_duration_s: float = Field(default=30.0, gt=0)


@dataclass(frozen=True)
class SpeechRegion:
    """A half-open speech interval addressed by sample indices."""

    start_sample: int
    end_sample: int
    sample_rate_hz: int = 16_000

    def __post_init__(self) -> None:
        if self.start_sample < 0:
            raise ValueError("start_sample must be non-negative")
        if self.end_sample <= self.start_sample:
            raise ValueError("end_sample must be greater than start_sample")
        if self.sample_rate_hz <= 0:
            raise ValueError("sample_rate_hz must be greater than zero")

    @property
    def start_sec(self) -> float:
        return self.start_sample / self.sample_rate_hz

    @property
    def end_sec(self) -> float:
        return self.end_sample / self.sample_rate_hz

    @property
    def duration_sec(self) -> float:
        return (self.end_sample - self.start_sample) / self.sample_rate_hz


class VoiceActivityDetector(Protocol):
    backend_name: str
    backend_version: str

    def detect(self, waveform: Tensor, sample_rate_hz: int) -> list[SpeechRegion]: ...


class SileroVAD:
    """CPU-backed adapter around the installed silero-vad pip package."""

    backend_name = "silero-vad"

    def __init__(self, config: VADConfig) -> None:
        from silero_vad import load_silero_vad

        self.config = config
        self.backend_version = version("silero-vad")
        self._model = load_silero_vad(onnx=False)
        self._model.eval()

    def detect(self, waveform: Tensor, sample_rate_hz: int) -> list[SpeechRegion]:
        from silero_vad import get_speech_timestamps

        if waveform.ndim != 1:
            raise ValueError("SileroVAD requires a one-dimensional mono waveform")
        if sample_rate_hz != 16_000:
            raise ValueError("Stage 2 SileroVAD requires 16 kHz audio")
        if waveform.device.type != "cpu":
            raise ValueError("Stage 2 SileroVAD is CPU-only")

        timestamps = get_speech_timestamps(
            waveform,
            self._model,
            threshold=self.config.threshold,
            sampling_rate=sample_rate_hz,
            min_speech_duration_ms=self.config.min_speech_duration_ms,
            max_speech_duration_s=self.config.max_speech_duration_s,
            min_silence_duration_ms=self.config.min_silence_duration_ms,
            speech_pad_ms=self.config.speech_pad_ms,
            return_seconds=False,
        )
        return [
            SpeechRegion(
                start_sample=int(timestamp["start"]),
                end_sample=int(timestamp["end"]),
                sample_rate_hz=sample_rate_hz,
            )
            for timestamp in timestamps
        ]

