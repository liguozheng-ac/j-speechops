"""Model-neutral ASR contracts and the faster-whisper Stage 3 adapter."""

from __future__ import annotations

import os
import sys
from collections.abc import Callable
from importlib.metadata import version
from pathlib import Path
from typing import Any, Literal, Protocol

import ctranslate2
from faster_whisper import WhisperModel
from pydantic import BaseModel, ConfigDict, Field, JsonValue, model_validator


class ASRConfig(BaseModel):
    """Reproducible Japanese large-v3 baseline configuration."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    model_name: Literal["large-v3"] = "large-v3"
    device: Literal["cuda"] = "cuda"
    device_index: int = Field(default=0, ge=0)
    compute_type: Literal["float16"] = "float16"
    language: Literal["ja"] = "ja"
    beam_size: int = Field(default=5, gt=0)
    vad_filter: Literal[False] = False
    condition_on_previous_text: bool = False
    word_timestamps: bool = False
    temperature: float = Field(default=0.0, ge=0)
    download_root: str | None = None
    local_files_only: bool = False


class ASRSegment(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    start_sec: float = Field(ge=0)
    end_sec: float = Field(gt=0)
    text: str
    avg_logprob: float | None = None
    no_speech_probability: float | None = Field(default=None, ge=0, le=1)

    @model_validator(mode="after")
    def validate_interval(self) -> "ASRSegment":
        if self.end_sec < self.start_sec:
            raise ValueError("end_sec must be greater than or equal to start_sec")
        return self


class ASRResult(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    text: str
    language: str = Field(min_length=1)
    language_probability: float | None = Field(default=None, ge=0, le=1)
    duration_sec: float | None = Field(default=None, ge=0)
    segments: list[ASRSegment] = Field(default_factory=list)


class ASRRuntimeInfo(BaseModel):
    """Model-neutral operational identity exposed to batch/report layers."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    backend_name: str = Field(min_length=1)
    backend_version: str = Field(min_length=1)
    model_name: str = Field(min_length=1)
    device: str = Field(min_length=1)
    compute_type: str = Field(min_length=1)
    language: str = Field(min_length=1)
    effective_config: dict[str, JsonValue]


class ASRAdapter(Protocol):
    @property
    def runtime_info(self) -> ASRRuntimeInfo: ...

    def transcribe(self, audio_path: Path) -> ASRResult: ...


class ASRInfrastructureError(RuntimeError):
    """Fatal ASR environment or model initialization failure."""


def _prepare_windows_cuda_runtime() -> list[Any]:
    """Expose CUDA wheel DLL directories to the current Python process only."""
    if os.name != "nt":
        return []

    site_packages = Path(sys.prefix) / "Lib" / "site-packages"
    candidates = (
        site_packages / "nvidia" / "cublas" / "bin",
        site_packages / "nvidia" / "cudnn" / "bin",
        site_packages / "nvidia" / "cuda_nvrtc" / "bin",
    )
    missing = [path for path in candidates[:2] if not path.is_dir()]
    if missing:
        formatted = ", ".join(str(path) for path in missing)
        raise ASRInfrastructureError(f"required CUDA DLL directories are missing: {formatted}")
    available = [path for path in candidates if path.is_dir()]
    current_path = os.environ.get("PATH", "")
    prefixes = [str(path) for path in available]
    os.environ["PATH"] = os.pathsep.join([*prefixes, current_path])
    return [os.add_dll_directory(str(path)) for path in available]


class FasterWhisperAdapter:
    """Japanese ASR adapter backed by one CUDA WhisperModel instance."""

    backend_name = "faster-whisper"

    def __init__(
        self,
        config: ASRConfig,
        *,
        model_factory: Callable[..., Any] = WhisperModel,
    ) -> None:
        self.config = config
        self.backend_version = version("faster-whisper")
        self.ctranslate2_version = version("ctranslate2")
        self._dll_handles = _prepare_windows_cuda_runtime()
        self._validate_cuda()
        try:
            self._model = model_factory(
                config.model_name,
                device=config.device,
                device_index=config.device_index,
                compute_type=config.compute_type,
                download_root=config.download_root,
                local_files_only=config.local_files_only,
            )
        except Exception as exc:
            raise ASRInfrastructureError(
                f"failed to initialize {config.model_name} on CUDA device "
                f"{config.device_index} with {config.compute_type}: {exc}"
            ) from exc

    def _validate_cuda(self) -> None:
        device_count = ctranslate2.get_cuda_device_count()
        if device_count <= self.config.device_index:
            raise ASRInfrastructureError(
                f"CUDA device {self.config.device_index} is unavailable; "
                f"CTranslate2 detected {device_count} device(s)"
            )
        supported = ctranslate2.get_supported_compute_types(
            "cuda", self.config.device_index
        )
        if self.config.compute_type not in supported:
            raise ASRInfrastructureError(
                f"CUDA device {self.config.device_index} does not support "
                f"compute_type={self.config.compute_type}; supported={sorted(supported)}"
            )

    @property
    def runtime_info(self) -> ASRRuntimeInfo:
        return ASRRuntimeInfo(
            backend_name=self.backend_name,
            backend_version=self.backend_version,
            model_name=self.config.model_name,
            device=self.config.device,
            compute_type=self.config.compute_type,
            language=self.config.language,
            effective_config=self.config.model_dump(mode="json"),
        )

    def transcribe(self, audio_path: Path) -> ASRResult:
        try:
            raw_segments, info = self._model.transcribe(
                str(audio_path),
                language=self.config.language,
                beam_size=self.config.beam_size,
                vad_filter=False,
                condition_on_previous_text=self.config.condition_on_previous_text,
                word_timestamps=self.config.word_timestamps,
                temperature=self.config.temperature,
            )
            segments = [self._map_segment(segment) for segment in raw_segments]
        except RuntimeError as exc:
            raise ASRInfrastructureError(
                f"CUDA inference failed for {audio_path.name}: {exc}"
            ) from exc

        text = "".join(segment.text for segment in segments).strip()
        return ASRResult(
            text=text,
            language=getattr(info, "language", self.config.language),
            language_probability=getattr(info, "language_probability", None),
            duration_sec=getattr(info, "duration", None),
            segments=segments,
        )

    @staticmethod
    def _map_segment(segment: Any) -> ASRSegment:
        return ASRSegment(
            start_sec=float(segment.start),
            end_sec=float(segment.end),
            text=str(segment.text),
            avg_logprob=getattr(segment, "avg_logprob", None),
            no_speech_probability=getattr(segment, "no_speech_prob", None),
        )
