"""Deterministic UTF-8 line ingestion for canonical TTS text samples."""

from __future__ import annotations

import hashlib
import re
from dataclasses import dataclass
from pathlib import Path

from pydantic import BaseModel, ConfigDict, field_validator

from .schemas import RightsStatus
from .text_manifest import write_text_manifest
from .tts_text import (
    TTSIntendedUse,
    TTSTextContent,
    TTSTextSample,
    TTSTextSource,
    TextSourceType,
)


class TextIngestionConfig(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    dataset_id: str
    language: str = "ja"
    locale: str | None = "ja-JP"
    source_name: str
    source_type: TextSourceType = TextSourceType.FILE
    rights_status: RightsStatus = RightsStatus.UNKNOWN
    intended_use: TTSIntendedUse = TTSIntendedUse.TTS_PRODUCTION
    domain: str | None = None
    id_prefix: str | None = None

    @field_validator("dataset_id", "language", "source_name")
    @classmethod
    def required_strings_must_not_be_blank(cls, value: str) -> str:
        if not value.strip():
            raise ValueError("value must not be blank")
        return value.strip()

    @field_validator("locale", "domain", "id_prefix")
    @classmethod
    def optional_strings_must_not_be_blank(cls, value: str | None) -> str | None:
        if value is not None and not value.strip():
            raise ValueError("value must not be blank")
        return value.strip() if value is not None else None


class TextIngestionError(ValueError):
    def __init__(self, path: Path, detail: str, line_number: int | None = None) -> None:
        self.path = path
        self.line_number = line_number
        self.detail = detail
        location = f"{path}:{line_number}" if line_number is not None else str(path)
        super().__init__(f"{location}: {detail}")


@dataclass(frozen=True)
class TextIngestionResult:
    input_path: Path
    output_manifest: Path
    sample_count: int
    samples: tuple[TTSTextSample, ...]


def _id_prefix(config: TextIngestionConfig) -> str:
    basis = config.id_prefix or config.dataset_id
    prefix = re.sub(r"[^A-Za-z0-9]+", "_", basis).strip("_").upper()
    if prefix:
        return prefix
    digest = hashlib.sha256(config.dataset_id.encode("utf-8")).hexdigest()[:8].upper()
    return f"TEXT_{digest}"


def ingest_text_file(
    input_path: Path, config: TextIngestionConfig
) -> tuple[TTSTextSample, ...]:
    path = Path(input_path)
    samples: list[TTSTextSample] = []
    prefix = _id_prefix(config)
    try:
        handle = path.open("r", encoding="utf-8", newline=None)
    except (OSError, UnicodeError) as exc:
        raise TextIngestionError(path, f"cannot read UTF-8 input: {exc}") from exc

    try:
        with handle:
            for line_number, line in enumerate(handle, start=1):
                raw_text = line.removesuffix("\n")
                if not raw_text.strip():
                    raise TextIngestionError(
                        path,
                        "empty or whitespace-only input line",
                        line_number,
                    )
                sample_id = f"{prefix}_{line_number:06d}"
                samples.append(
                    TTSTextSample(
                        sample_id=sample_id,
                        dataset_id=config.dataset_id,
                        language=config.language,
                        locale=config.locale,
                        text=TTSTextContent(raw_text=raw_text),
                        source=TTSTextSource(
                            source_type=config.source_type,
                            source_name=config.source_name,
                            source_id=f"line:{line_number}",
                            source_path=path.as_posix(),
                        ),
                        rights_status=config.rights_status,
                        intended_use=config.intended_use,
                        domain=config.domain,
                    )
                )
    except (OSError, UnicodeError) as exc:
        raise TextIngestionError(path, f"cannot read UTF-8 input: {exc}") from exc

    if not samples:
        raise TextIngestionError(path, "input contains no text samples")
    return tuple(samples)


def ingest_text_to_manifest(
    input_path: Path,
    output_manifest: Path,
    config: TextIngestionConfig,
) -> TextIngestionResult:
    samples = ingest_text_file(input_path, config)
    write_text_manifest(output_manifest, samples)
    return TextIngestionResult(
        input_path=Path(input_path),
        output_manifest=Path(output_manifest),
        sample_count=len(samples),
        samples=samples,
    )
