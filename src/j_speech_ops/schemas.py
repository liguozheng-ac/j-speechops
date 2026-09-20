"""Canonical Stage 1 data models for speech datasets."""

from __future__ import annotations

import math
from enum import Enum
from typing import Annotated, Any, Literal

from pydantic import BaseModel, ConfigDict, Field, JsonValue, StringConstraints, field_validator, model_validator

SCHEMA_VERSION = "1.0"
NonEmptyStr = Annotated[str, StringConstraints(strip_whitespace=True, min_length=1)]


def _validate_json_value(value: Any, path: str = "metadata") -> None:
    """Reject values that cannot be represented by standards-compliant JSON."""
    if value is None or isinstance(value, (str, bool, int)):
        return
    if isinstance(value, float):
        if not math.isfinite(value):
            raise ValueError(f"{path} contains a non-finite float")
        return
    if isinstance(value, list):
        for index, item in enumerate(value):
            _validate_json_value(item, f"{path}[{index}]")
        return
    if isinstance(value, dict):
        for key, item in value.items():
            if not isinstance(key, str):
                raise ValueError(f"{path} contains a non-string object key")
            _validate_json_value(item, f"{path}.{key}")
        return
    raise ValueError(f"{path} contains non-JSON value of type {type(value).__name__}")


class SchemaModel(BaseModel):
    """Shared strictness for canonical schema objects."""

    model_config = ConfigDict(extra="forbid", validate_assignment=True)


class ProcessingStage(str, Enum):
    INGESTED = "ingested"
    AUDIO_PREPARED = "audio_prepared"
    TRANSCRIBED = "transcribed"
    TEXT_PREPARED = "text_prepared"
    CURATED = "curated"


class CurationDecision(str, Enum):
    UNDECIDED = "undecided"
    PASS = "pass"
    REVIEW = "review"
    DROP = "drop"


class DatasetSplit(str, Enum):
    UNSPLIT = "unsplit"
    TRAIN = "train"
    VALIDATION = "validation"
    TEST = "test"
    BENCHMARK = "benchmark"


class RightsStatus(str, Enum):
    UNKNOWN = "unknown"
    CLEARED = "cleared"
    RESTRICTED = "restricted"


class AudioInfo(SchemaModel):
    path: NonEmptyStr
    format: NonEmptyStr | None = None
    duration_sec: float | None = Field(default=None, ge=0)
    sample_rate_hz: int | None = Field(default=None, gt=0)
    channels: int | None = Field(default=None, gt=0)


class TextInfo(SchemaModel):
    reference_text: str | None = None
    asr_text: str | None = None
    normalized_text: str | None = None
    reading_kana: str | None = None
    phonemes: list[str] | None = None


class SpeakerInfo(SchemaModel):
    speaker_id: NonEmptyStr | None = None


class SourceInfo(SchemaModel):
    source_name: NonEmptyStr
    source_record_id: NonEmptyStr | None = None
    source_url: NonEmptyStr | None = None
    license: NonEmptyStr | None = None
    rights_status: RightsStatus = RightsStatus.UNKNOWN


class LineageInfo(SchemaModel):
    parent_sample_id: NonEmptyStr | None = None
    segment_start_sec: float | None = Field(default=None, ge=0)
    segment_end_sec: float | None = Field(default=None, ge=0)

    @model_validator(mode="after")
    def validate_segment(self) -> "LineageInfo":
        start = self.segment_start_sec
        end = self.segment_end_sec
        if (start is None) != (end is None):
            raise ValueError("segment_start_sec and segment_end_sec must be provided together")
        if start is not None and self.parent_sample_id is None:
            raise ValueError("parent_sample_id is required for a segmented sample")
        if start is not None and end is not None and end < start:
            raise ValueError("segment_end_sec must be greater than or equal to segment_start_sec")
        return self


class ProcessingState(SchemaModel):
    stage: ProcessingStage = ProcessingStage.INGESTED
    curation_decision: CurationDecision = CurationDecision.UNDECIDED
    decision_reasons: list[str] = Field(default_factory=list)


class SpeechSample(SchemaModel):
    schema_version: Literal[SCHEMA_VERSION] = SCHEMA_VERSION
    sample_id: NonEmptyStr
    dataset_id: NonEmptyStr
    language: NonEmptyStr = "ja"
    locale: NonEmptyStr | None = "ja-JP"
    audio: AudioInfo
    text: TextInfo = Field(default_factory=TextInfo)
    speaker: SpeakerInfo = Field(default_factory=SpeakerInfo)
    source: SourceInfo
    lineage: LineageInfo = Field(default_factory=LineageInfo)
    split: DatasetSplit = DatasetSplit.UNSPLIT
    state: ProcessingState = Field(default_factory=ProcessingState)
    metadata: dict[str, JsonValue] = Field(default_factory=dict)

    @field_validator("metadata", mode="before")
    @classmethod
    def metadata_must_be_json_compatible(cls, value: Any) -> Any:
        _validate_json_value(value)
        return value

    @model_validator(mode="after")
    def validate_lineage_identity(self) -> "SpeechSample":
        if self.lineage.parent_sample_id == self.sample_id:
            raise ValueError("parent_sample_id must differ from sample_id")
        return self


class DatasetMetadata(SchemaModel):
    schema_version: Literal[SCHEMA_VERSION] = SCHEMA_VERSION
    dataset_id: NonEmptyStr
    name: NonEmptyStr
    version: NonEmptyStr
    description: str | None = None
    default_language: NonEmptyStr = "ja"
    license: NonEmptyStr | None = None
    source: SourceInfo
    manifest: NonEmptyStr
    metadata: dict[str, JsonValue] = Field(default_factory=dict)

    @field_validator("metadata", mode="before")
    @classmethod
    def metadata_must_be_json_compatible(cls, value: Any) -> Any:
        _validate_json_value(value)
        return value

