"""Canonical text-data contract for the independent TTS Production Pipeline."""

from __future__ import annotations

import math
from enum import Enum
from typing import Annotated, Any, Literal

from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    JsonValue,
    StringConstraints,
    field_validator,
)

from .schemas import CurationDecision, RightsStatus

TTS_TEXT_SCHEMA_VERSION = "1.0"
NonEmptyStr = Annotated[str, StringConstraints(strip_whitespace=True, min_length=1)]


def _validate_json_value(value: Any, path: str = "metadata") -> None:
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


class TTSTextModel(BaseModel):
    """Strict immutable base; later stages create validated copies."""

    model_config = ConfigDict(extra="forbid", frozen=True)


class TextSourceType(str, Enum):
    MANUAL = "manual"
    FILE = "file"
    DATASET = "dataset"
    APPLICATION = "application"
    GENERATED = "generated"
    UNKNOWN = "unknown"


class TTSIntendedUse(str, Enum):
    TTS_PRODUCTION = "tts_production"
    BENCHMARK = "benchmark"
    DEMO = "demo"
    APPLICATION = "application"
    UNKNOWN = "unknown"


class TTSTextStage(str, Enum):
    RAW = "raw"
    NORMALIZED = "normalized"
    READING_PREPARED = "reading_prepared"
    CURATED = "curated"


class TTSTextContent(TTSTextModel):
    raw_text: str = Field(min_length=1)
    normalized_text: str | None = None
    reading_kana: str | None = None
    phonemes: tuple[str, ...] | None = None

    @field_validator("raw_text")
    @classmethod
    def raw_text_must_have_visible_content(cls, value: str) -> str:
        if not value.strip():
            raise ValueError("raw_text must not be empty or whitespace-only")
        return value


class TTSTextSource(TTSTextModel):
    source_type: TextSourceType = TextSourceType.UNKNOWN
    source_name: NonEmptyStr
    source_id: NonEmptyStr | None = None
    source_path: NonEmptyStr | None = None


class TTSTextState(TTSTextModel):
    stage: TTSTextStage = TTSTextStage.RAW
    curation_decision: CurationDecision = CurationDecision.UNDECIDED
    decision_reasons: tuple[str, ...] = ()


class TTSTextSample(TTSTextModel):
    """Canonical input text, deliberately unrelated to SpeechSample inheritance."""

    schema_version: Literal[TTS_TEXT_SCHEMA_VERSION] = TTS_TEXT_SCHEMA_VERSION
    sample_id: NonEmptyStr
    dataset_id: NonEmptyStr
    language: NonEmptyStr = "ja"
    locale: NonEmptyStr | None = "ja-JP"
    text: TTSTextContent
    source: TTSTextSource
    rights_status: RightsStatus = RightsStatus.UNKNOWN
    intended_use: TTSIntendedUse = TTSIntendedUse.UNKNOWN
    domain: NonEmptyStr | None = None
    state: TTSTextState = Field(default_factory=TTSTextState)
    metadata: dict[str, JsonValue] = Field(default_factory=dict)

    @field_validator("metadata", mode="before")
    @classmethod
    def metadata_must_be_json_compatible(cls, value: Any) -> Any:
        _validate_json_value(value)
        return value


_ALLOWED_TEXT_TRANSITIONS = {
    TTSTextStage.RAW: {TTSTextStage.NORMALIZED},
    TTSTextStage.NORMALIZED: {TTSTextStage.READING_PREPARED},
    TTSTextStage.READING_PREPARED: {TTSTextStage.CURATED},
    TTSTextStage.CURATED: set(),
}


class TTSTextLifecycleTransitionError(ValueError):
    """Raised when a text-processing transition is not an explicit forward edge."""


def transition_text_stage(
    sample: TTSTextSample, target_stage: TTSTextStage
) -> TTSTextSample:
    target_stage = TTSTextStage(target_stage)
    current_stage = sample.state.stage
    if target_stage is current_stage:
        return sample.model_copy(deep=True)
    if target_stage not in _ALLOWED_TEXT_TRANSITIONS[current_stage]:
        raise TTSTextLifecycleTransitionError(
            f"invalid TTS text transition: {current_stage.value} -> "
            f"{target_stage.value}"
        )
    next_state = TTSTextState(
        stage=target_stage,
        curation_decision=sample.state.curation_decision,
        decision_reasons=sample.state.decision_reasons,
    )
    return sample.model_copy(update={"state": next_state}, deep=True)
