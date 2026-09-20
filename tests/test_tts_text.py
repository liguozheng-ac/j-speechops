from __future__ import annotations

import pytest
from pydantic import ValidationError

from j_speech_ops.schemas import CurationDecision, SpeechSample
from j_speech_ops.tts_text import (
    TTSIntendedUse,
    TTSTextContent,
    TTSTextLifecycleTransitionError,
    TTSTextSample,
    TTSTextSource,
    TTSTextStage,
    TextSourceType,
    transition_text_stage,
)


def make_sample(raw_text: str = "佐伯様、ご予約は15時30分でございます。") -> TTSTextSample:
    return TTSTextSample(
        sample_id="HOSPITALITY_000001",
        dataset_id="hospitality-demo",
        language="ja",
        locale="ja-JP",
        text=TTSTextContent(raw_text=raw_text),
        source=TTSTextSource(
            source_type=TextSourceType.MANUAL,
            source_name="stage5-test",
            source_id="line:1",
        ),
        rights_status="cleared",
        intended_use=TTSIntendedUse.TTS_PRODUCTION,
        domain="hospitality",
    )


@pytest.mark.parametrize(
    "raw_text",
    [
        "本日はよろしくお願いいたします。",
        "Wi-FiとChatGPTをご利用いただけます。",
        "9月21日の15時30分に、3,500円をお支払いください。",
        "全角記号！？とEmoji😊も保持します。",
    ],
)
def test_valid_unicode_text_is_preserved(raw_text: str) -> None:
    sample = make_sample(raw_text)
    assert sample.text.raw_text == raw_text
    assert sample.text.normalized_text is None
    assert sample.text.reading_kana is None
    assert sample.text.phonemes is None
    assert sample.state.stage is TTSTextStage.RAW
    assert sample.state.curation_decision is CurationDecision.UNDECIDED


@pytest.mark.parametrize("raw_text", ["", "   ", "\t\u3000"])
def test_empty_or_whitespace_raw_text_is_rejected(raw_text: str) -> None:
    with pytest.raises(ValidationError):
        TTSTextContent(raw_text=raw_text)


def test_raw_text_is_immutable() -> None:
    sample = make_sample()
    with pytest.raises(ValidationError):
        sample.text.raw_text = "上書き禁止"
    with pytest.raises(ValidationError):
        sample.text = TTSTextContent(raw_text="差し替え禁止")


def test_text_contract_is_independent_from_speech_contract() -> None:
    assert not issubclass(TTSTextSample, SpeechSample)
    fields = set(TTSTextSample.model_fields)
    assert "audio" not in fields
    assert "speaker" not in fields
    assert "model_name" not in fields
    assert "voice_id" not in fields
    assert "output_audio_path" not in fields
    assert "asr_text" not in TTSTextContent.model_fields


def test_text_lifecycle_allows_only_explicit_forward_edges() -> None:
    raw = make_sample()
    normalized = transition_text_stage(raw, TTSTextStage.NORMALIZED)
    reading = transition_text_stage(normalized, TTSTextStage.READING_PREPARED)
    curated = transition_text_stage(reading, TTSTextStage.CURATED)
    assert raw.state.stage is TTSTextStage.RAW
    assert curated.state.stage is TTSTextStage.CURATED


def test_text_lifecycle_rejects_skip_and_backward_transition() -> None:
    raw = make_sample()
    with pytest.raises(TTSTextLifecycleTransitionError):
        transition_text_stage(raw, TTSTextStage.CURATED)
    normalized = transition_text_stage(raw, TTSTextStage.NORMALIZED)
    with pytest.raises(TTSTextLifecycleTransitionError):
        transition_text_stage(normalized, TTSTextStage.RAW)


def test_same_text_stage_is_idempotent_copy() -> None:
    sample = make_sample()
    copy = transition_text_stage(sample, TTSTextStage.RAW)
    assert copy == sample
    assert copy is not sample
