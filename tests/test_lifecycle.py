import pytest

from j_speech_ops.lifecycle import LifecycleTransitionError, transition_stage
from j_speech_ops.schemas import ProcessingStage, SpeechSample


def make_sample(stage: ProcessingStage) -> SpeechSample:
    return SpeechSample.model_validate(
        {
            "sample_id": "JP_0001",
            "dataset_id": "demo-ja",
            "audio": {"path": "audio/JP_0001.wav"},
            "source": {"source_name": "demo"},
            "state": {"stage": stage.value},
        }
    )


def test_transition_to_next_stage() -> None:
    original = make_sample(ProcessingStage.INGESTED)
    updated = transition_stage(original, ProcessingStage.AUDIO_PREPARED)
    assert original.state.stage is ProcessingStage.INGESTED
    assert updated.state.stage is ProcessingStage.AUDIO_PREPARED


def test_backward_transition_fails() -> None:
    sample = make_sample(ProcessingStage.TEXT_PREPARED)
    with pytest.raises(LifecycleTransitionError):
        transition_stage(sample, ProcessingStage.INGESTED)


def test_forward_skip_fails() -> None:
    sample = make_sample(ProcessingStage.INGESTED)
    with pytest.raises(LifecycleTransitionError):
        transition_stage(sample, ProcessingStage.TRANSCRIBED)


def test_same_stage_is_idempotent() -> None:
    sample = make_sample(ProcessingStage.INGESTED)
    updated = transition_stage(sample, ProcessingStage.INGESTED)
    assert updated == sample
    assert updated is not sample


def test_transcribed_can_advance_directly_to_curated() -> None:
    sample = make_sample(ProcessingStage.TRANSCRIBED)
    updated = transition_stage(sample, ProcessingStage.CURATED)
    assert updated.state.stage is ProcessingStage.CURATED
