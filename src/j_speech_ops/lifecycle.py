"""Small, explicit transition helper for source-data processing."""

from __future__ import annotations

from .schemas import ProcessingStage, SpeechSample

_ALLOWED_FORWARD_TRANSITIONS = {
    ProcessingStage.INGESTED: {ProcessingStage.AUDIO_PREPARED},
    ProcessingStage.AUDIO_PREPARED: {ProcessingStage.TRANSCRIBED},
    ProcessingStage.TRANSCRIBED: {
        ProcessingStage.TEXT_PREPARED,
        ProcessingStage.CURATED,
    },
    ProcessingStage.TEXT_PREPARED: {ProcessingStage.CURATED},
    ProcessingStage.CURATED: set(),
}


class LifecycleTransitionError(ValueError):
    """Raised when a processing-stage transition violates Stage 1 policy."""


def transition_stage(sample: SpeechSample, target_stage: ProcessingStage) -> SpeechSample:
    """Return a copy after an idempotent or explicitly allowed forward move."""
    target_stage = ProcessingStage(target_stage)
    current_stage = sample.state.stage
    if target_stage == current_stage:
        return sample.model_copy(deep=True)
    if target_stage not in _ALLOWED_FORWARD_TRANSITIONS[current_stage]:
        raise LifecycleTransitionError(
            f"invalid processing transition: {current_stage.value} -> {target_stage.value}; "
            "transition is not allowed by the lifecycle policy"
        )

    updated = sample.model_copy(deep=True)
    updated.state.stage = target_stage
    return updated
