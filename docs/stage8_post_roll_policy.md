# Stage 8 Post-roll Policy

## Background

Human listening found that the six Stage 8 Japanese Qwen3-TTS samples ended with too little acoustic space. Diagnostic reproduction showed that the in-memory Qwen waveform and exported PCM16 WAV had identical frame counts, ruling out export trimming. Adding 300 ms of silence made the ending sound natural; 500 ms offered no useful improvement for the added latency.

## Processing position

```text
Qwen3-TTS inference waveform
        |
        v
tail-padding post-processing
        |
        v
PCM16 WAV export
        |
        v
GeneratedAudioArtifact
```

The processor concatenates the original waveform samples with exact zero-valued samples. It never trims, resamples, normalizes, fades, or modifies the original sample prefix.

## Configuration

`AudioPostProcessingPolicy` centrally defines the operation:

- `type`: `tail_padding`
- `duration_ms`: `300` by default
- `policy_version`: `tail-padding-v1`

The Stage 8 CLI exposes `--post-roll-ms`. Non-negative values such as `0`, `100`, `300`, and `500` are supported. A value of `0` preserves the model waveform length while retaining explicit policy provenance.

At sample rate `R`, the processor appends `round(R * duration_ms / 1000)` frames for every channel. The Qwen baseline uses 24 kHz, so the default appends 7,200 frames.

## Provenance and identity

Every newly produced `SynthesisRun` and `GeneratedAudioArtifact` records:

```json
{
  "audio_post_processing": {
    "type": "tail_padding",
    "duration_ms": 300,
    "policy_version": "tail-padding-v1",
    "policy_fingerprint_sha256": "..."
  }
}
```

The policy is included in the synthesis configuration fingerprint. Changing its duration therefore changes `run_id`, output path, WAV SHA-256, downstream evidence identity, and human-review queue identity. Existing artifacts are not overwritten, and human decisions tied to an older artifact cannot authorize the new one.

Legacy manifests without `audio_post_processing` remain readable and accurately indicate that no structured post-processing provenance was recorded.

## Limitations

Post-roll provides playback space after the generated utterance. It cannot restore a phoneme missing from the model waveform and is not a pronunciation, prosody, speaker, or style correction. It does not alter Stage 9 QA thresholds, Stage 10A evidence semantics, or Stage 10B human-review and release policy.
