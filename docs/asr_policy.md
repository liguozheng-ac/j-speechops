# Japanese ASR Policy

## Fixed baseline

| Setting | Value |
|---|---|
| Model | Whisper `large-v3` |
| Model source | Explicit local CTranslate2 directory; configurable with `--model-path` |
| Backend | faster-whisper / CTranslate2 |
| Device | `cuda:0` |
| Compute type | `float16` |
| Language | `ja` |
| Beam size | `5` |
| Temperature | `0.0` |
| Condition on previous text | `false` |
| Word timestamps | `false` |
| faster-whisper VAD | `false` |

The verified stable local model root is `D:\AI-Models\Whisper\faster-whisper-large-v3`. This location is documentation and deployment configuration, not a hardcoded source default. `--local-files-only` is required for offline validation.

Language detection metadata may be recorded by the backend, but decoding is explicitly constrained to Japanese. Stage 2 already supplies speech-only segments, so enabling faster-whisper VAD would duplicate responsibility and could change established lineage boundaries.

## Text mutation boundary

Successful ASR writes only `SpeechSample.text.asr_text`. Reference, normalized, reading, and phoneme fields are owned by other data sources or future stages and remain untouched.

## Lifecycle

```text
audio_prepared -> transcribed
```

Already-transcribed records are skipped unless `--force` is present. Forced processing uses the existing idempotent transition and never moves backward.

## Failure classification

Infrastructure failures stop the batch:

- no requested CUDA device
- float16 unsupported
- CUDA runtime library missing
- model download or initialization failure
- CUDA inference failure

Expected data failures affect only one record:

- source audio missing, corrupt, empty, or non-finite
- source record is not at an ASR-compatible lifecycle stage
- ASR returns no transcript text

Data failures remain in the output manifest for auditability and are marked `drop` with a stable reason. No broad exception handler suppresses implementation defects.

## Evaluation boundary

Stage 3 records transcripts and operational evidence only. It does not compute CER, WER, Kana-CER, normalized scores, rankings, or acceptance thresholds—even when reference text exists.
