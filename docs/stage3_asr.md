# Stage 3 Japanese ASR Baseline

## Scope and architectural position

Stage 3 belongs only to the Speech Data Pipeline:

```text
Stage 2 audio_segments.jsonl -> Japanese ASR -> transcript-bearing SpeechSamples
```

It does not feed a TTS pipeline by architectural necessity. The independent future TTS Production Pipeline starts from text datasets, while the future Quality & Release Pipeline evaluates generated artifacts. Stage 3 includes no LLM, dialogue management, TTS, pronunciation feedback, CER/WER, or model comparison.

## Model-neutral core

`ASRAdapter` accepts a resolved audio path and returns J-SpeechOps `ASRResult`. The result contains aggregate transcript text, detected language metadata, duration, and J-SpeechOps `ASRSegment` records. Third-party faster-whisper segment objects do not leave the adapter.

`ASRBatchPipeline` depends on the protocol, not on `WhisperModel`. A future `SenseVoiceAdapter` can therefore reuse the same batch, manifests, reports, CLI-facing service boundary, and `SpeechSample` integration without changing the core contract.

## Baseline

```text
model:          Whisper large-v3
model source:   explicit local CTranslate2 directory
runtime:        faster-whisper / CTranslate2
device:         cuda:0
compute type:   float16
language:       ja
beam size:      5
internal VAD:   false
```

The adapter validates that CTranslate2 sees the requested CUDA device and supports float16 before loading the model. Missing CUDA, missing runtime libraries, model initialization failures, and CUDA inference failures raise `ASRInfrastructureError` and stop the batch. CPU fallback is not implemented.

## SpeechSample integration

On success, Stage 3 changes exactly two canonical values:

- `text.asr_text` receives the aggregate transcript.
- `state.stage` advances from `audio_prepared` to `transcribed`.

It does not modify `reference_text`, `normalized_text`, `reading_kana`, or `phonemes`. It does not compute an evaluation score or make a pass decision.

Missing, corrupt, or empty per-sample WAV data is recorded as a failed report, leaves the lifecycle at `audio_prepared`, applies dataset curation `drop`, and lets the batch continue. An empty ASR result follows the same policy with reason `asr_empty_transcript`. Unexpected runtime failures are not converted into hundreds of sample failures.

## Rerun policy

The output manifest is rewritten, never appended. Without `--force`, the pipeline skips input records already at `transcribed`. It also reuses a prior output transcript with the same `sample_id` and audio path, allowing the same original Stage 2 command input to be rerun without repeating inference or duplicating rows.

`--force` ignores prior output and runs inference again. A forced run on an already-transcribed sample uses an idempotent same-stage transition; it never moves lifecycle backward.

## Outputs

- `transcribed_segments.jsonl`: Stage 1-valid `SpeechSample` records, including failures and skipped records in input order.
- `asr_run_report.jsonl`: one current-run operational record per input, containing status, backend/version, model, CUDA policy, complete effective config, detected language metadata, audio duration, segment count, text length, and structured errors.

ASR segments remain operational result detail rather than being embedded in the canonical `SpeechSample`. The canonical sample stores only `text.asr_text`.

## CLI

```powershell
.\.venv\Scripts\python.exe -m j_speech_ops.asr_transcribe `
    --manifest data/manifests/audio_segments.jsonl `
    --dataset-root . `
    --output-dir data `
    --model-path <external-whisper-model-dir> `
    --local-files-only
```

`--model-path` is the preferred stable, explicit model location and is mutually exclusive with the legacy `--download-root` cache option. Use `--local-files-only` to prohibit network access and `--force` only when explicit re-transcription is intended. No reusable source code hardcodes the machine-specific path.

## GPU integration smoke test

The test is opt-in so ordinary unit tests do not download a multi-gigabyte model:

```powershell
$env:J_SPEECH_OPS_JA_SMOKE_WAV = "path/to/real-japanese-segment.wav"
$env:J_SPEECH_OPS_WHISPER_MODEL_PATH = "<external-whisper-model-dir>"
.\.venv\Scripts\python.exe -m pytest -q -m gpu_integration
```

The verified development smoke used the CC0 Wikimedia Commons file `Naruhodo-pcm.wav`, first passed through Stage 2. Test media remains under ignored `work/`; model weights live in an external model directory and neither is committed.

## Limitations

- large-v3 is the only implemented ASR backend.
- CUDA/float16 and Japanese are intentionally fixed for this baseline.
- There is no CPU execution mode or fallback.
- Model downloads require network access only when using the optional cache-based discovery path without an existing local snapshot; the verified explicit local path requires no network.
- Existing-output reuse keys on sample ID plus audio path; content hashes are not yet part of the canonical audio contract.
- Output replacement is deterministic but not transactional across manifest and report files.
- No ASR accuracy conclusions are drawn from the smoke sample.
