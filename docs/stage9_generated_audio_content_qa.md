# Stage 9 — Generated Audio Content QA & Routing

## Scope

Stage 9 validates successful Stage 8 generated-audio artifacts and collects Whisper round-trip evidence about linguistic-content consistency:

```text
GeneratedAudioArtifact
  -> WAV integrity and provenance
  -> Whisper large-v3 round trip
  -> JapaneseNormalizer
  -> OpenJTalk reading
  -> canonical reading comparison
  -> PASS / REVIEW / DROP
```

Generated TTS audio has an independent QA lifecycle. It is not converted into `SpeechSample`, and Stage 8 manifests are never mutated.

Round-trip ASR is evidence rather than ground truth. Whisper can be wrong, so a mismatch or empty transcript routes to REVIEW rather than DROP. Reading-level comparison is required because comparing Whisper Kanji directly with expected Kana would be meaningless.

## Contracts

Stage 9 defines frozen operational contracts for:

- `GeneratedAudioQAResult`;
- `AudioIntegrityEvidence`;
- `ASRRoundTripEvidence`;
- `ContentComparisonEvidence`;
- `QAArtifactIssue`;
- `QABatchSummary`;
- PASS-manifest and review-queue records.

No `SpeechSample`, `TTSTextSample`, or `SynthesisRun` schema is changed.

## Runtime

Stage 9 runs entirely in the Core `.venv`. It reuses one `FasterWhisperAdapter` instance for the whole batch with Japanese, CUDA FP16, beam size 5, temperature 0, no word timestamps, no previous-text conditioning, no VAD, and `local_files_only=true`.

The default stable model directory is supplied explicitly at invocation time; reusable source code does not hardcode it. Model/CUDA initialization failures fail the batch. A recoverable per-file exception creates a FAILED/UNDECIDED result and later samples continue.

Whisper initialization is lazy: a batch with work loads the model once, while a fully idempotent rerun can validate hashes and skip every existing `qa_id` without loading the 2.879 GiB model.

## CLI

```powershell
.\.venv\Scripts\python.exe -m j_speech_ops.qa_generated_audio `
    --manifest outputs\stage8_tts_generation\integration_smoke\generated_audio_manifest.jsonl `
    --model-path D:\AI-Models\Whisper\faster-whisper-large-v3 `
    --output-dir outputs\stage9_generated_audio_qa\integration_smoke
```

Use `--force` only when the same QA identities should be retranscribed. The CLI never invokes Qwen or `.venv-tts`.

## Tests

Normal tests are CPU/offline/model-free and use fake ASR evidence. The real GPU integration is opt-in:

```powershell
$env:J_SPEECH_OPS_RUN_STAGE9_GPU = "1"
$env:J_SPEECH_OPS_WHISPER_MODEL_PATH = "D:\AI-Models\Whisper\faster-whisper-large-v3"
$env:J_SPEECH_OPS_STAGE8_AUDIO_MANIFEST = "path\to\generated_audio_manifest.jsonl"
.\.venv\Scripts\python.exe -m pytest -q tests\test_generated_audio_qa_gpu_integration.py
```

The test logic contains no machine-specific model or artifact path.

## Outputs

- `generated_audio_qa_results.jsonl`: all PASS, REVIEW, DROP, and FAILED/UNDECIDED results.
- `qa_pass_audio_manifest.jsonl`: PASS records only; the Stage 10 input boundary, not released audio.
- `qa_review_queue.jsonl`: REVIEW records with text, readings, edit evidence, and issues.
- `qa_summary.json`: routing counts, policy/ASR identities, model-load time, batch time, and GPU identity.

DROP records remain only in the complete results manifest. FAILED/UNDECIDED records enter neither downstream manifest.

## Interpretation and limitations

The exact-match baseline is intentionally conservative and explainable. CER is retained for diagnosis, not thresholding or scoring. Stage 9 does not evaluate naturalness, pronunciation correctness, pitch accent, prosody, emotion, speaker similarity, MOS, or human preference. PASS is not final release; pronunciation/regression checks, human review, and the release gate remain downstream.
