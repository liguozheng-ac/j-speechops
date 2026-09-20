# Stage 8.1 Whisper model relocation

## Stable model storage

The verified Whisper large-v3 CTranslate2 model is stored outside the Git repository:

```text
D:\AI-Models\Whisper\faster-whisper-large-v3
```

The directory contains the copied model snapshot files directly. Model weights are not project source and must not be placed under `D:\AI-Projects\J-SpeechOps` or committed to Git.

The Qwen3-TTS model remains unchanged at:

```text
D:\AI-Models\Qwen3-TTS\Qwen3-TTS-12Hz-1.7B-CustomVoice
```

## Offline ASR selection

Production and smoke invocations select the Whisper directory explicitly and prohibit downloads:

```powershell
.\.venv\Scripts\python.exe -m j_speech_ops.asr_transcribe `
    --manifest data\manifests\audio_segments.jsonl `
    --dataset-root . `
    --output-dir data `
    --model-path D:\AI-Models\Whisper\faster-whisper-large-v3 `
    --local-files-only
```

The opt-in GPU integration test uses `J_SPEECH_OPS_WHISPER_MODEL_PATH` to supply the same explicit directory. The reusable ASR source has no machine-specific default. `--model-path` and the legacy cache-oriented `--download-root` option are mutually exclusive.

## Execution boundary

The canonical project root is `D:\AI-Projects\J-SpeechOps`. Its Stage 3 runtime now depends only on the canonical source tree, its rebuilt environment, the stable external model directory, and the selected input data. The retained former workspace is a backup only and is not required for J-SpeechOps execution.

The relocation was copy-first. The source cache and former workspace were not deleted.
