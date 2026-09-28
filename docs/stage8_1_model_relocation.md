# External model storage and offline discovery

## Model storage boundary

Whisper large-v3 and Qwen3-TTS checkpoints are stored in operator-selected directories outside the Git repository. Model weights are runtime resources, not project source, and must not be committed or copied under the project root.

Use these logical locations when following the examples:

```text
<external-whisper-model-dir>
<external-qwen-model-dir>
```

## Offline ASR selection

Production and smoke invocations select the Whisper directory explicitly and prohibit downloads:

```powershell
.\.venv\Scripts\python.exe -m j_speech_ops.asr_transcribe `
    --manifest data\manifests\audio_segments.jsonl `
    --dataset-root . `
    --output-dir data `
    --model-path <external-whisper-model-dir> `
    --local-files-only
```

The opt-in GPU integration test uses `J_SPEECH_OPS_WHISPER_MODEL_PATH` to supply the same explicit directory. Reusable ASR source has no machine-specific default. `--model-path` and the legacy cache-oriented `--download-root` option are mutually exclusive.

## TTS selection

Qwen3-TTS follows the same external-storage rule. Configure `<external-qwen-model-dir>` through the Stage 8 runtime interface. Moving the repository must not change, duplicate, or download the checkpoint.

## Execution boundary

An offline execution depends only on:

- the checked-out source tree;
- an environment rebuilt from project metadata;
- explicitly selected external model directories;
- the selected input data.

No runtime component may rely on a former checkout, cache location, or user-specific filesystem path.
