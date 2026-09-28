# Stage 8 — Model-Neutral TTS Generation

## Scope

Stage 8 converts Stage 7 `CURATED + PASS` text into validated Japanese WAV artifacts. It implements generation and provenance only. It does not assess pronunciation, intelligibility, speaker similarity, style, pitch accent, MOS, or release fitness.

```text
synthesis_ready_text_samples.jsonl
        + Stage 6 preparation report
        + Stage 7 curation report
                    |
                    v
       synthesis text planning
                    |
                    v
        model-neutral batch job
                    |
          JSON subprocess boundary
                    |
                    v
      .venv-tts Qwen runtime worker
                    |
                    v
       zero-valued tail padding
                    |
                    v
      WAV validation + SHA-256
                    |
       +------------+-------------+
       |                          |
synthesis_runs.jsonl   generated_audio_manifest.jsonl
 all success/failure       validated success only
```

## One text to many runs

`TTSTextSample` remains unchanged at `CURATED + PASS`. Every synthesis attempt is a separate `SynthesisRun`, keyed by a stable `run_id` and linked through `sample_id`. Changing the backend, model, speaker, effective configuration, audio post-processing policy, seed, strategy, or actual synthesis text produces a distinct identity. Existing valid successful runs are skipped by default; `--force` allows regeneration.

This separation permits future backends and voices without adding synthesis fields or a `SYNTHESIZED` state to the text lifecycle.

## Synthesis text strategy

The strategy identifier is `normalized-with-confirmed-overrides-v1`:

1. Start with canonical `normalized_text`.
2. Join the matching Stage 6 preparation and Stage 7 curation reports.
3. Verify their applied-override provenance agrees.
4. Replace only each explicitly confirmed surface with its recorded Kana reading.
5. Require the observed surface count to equal the recorded occurrence count.
6. Retain `reading_kana` as expected-reading evidence.

The complete `reading_kana` sentence is never the default synthesis input. An occurrence mismatch creates a sample-level planning failure rather than a guessed rewrite.

## Runtime isolation

The normal command runs under Core `.venv`. `Qwen3TTSRuntimeAdapter` verifies and invokes project-local `.venv-tts\\Scripts\\python.exe`, passing a JSON batch job to `runtime/qwen3_tts_worker.py`. The worker depends only on its standard library plus PyTorch, qwen-tts, and SoundFile; Core never imports qwen-tts.

The worker loads the model once, validates CUDA/BF16/language/speaker, then processes all requests. Ordinary sample exceptions are returned as failed results and later samples continue. Initialization failures and CUDA/runtime corruption fail fast. CPU fallback is forbidden.

## Baseline configuration

- Model: `Qwen/Qwen3-TTS-12Hz-1.7B-CustomVoice`
- Language: `Japanese`
- Speaker: `Ono_Anna`
- Device: `cuda:0`
- dtype: `bfloat16`
- Attention: `sdpa`
- Instruction: none
- Mode: offline, non-streaming
- `do_sample=true`
- `top_k=50`, `top_p=1.0`, `temperature=0.9`
- `repetition_penalty=1.05`
- `subtalker_dosample=true`
- `subtalker_top_k=50`, `subtalker_top_p=1.0`, `subtalker_temperature=0.9`
- `max_new_tokens=8192`
- Post-roll: `tail_padding`, 300 ms by default, policy `tail-padding-v1`

The inference values were inspected from the installed qwen-tts merge logic and the local checkpoint `generation_config.json`, then pinned explicitly in `GenerationParameters`. Post-roll is deterministic audio post-processing after model inference and does not change these generation parameters.

## Seed and determinism

The default base seed is `20260921`. A stable SHA-256-derived component of `sample_id` is added modulo 2,147,483,647. Both the derived seed and policy identifier are stored on each run.

This is provenance determinism: the same input identity, configuration, strategy, and seed lead to the same run identity. It is not a promise of byte-identical GPU audio across different PyTorch, CUDA, qwen-tts, driver, or model versions.

## Outputs

- `synthesis_runs.jsonl`: successful and failed operational runs.
- `generated_audio_manifest.jsonl`: successful validated artifacts only.
- `tts_generation_summary.json`: counts plus batch model-load/GPU/peak-memory measurements.
- `audio/<run_id>.wav`: model-native sample-rate audio.

Validation requires a present, nonempty, decodable WAV with positive sample rate, channel count, and duration, finite samples, and non-silent content. Every success records SHA-256, generation time, audio duration, RTF, and post-processing provenance (`type`, `duration_ms`, policy version, and policy fingerprint).

## CLI

```powershell
.\.venv\Scripts\python.exe -m j_speech_ops.synthesize_tts `
    --manifest data/manifests/synthesis_ready_text_samples.jsonl `
    --prep-report data/reports/text_preparation_report.jsonl `
    --curation-report data/reports/tts_text_curation_report.jsonl `
    --model-path D:\AI-Models\Qwen3-TTS\Qwen3-TTS-12Hz-1.7B-CustomVoice `
    --output-dir outputs/stage8_tts_generation `
    --base-seed 20260921
```

Use `--force` to regenerate the same run identity. The model path is operational configuration and is not hardcoded in canonical source data.

`--post-roll-ms` accepts any non-negative integer. `0` disables added silence; `100`, `300`, and `500` select those durations without changing inference. See `stage8_post_roll_policy.md` for exact semantics.

## Performance limitation

The Stage 7.6 SDPA probe observed RTF around 5.8–6.6. The six-item Stage 8 integration smoke observed RTF 5.55–6.28, a 7.38-second model load, and approximately 4.0 GiB peak allocated VRAM on the RTX 5080 Laptop GPU. This baseline is suitable for production and batch validation, but it has not demonstrated real-time conversational latency. Stage 8 does not add FlashAttention, streaming, or performance optimization.

## Tests

Normal unit tests fake the runtime boundary and require no GPU, model, network, or `.venv-tts`. The `stage8_gpu_integration` test is skipped by default. Run it explicitly with:

```powershell
$env:J_SPEECH_OPS_RUN_STAGE8_GPU = "1"
$env:J_SPEECH_OPS_QWEN_MODEL_PATH = "D:\AI-Models\Qwen3-TTS\Qwen3-TTS-12Hz-1.7B-CustomVoice"
.\.venv\Scripts\python.exe -m pytest -q tests/test_tts_gpu_integration.py
```
