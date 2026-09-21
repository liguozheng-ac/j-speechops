# J-SpeechOps

**Japanese Speech Data & TTS Operations Pipeline**

**Current Stage: Stage 9 — Generated Audio Content QA & Routing**

J-SpeechOps is a shared engineering core for Japanese speech-data production, future TTS production, and quality/release operations. These are related pipelines with separate responsibilities—not one mandatory Speech → ASR → TTS chain.

## System architecture

### 1. Speech Data Pipeline — MVP complete

```text
Raw Speech -> Audio Validation -> Resample / Mono -> VAD / Segmentation
-> Japanese ASR -> Transcript -> Speech Data Curation -> Usable Speech Dataset
```

Stages 1–4 are complete. Stage 4 consumes `transcribed_segments.jsonl`, records deterministic `pass`, `review`, or `drop` routing, and exports only PASS records to `usable_speech.jsonl`.

### 2. TTS Production Pipeline — Stage 8 complete

```text
Raw Text -> Text Data Contract ✓ -> Japanese Normalization ✓
-> Reading Preparation ✓ -> Text Curation ✓ -> Synthesis-ready Text
-> TTS Generation ✓ -> Generated Japanese Audio
```

Stage 5 established the independent `TTSTextSample` contract. Stage 6 fills `normalized_text` and model-independent `reading_kana` through separate layers. Stage 7 routes prepared text to PASS, REVIEW, or DROP and exports only PASS records as model-neutral synthesis-ready text. Stage 8 creates separate, one-to-many `SynthesisRun` records and validated audio artifacts without mutating that text contract. ASR remains optional provenance, not a mandatory TTS upstream step.

### 3. Quality & Release Pipeline

```text
Generated Audio
-> Artifact Integrity             ✓ Stage 9
-> ASR Round-trip Content QA      ✓ Stage 9
-> PASS / REVIEW / DROP           ✓ Stage 9
-> QA-pass Audio
-> Pronunciation / Regression
-> Human Review
-> Release Gate                   NEXT
```

Stage 9 evaluates artifact integrity and machine content-consistency evidence. Whisper disagreement routes to REVIEW rather than DROP. PASS is not a claim about naturalness, pronunciation quality, pitch accent, speaker similarity, human acceptance, or release fitness.

## Current implementation

- Stage 1: canonical Pydantic `SpeechSample` and dataset contract, lifecycle, JSONL IO, provenance, lineage, and JSON Schema.
- Stage 2: WAV validation, mono conversion, 16 kHz resampling, CPU Silero VAD, separate PCM_16 segments, operational reports, and CLI.
- Stage 3: model-neutral `ASRAdapter`, `ASRResult`, and `ASRSegment`; faster-whisper large-v3 baseline; fixed Japanese decoding; CUDA/float16 fail-fast initialization; batch reports and CLI.
- Stage 4: versioned curation policy; explicit issues and evidence; conservative DROP/REVIEW/PASS routing; exact-audio SHA-256 duplicate detection; canonical, usable, and operational report exports.
- Stage 5: immutable `TTSTextSample`; independent text lifecycle; text provenance, rights, locale, intended-use, and domain; stable-ID UTF-8 ingestion; text JSONL and JSON Schema.
- Stage 6/6.1: deterministic `jaconv` normalization with strict grouped-integer separator removal; model-neutral `ReadingProvider`; OpenJTalk Katakana baseline; traceable pronunciation overrides; layered manifests, operational reports, CLI, and regression fixture.
- Stage 7: deterministic TTS-text risk inspection; rights, prepared-data, Unicode reading-script, emoji, control-character, and pronunciation-watchlist routing; PASS-only synthesis-ready export; operational provenance and CLI.
- Stage 8: model-neutral synthesis contracts; normalized-text plus confirmed-override planning; automatic Core-to-TTS subprocess isolation; Qwen3-TTS 1.7B/Ono_Anna Japanese SDPA baseline; stable seed/run identity; idempotent generation; validated WAV hashes; success-only audio manifest and CLI.
- Stage 9: independent generated-audio QA contracts; objective integrity routing; one-load Whisper round trip; Stage 6 normalization/OpenJTalk reuse; conservative reading canonicalization; exact-match PASS; mismatch REVIEW; PASS-only and review manifests; idempotent CLI.

The ASR core depends only on J-SpeechOps contracts. Future adapters such as SenseVoice can implement the same boundary without changing the batch pipeline or `SpeechSample`.

## Installation

The verified environment is Windows with Python 3.14 and an NVIDIA GPU:

```powershell
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install -e ".[dev]"
```

Windows CUDA 12 cuBLAS/cuDNN wheels are installed inside `.venv`; J-SpeechOps adds their DLL directories to the current Python process only. It does not modify the system PATH. GPU initialization and inference errors fail the batch—there is no silent CPU fallback.

## Stage 2 CLI

```powershell
.\.venv\Scripts\python.exe -m j_speech_ops.audio_prepare `
    --manifest source_samples.jsonl `
    --dataset-root . `
    --output-dir data
```

This produces `data/manifests/audio_segments.jsonl` for Stage 3.

## Stage 3 CLI

```powershell
.\.venv\Scripts\python.exe -m j_speech_ops.asr_transcribe `
    --manifest data/manifests/audio_segments.jsonl `
    --dataset-root . `
    --output-dir data `
    --model large-v3 `
    --model-path D:\AI-Models\Whisper\faster-whisper-large-v3 `
    --device cuda `
    --compute-type float16 `
    --language ja `
    --local-files-only
```

Outputs are rewritten on each run:

```text
data/manifests/transcribed_segments.jsonl
data/reports/asr_run_report.jsonl
```

`--model-path` selects the stable external CTranslate2 model directory explicitly; it is mutually exclusive with the legacy Hugging Face `--download-root` cache option. `--local-files-only` prevents downloads. An existing `transcribed` record is skipped by default. `--force` explicitly reruns ASR without moving lifecycle state backward. Faster-whisper internal VAD is always disabled because Stage 2 already owns segmentation.

## Stage 4 CLI

```powershell
.\.venv\Scripts\python.exe -m j_speech_ops.curate_speech `
    --manifest data/manifests/transcribed_segments.jsonl `
    --dataset-root . `
    --output-dir data
```

Outputs are deterministically rewritten on each run:

```text
data/manifests/curated_segments.jsonl
data/manifests/usable_speech.jsonl
data/reports/speech_curation_report.jsonl
```

`usable_speech.jsonl` contains PASS samples only. Existing `curated` samples are skipped unless `--force` is supplied. Curation never changes transcript or other canonical content fields; it updates only processing stage and curation state.

## Stage 5 CLI

```powershell
.\.venv\Scripts\python.exe -m j_speech_ops.ingest_text `
    --input examples/hospitality_text.txt `
    --dataset-id hospitality-demo `
    --language ja `
    --locale ja-JP `
    --source-name japanese-work-rehearsal `
    --rights-status cleared `
    --intended-use tts_production `
    --domain hospitality `
    --id-prefix HOSPITALITY `
    --output data/manifests/tts_text_samples.jsonl
```

The input contract is UTF-8 plain text with one line per sample. Stage 5 removes only line endings; it preserves all other text exactly and rejects empty or whitespace-only lines. Repeating the same command against identical input produces the same IDs, ordering, and canonical manifest bytes.

## Stage 6 CLI

```powershell
.\.venv\Scripts\python.exe -m j_speech_ops.prepare_japanese_text `
    --manifest data/manifests/tts_text_samples.jsonl `
    --output-dir data `
    --overrides config/japanese_reading_overrides.json
```

Outputs:

```text
data/manifests/normalized_text_samples.jsonl
data/manifests/reading_prepared_text_samples.jsonl
data/reports/text_preparation_report.jsonl
```

`raw_text` remains unchanged. `normalized_text` remains written text rather than Kana; valid thousands-grouped integers such as `1,234,567` normalize to `1234567`, while malformed grouping, decimals, and ordinary commas remain untouched. `reading_kana` is an OpenJTalk-generated baseline, not human-verified ground truth. `phonemes` remains null. Existing reading-prepared records are skipped unless `--force` is supplied.

## Tests

```powershell
.\.venv\Scripts\python.exe -m pytest -q
```

The Stage 3, Stage 8, and Stage 9 GPU integration tests are opt-in and default-skipped. See their stage documentation for environment variables and verified smoke procedures.

## Stage 7 CLI

```powershell
.\.venv\Scripts\python.exe -m j_speech_ops.curate_tts_text `
    --manifest data/manifests/reading_prepared_text_samples.jsonl `
    --prep-report data/reports/text_preparation_report.jsonl `
    --risk-watchlist config/japanese_pronunciation_risks.json `
    --output-dir data
```

Outputs are `curated_tts_text_samples.jsonl`, PASS-only `synthesis_ready_text_samples.jsonl`, and `tts_text_curation_report.jsonl`. PASS means no current policy rule fired; it is not human-verified pronunciation. REVIEW is an expected routing result, not a pipeline failure. Use `--force` to recurate existing CURATED records without changing prepared text fields.

## Stage 8 CLI

Run this command from the normal Core environment. It automatically invokes the project-local `.venv-tts` worker; manual environment switching is not required.

```powershell
.\.venv\Scripts\python.exe -m j_speech_ops.synthesize_tts `
    --manifest data/manifests/synthesis_ready_text_samples.jsonl `
    --prep-report data/reports/text_preparation_report.jsonl `
    --curation-report data/reports/tts_text_curation_report.jsonl `
    --model-path D:\AI-Models\Qwen3-TTS\Qwen3-TTS-12Hz-1.7B-CustomVoice `
    --output-dir outputs/stage8_tts_generation
```

Only `CURATED + PASS` samples are accepted. The default text strategy sends `normalized_text` with only confirmed Stage 6 pronunciation overrides applied locally; full `reading_kana` is retained as expected-reading evidence. Existing successful runs with a matching WAV hash are skipped unless `--force` is supplied.

Outputs are `synthesis_runs.jsonl`, success-only `generated_audio_manifest.jsonl`, `tts_generation_summary.json`, and native-rate WAV files under `audio/`. The current BF16/SDPA baseline is intended for production and batch validation but has not demonstrated real-time conversational latency.

## Stage 9 CLI

Run Stage 9 from the Core environment. It reuses existing Stage 8 audio and does not invoke Qwen or `.venv-tts`.

```powershell
.\.venv\Scripts\python.exe -m j_speech_ops.qa_generated_audio `
    --manifest outputs\stage8_tts_generation\integration_smoke\generated_audio_manifest.jsonl `
    --model-path D:\AI-Models\Whisper\faster-whisper-large-v3 `
    --output-dir outputs\stage9_generated_audio_qa\integration_smoke
```

Outputs are the complete `generated_audio_qa_results.jsonl`, PASS-only `qa_pass_audio_manifest.jsonl`, REVIEW-only `qa_review_queue.jsonl`, and `qa_summary.json`. A Stage 9 PASS means only that current integrity checks succeeded and canonical expected/observed round-trip readings matched exactly. It is not released audio.

## Documentation

- `docs/stage3_asr.md` — Stage 3 scope, batch behavior, CLI, and verification
- `docs/asr_policy.md` — fixed Japanese ASR and rerun policy
- `docs/stage4_speech_curation.md` — Stage 4 architecture, outputs, and operation
- `docs/data_curation_policy.md` — versioned signals, thresholds, and limitations
- `docs/stage5_tts_foundation.md` — Stage 5 architecture, ingestion, and verification
- `docs/tts_text_contract.md` — canonical TTS text fields and lifecycle
- `docs/stage6_japanese_text_preparation.md` — Stage 6 pipeline, outputs, and verification
- `docs/japanese_normalization_policy.md` — deterministic character-form policy
- `docs/japanese_reading_policy.md` — OpenJTalk baseline and override policy
- `docs/stage7_tts_text_curation.md` — Stage 7 architecture, outputs, CLI, and operations
- `docs/tts_text_curation_policy.md` — deterministic PASS/REVIEW/DROP policy
- `docs/pronunciation_risk_watchlist.md` — watchlist/override separation and feedback loop
- `docs/stage8_tts_generation.md` — Stage 8 architecture, runtime isolation, contracts, CLI, and limitations
- `docs/stage9_generated_audio_content_qa.md` — Stage 9 architecture, contracts, CLI, outputs, and limitations
- `docs/generated_audio_qa_policy.md` — integrity, reading comparison, and PASS/REVIEW/DROP policy
- `docs/stage2_audio_preparation.md` — Stage 2 audio preparation
- `docs/audio_preparation_policy.md` — audio and VAD policy
- `docs/data_schema.md` — canonical `SpeechSample` contract
- `docs/architecture_decisions.md` — architecture decisions
- `docs/third_party.md` — dependencies, versions, licenses, and attribution
