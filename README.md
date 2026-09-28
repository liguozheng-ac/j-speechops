# J-SpeechOps

[![Core tests](https://github.com/liguozheng-ac/j-speechops/actions/workflows/test.yml/badge.svg)](https://github.com/liguozheng-ac/j-speechops/actions/workflows/test.yml)

Japanese Speech Data Quality Control and Release Pipeline

## Overview

J-SpeechOps builds a complete, auditable workflow from Japanese text input through speech generation, automated quality checks, pronunciation evidence, human review, and release management.

The project separates generation, machine evidence, human judgment, and release authority. A generated WAV is not automatically considered good, a machine PASS is not human approval, and human approval is valid only for the exact artifact and evidence context that was reviewed.

The repository also supports an independent speech-data workflow for validating recordings, segmenting speech, transcribing Japanese audio, and curating usable datasets.

This is a production-oriented prototype for controlled batch workflows, not a deployed service or model-training platform.

## Quick Start

On Windows with Python 3.14, clone the repository and verify the Core package without a GPU or model weights:

```powershell
git clone https://github.com/liguozheng-ac/j-speechops.git J-SpeechOps
cd J-SpeechOps
py -3.14 -m venv .venv
.\.venv\Scripts\python.exe -m pip install --upgrade pip
.\.venv\Scripts\python.exe -m pip install -e ".[dev]"
.\.venv\Scripts\python.exe -m pip check
.\.venv\Scripts\python.exe -m pytest -q
```

The three GPU integration tests are opt-in and skip by default. Whisper and Qwen3-TTS model downloads are not part of this Core validation. See the stage documents for optional model-backed runs.

## Architecture

```text
Japanese Text
    |
    v
Text Preparation and Curation
    |
    v
TTS Generation
    |
    v
Audio Post-processing
    |
    v
GeneratedAudioArtifact
    |
    v
Audio QA
    |
    v
Pronunciation Evidence and Regression
    |
    v
Human Review
    |
    v
ReleasedAudioArtifact
```

See [Architecture Overview](docs/architecture_overview.md) for the system boundaries and design principles.

## Features

- Immutable Pydantic contracts and deterministic JSONL manifests.
- Japanese normalization and model-independent reading preparation.
- Explicit PASS, REVIEW, and DROP routing for text and generated audio.
- Model-neutral TTS planning with isolated GPU runtime execution.
- Configurable zero-valued audio post-roll with versioned provenance.
- WAV integrity checks and SHA-256 artifact identity.
- Local-only Japanese Whisper round-trip evidence.
- Pronunciation-risk evidence and logical-case regression comparison.
- Mandatory human APPROVE, REWORK, or REJECT records.
- Freshness-aware release gate with non-overridable artifact blockers.
- Released-audio manifests linking generation, QA, evidence, review, and policy.

## Workflow

The TTS quality and release path is organized into explicit stages:

1. Define and ingest canonical Japanese text records.
2. Normalize text and prepare expected readings.
3. Curate rights, script, and pronunciation risks.
4. Generate audio and apply the configured post-processing policy.
5. Validate the artifact and collect content-consistency evidence.
6. Build pronunciation and regression evidence.
7. Present the complete evidence context to a human reviewer.
8. Release only artifacts with a fresh, policy-valid approval.

Generated models, caches, virtual environments, WAV outputs, and operational manifests remain outside Git through the repository ignore policy.

### Example Flow

The [synthetic release-flow records](examples/release_flow/README.md) show how one `TTSTextSample` links to a `GeneratedAudioArtifact`, `GeneratedAudioQAResult`, pronunciation evidence, a `HumanReviewRecord`, and a `ReleasedAudioArtifact`. They contain no WAV or real review. Each decision is separate: generation success does not imply QA PASS; QA PASS does not imply human APPROVE; APPROVE still requires a current release-gate decision before RELEASED.

### Optional Model-backed Runs

The verified local GPU setup uses an NVIDIA CUDA GPU. Qwen uses a separate project-local `.venv-tts` environment; model weights stay in an operator-selected directory outside the repository. Stage-specific documents describe the environment variables, model paths, and expected outputs for opt-in integration tests.

## Technical Decisions

- Text data, synthesis runs, generated artifacts, QA evidence, human decisions, and release records have separate lifecycles.
- Effective generation and post-processing configuration contributes to stable run identity.
- A 300 ms exact-zero post-roll is the default artifact policy; it does not alter the model waveform prefix.
- Model runtimes are isolated from the Core environment through explicit subprocess boundaries.
- Machine REVIEW routes uncertainty to a human; it is not a permanent release denial.
- Missing, corrupt, silent, hash-invalid, failed, or provenance-inconsistent artifacts remain hard blockers.
- There is no force-release path and no fabricated MOS, naturalness, or pronunciation score.

Detailed decisions are recorded in [Architecture Decisions](docs/architecture_decisions.md).

## Limitations

J-SpeechOps currently has no pitch-accent validator, MOS evaluator, automatic naturalness score, reviewer authentication, graphical review UI, or production deployment layer. ASR and reading providers are evidence sources rather than phonetic ground truth.

The current GPU baselines are suitable for controlled batch workflows. They do not establish real-time conversational latency or cross-machine byte-identical synthesis.

## Future Work

Future work may add stronger phonetic evidence, reviewer identity integration, release packaging, additional model adapters, and deployment tooling. Each addition should preserve the current evidence, provenance, and human-approval boundaries.

## Documentation

- [Architecture overview](docs/architecture_overview.md)
- [Architecture decisions](docs/architecture_decisions.md)
- [Data contracts](docs/data_schema.md)
- [Japanese normalization policy](docs/japanese_normalization_policy.md)
- [Japanese reading policy](docs/japanese_reading_policy.md)
- [TTS text curation](docs/stage7_tts_text_curation.md)
- [Stage 8 TTS generation](docs/stage8_tts_generation.md)
- [Stage 8 post-roll policy](docs/stage8_post_roll_policy.md)
- [Stage 9 generated-audio QA](docs/stage9_generated_audio_content_qa.md)
- [Stage 10A pronunciation regression](docs/stage10a_pronunciation_regression.md)
- [Stage 10B human review and release](docs/stage10b_human_review_release.md)
- [Diagnostic utilities](tools/diagnostics/README.md)

## License

Copyright 2026 Guozheng Li.

Licensed under the Apache License, Version 2.0. See [LICENSE](LICENSE) for details.
