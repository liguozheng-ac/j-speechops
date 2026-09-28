# Diagnostic Utilities

This directory contains isolated engineering tools used to investigate TTS output. They are not production pipeline components, are not imported by `j_speech_ops`, and are not included in package discovery under `src/`.

## Tools

### `analyze_audio_tail.py`

Inspects WAV metadata, amplitude statistics, the last sample above a configurable silence threshold, and trailing-silence duration. It can also create diagnostic copies with exact zero-valued tail padding while verifying that the original PCM prefix remains unchanged.

Typical uses:

- diagnose abrupt end-of-utterance behavior;
- compare existing artifact tails;
- create temporary 300 ms or 500 ms listening variants.

### `qwen3_tts_jr_experiment.py`

Runs an isolated Qwen3-TTS A/B experiment for the JR service sentence. It preserves the baseline speaker, language, seed, and generation parameters while comparing the current no-instruction behavior with one explicit service-style instruction.

This script is for controlled style investigation only. It does not change Stage 8 defaults or write canonical manifests.

## Operational boundary

- Run tools explicitly from the repository root.
- Store generated files under ignored `outputs/` or another disposable location.
- Do not treat diagnostic output as a canonical artifact or release decision.
- Do not import these scripts into Stage 8, Stage 9, Stage 10A, or Stage 10B runtime code.
- GPU/model requirements apply only to the Qwen experiment; tail analysis runs in the Core environment.
