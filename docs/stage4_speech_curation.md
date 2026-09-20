# Stage 4 — Speech Data Curation

## Scope

Stage 4 is the final implemented step of the Speech Data Pipeline:

```text
transcribed_segments.jsonl
  -> deterministic speech-data curation
  -> curated_segments.jsonl
  -> usable_speech.jsonl
```

It decides how a speech-text pair should be routed under a named operational policy. It is not ASR benchmarking, Japanese grammar evaluation, pronunciation evaluation, normalization, G2P, TTS, or a human-review platform.

The three project pipelines remain separate:

```text
Speech Data:       Speech -> ASR -> Curation
TTS Production:    Text -> Japanese Normalization -> Reading/G2P -> TTS
Quality & Release: Generated Audio -> QA -> Review -> Release
```

## Architecture

`SpeechCurationPipeline` reads canonical `SpeechSample` records in `transcribed` state. For each non-skipped record it calculates lightweight signals, produces a `SpeechCurationReport`, and advances the canonical copy directly to `curated`. Processing stage and curation decision remain independent, so `curated + review` is valid.

The current Stage 3 report does not expose model-confidence fields needed by a curation rule, so Stage 4 neither consumes those fields nor reruns Whisper. If such operational evidence is added later, it may contribute only a REVIEW signal under a newly versioned policy.

The engine updates only:

- `state.stage`
- `state.curation_decision`
- `state.decision_reasons`

Audio metadata, all text fields, source/provenance, lineage, speaker, split, and metadata are preserved. Internal analysis views are never written back to the sample.

## Outputs

- `manifests/curated_segments.jsonl`: all input records in input order, with final routing state.
- `manifests/usable_speech.jsonl`: PASS records only; this is not called a training manifest.
- `reports/speech_curation_report.jsonl`: policy version, computed signals, content hash, detailed issues, evidence, and decision.

The batch result also reports total record counts, PASS/REVIEW/DROP counts, corresponding declared audio durations, and issue-code frequencies. It does not report ASR accuracy.

## Decision behavior

Issues have curation-local severities: `critical`, `warning`, and `info`. Version 1 uses critical issues for hard-drop conditions and warnings for ambiguous signals:

```text
any critical issue -> DROP
otherwise any warning -> REVIEW
otherwise -> PASS
```

This priority is deterministic. There is no aggregate quality score.

Existing `curated` records are skipped by default. Their decision must already be PASS, REVIEW, or DROP. A matching prior report is preserved with its original policy version; when no prior evidence exists, the generated skip record explicitly uses `existing-curation-policy-unknown` rather than claiming that the current policy ran. `--force` recalculates the signals and decision while keeping lifecycle state at `curated`. Duplicate sample IDs, invalid states, and schema-invalid JSONL are considered fundamentally broken manifests and stop the run; per-record audio and content problems become issues instead.

## CLI

```powershell
python -m j_speech_ops.curate_speech `
  --manifest data/manifests/transcribed_segments.jsonl `
  --dataset-root . `
  --output-dir data
```

Thresholds can be overridden with the corresponding duration, density, script-ratio, and repetition arguments. `--allow-unknown-rights` disables only the UNKNOWN-rights warning; RESTRICTED always drops. Use `--force` to recompute existing curated records.

## Determinism and limitations

Given the same ordered manifest, policy, and audio bytes, decisions, issue order, output order, and JSONL serialization are stable. SHA-256 detects exact file-byte duplicates only, not perceptual or decoded-audio equivalence. Declared durations are trusted rather than re-running Stage 2 validation. The Unicode-range signal recognizes Hiragana, Katakana, and common/CJK compatibility ideographs but is not complete language identification.

**Curation heuristics are data-routing signals, not ground-truth ASR accuracy measurements.**

**PASS means no current curation rule was triggered. It does not guarantee transcript correctness.**
