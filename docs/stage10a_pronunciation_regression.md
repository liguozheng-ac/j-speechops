# Stage 10A — Japanese Pronunciation Evidence & Regression

## Purpose

Stage 10A turns existing Stage 7–9 provenance into traceable pronunciation-risk and regression evidence. It does not automatically judge whether Japanese pronunciation is correct, native-like, natural, or release-ready.

```text
Stage 7 curation/risk/override provenance
  + Stage 8 synthesis run and generated artifact
  + Stage 9 QA-pass result and observed round-trip reading
  -> pronunciation evidence
  -> persistent regression baseline
  -> current-versus-baseline comparison
  -> no-regression-detected or REVIEW
```

Generated audio remains outside the `SpeechSample` lifecycle. Stage 10A does not mutate Stage 7, Stage 8, or Stage 9 artifacts.

## Contracts

- `ExpectedPronunciationReference`: expected sequence, scope, source, confirmation status, and provenance reference.
- `PronunciationRiskEvidence`: source/normalized/synthesis spans and conservative round-trip support status.
- `ObservedPronunciationEvidence`: Stage 9 QA identity, transcript, reading, canonical reading, edit evidence, and issues.
- `PronunciationEvidenceResult`: joined Stage 7–9 provenance plus Stage 10A routing.
- `PronunciationRegressionBaseline`: persistent logical-case snapshot; WAV bytes are provenance, not the baseline definition.
- `PronunciationRegressionResult`: material pronunciation changes and non-material provenance changes.
- `PronunciationRegressionSummary`: deterministic batch counts.

## Evidence semantics

Expectation sources remain distinct:

- `NORMALIZATION_RULE`: an explicit deterministic rule-derived expectation, when a future producer can provide a reliable span mapping;
- `READING_PROVIDER`: OpenJTalk-derived evidence, never human truth;
- `CONFIRMED_OVERRIDE`: a configured Stage 6 pronunciation override;
- `HUMAN_CONFIRMED`: reserved for explicit human-confirmation provenance.

The current production builder creates full-text `READING_PROVIDER` references and span-level `CONFIRMED_OVERRIDE` references. It records normalization transformations but does not manufacture a local pronunciation expectation from a transformation alone. No current fixture is labeled human-confirmed.

For a confirmed span, canonical substring matching may report:

- `EXPECTED_SEQUENCE_OBSERVED`;
- `EXPECTED_SEQUENCE_NOT_OBSERVED`.

For an unconfirmed Stage 7 watchlist risk without a trustworthy local expected sequence, it reports `ALIGNMENT_UNRESOLVED`. Observed means only that the expected sequence appears somewhere in the complete Stage 9 canonical reading; it is supporting round-trip evidence, not phonetic alignment or proof of correctness. Missing or unresolved sequences route to REVIEW, never DROP.

No detected risk does not imply perfect pronunciation.

## Regression semantics

The baseline captures logical content and provenance:

- source text and expected references;
- risk spans/status;
- observed canonical reading and Stage 9 comparison evidence;
- TTS backend/model/speaker/strategy/config;
- Stage 7, watchlist, override, Stage 9 ASR/reading, and Stage 10A policy identities.

Changed expected readings, observed readings, risk status, source text, or missing cases are material evidence and route to REVIEW. Backend/model/speaker/policy/config changes are recorded but do not independently establish a pronunciation regression. Audio SHA-256 changes are recorded only as artifact-identity changes; different WAV bytes alone are not a pronunciation regression.

There is no pronunciation score, pitch-accent score, MOS, or arbitrary regression threshold.

## CLI

Build evidence from the Stage 9 PASS boundary:

```powershell
.\.venv\Scripts\python.exe -m j_speech_ops.pronunciation_regression_cli build `
    --artifact-manifest outputs\stage8_tts_generation\integration_smoke\generated_audio_manifest.jsonl `
    --synthesis-runs outputs\stage8_tts_generation\integration_smoke\synthesis_runs.jsonl `
    --qa-results outputs\stage9_generated_audio_qa\integration_smoke\generated_audio_qa_results.jsonl `
    --qa-pass-manifest outputs\stage9_generated_audio_qa\integration_smoke\qa_pass_audio_manifest.jsonl `
    --curation-report outputs\stage8_integration_smoke_fixture\stage7_replay\reports\tts_text_curation_report.jsonl `
    --output-dir outputs\stage10a_pronunciation_regression\integration_smoke
```

Capture a baseline:

```powershell
.\.venv\Scripts\python.exe -m j_speech_ops.pronunciation_regression_cli capture-baseline `
    --evidence outputs\stage10a_pronunciation_regression\integration_smoke\pronunciation_evidence.jsonl `
    --output outputs\stage10a_pronunciation_regression\integration_smoke\pronunciation_regression_baseline.jsonl
```

Compare current evidence:

```powershell
.\.venv\Scripts\python.exe -m j_speech_ops.pronunciation_regression_cli compare `
    --evidence outputs\stage10a_pronunciation_regression\integration_smoke\pronunciation_evidence.jsonl `
    --baseline outputs\stage10a_pronunciation_regression\integration_smoke\pronunciation_regression_baseline.jsonl `
    --output-dir outputs\stage10a_pronunciation_regression\integration_smoke
```

## Outputs

- `pronunciation_evidence.jsonl`
- `pronunciation_review_candidates.jsonl`
- `pronunciation_regression_baseline.jsonl`
- `pronunciation_regression_results.jsonl`
- `pronunciation_regression_review_candidates.jsonl`
- `pronunciation_regression_summary.json`

All outputs are deterministic, machine-readable, and include identifiers needed by a future Stage 10B human-review workflow.

## Limitations

- No automatic pitch-accent validation.
- No MOS or naturalness judgment.
- No speaker-similarity or emotion judgment.
- Canonical substring presence is not time-aligned phonetic evidence.
- Whisper and OpenJTalk remain fallible evidence sources.
- No human approval has occurred.
- No Human Review UI or release gate is implemented.

