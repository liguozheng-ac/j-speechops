# Stage 7 — Japanese TTS Text Curation

## Scope and architecture

Stage 7 consumes existing Stage 6 outputs; it never reruns normalization or reading preparation:

```text
reading_prepared_text_samples.jsonl
  + text_preparation_report.jsonl
  + japanese_pronunciation_risks.json
  -> deterministic risk inspection
  -> PASS / REVIEW / DROP
  -> CURATED
```

It asks whether a prepared text may proceed under the current production policy. It does not score OpenJTalk, verify pronunciation, modify `normalized_text` or `reading_kana`, generate phonemes/audio, invoke a TTS model, or use an LLM/NER system.

## Outputs

- `manifests/curated_tts_text_samples.jsonl`: every successfully curated canonical sample, including PASS, REVIEW, and DROP.
- `manifests/synthesis_ready_text_samples.jsonl`: PASS samples only; this is the model-neutral Stage 8 input.
- `reports/tts_text_curation_report.jsonl`: issues, evidence, preparation/provider/override provenance (including Stage 6 and provider warnings/errors), reading-script inspection, risk terms, policy/watchlist fingerprints, decisions, and failures.

Only `state.stage`, `state.curation_decision`, and `state.decision_reasons` may change. Sample identity, source text, normalized text, reading, phonemes, locale, rights, intended use, domain, provenance, and metadata remain unchanged. `phonemes` must remain null.

## Input integrity and failure boundaries

Manifest and preparation report rows join by `sample_id`. Duplicate manifest/report IDs, a completely unrelated report, invalid configuration, or contradictory same-ID text/reading provenance fail the batch. A safely localized missing companion report or unexpected input stage is recorded as a failed sample while unrelated samples continue.

`READING_PREPARED` records with missing normalized text or reading are curated as DROP because their canonical state contradicts their content. REVIEW is not a failure: it is the expected route when deterministic inspection cannot establish production readiness.

## Lifecycle and reruns

The ordinary transition is:

```text
READING_PREPARED -> CURATED
```

Existing CURATED records are skipped by default. `--force` reruns policy inspection and may update the decision/reasons while keeping every prepared text field unchanged. No Stage 7 path creates a synthesized state.

## CLI

```powershell
python -m j_speech_ops.curate_tts_text `
  --manifest data/manifests/reading_prepared_text_samples.jsonl `
  --prep-report data/reports/text_preparation_report.jsonl `
  --risk-watchlist config/japanese_pronunciation_risks.json `
  --output-dir data
```

Add `--force` only for explicit recuration. The console summary reports ordinary counts: total, PASS, REVIEW, DROP, failed, top issue codes, override-applied samples, and pronunciation-risk samples. It emits no quality or accuracy score.

## Human-review feedback loop

Stage 7 never edits a risky reading. A language specialist confirms the desired pronunciation, the confirmed mapping is added to the Stage 6 reading-override configuration, Stage 6 is rerun, and Stage 7 then consumes the new report. Explicit applied-override evidence can suppress the matching watchlist warning.

```text
Stage 7 REVIEW
  -> human confirms reading
  -> Stage 6 pronunciation override
  -> rerun preparation
  -> Stage 7 recuration
```

PASS means only that no current policy rule fired. It is neither ground-truth pronunciation validation nor a guarantee that future synthesized audio will be correct.
