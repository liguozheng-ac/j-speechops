# Stage 6 — Japanese Text Normalization and Reading Preparation

## Scope and architecture

Stage 6 is a deterministic, offline Japanese text frontend:

```text
TTSTextSample(raw)
  -> JapaneseNormalizer
  -> TTSTextSample(normalized)
  -> ReadingProvider
  -> OpenJTalkReadingProvider
  -> TTSTextSample(reading_prepared)
```

Normalization and reading are independently represented and reported. The canonical object remains Stage 5 `TTSTextSample`; no duplicate schema was created. Each update is copy-on-write, and `raw_text` remains byte-for-byte equal at the Python string level.

Stage 6 does not perform text curation, set PASS/REVIEW/DROP, generate phonemes, estimate pitch accent, invoke an LLM, load a TTS model, or create audio.

## Outputs

- `manifests/normalized_text_samples.jsonl`: the successful normalization-layer snapshots at `normalized`.
- `manifests/reading_prepared_text_samples.jsonl`: samples with a successful Katakana reading at `reading_prepared`.
- `reports/text_preparation_report.jsonl`: policy, transformations, provider and package versions, dictionary identity, provider settings, baseline and final reading, override evidence and fingerprint, warnings, and errors.

If reading fails for a RAW sample, its normalized snapshot remains in the first manifest and the report status is `normalized_only`. Per-sample normalization/reading failures do not stop unrelated samples. Provider, dictionary, override-config, or policy initialization failures stop the batch before output processing.

## Lifecycle and idempotency

The ordinary path strictly executes:

```text
raw -> normalized -> reading_prepared
```

An existing normalized record reuses `normalized_text` by default and proceeds to reading. An existing reading-prepared record is skipped. `--force` recalculates fields without moving lifecycle state backward. If forced recomputation of an already reading-prepared sample fails, its prior prepared canonical record is preserved and the failed attempt is reported.

## Runtime baseline

- Policy: `japanese-text-prep-v1.1`
- Normalizer: jaconv 0.5.0, NFKC-based character normalization plus documented numeric-thousands-separator, whitespace, and Latin-hyphen rules
- Reading package: pyopenjtalk-plus 0.4.1.post9, imported as `pyopenjtalk`
- Provider output: `g2p(..., kana=True)`
- Disabled: Marine, tsqyomi, Sudachi Kanji-reading correction, optional “何” predictor, provider-side Unicode normalization
- Dictionary: the customized OpenJTalk/NAIST-jdic derivative bundled in the pyopenjtalk-plus wheel; it exposes no independent reliable version, so the report records identity and a null dictionary version

The package contains synthesis APIs and an HTS voice, but J-SpeechOps imports only the frontend call and never invokes `tts()` or HTSEngine.

## Failure boundaries and limitations

A final reading must be nonempty, contain Katakana, and contain no unresolved Latin letters or digits. Otherwise the record remains normalized. Emoji may remain alongside a valid Japanese reading; an emoji-only input is unresolved.

OpenJTalk readings are generated baselines, not human-verified pronunciations. Proper nouns, names, organizations, acronyms, brands, new words, and numeric context may require review or an explicit override. Stage 7 should route these risks; Stage 6 does not curate them.

## Regression verification

`tests/fixtures/japanese_text_frontend_regression.json` pins 27 normalization/reading pairs covering hospitality sentences, width forms, whitespace, emoji, Kanji, Katakana, integers, valid and malformed grouped amounts, percentages, dates, irregular day/month/time readings, acronyms, a brand token, and a proper name. It is a dependency-upgrade regression fixture, not a TTS quality benchmark.
