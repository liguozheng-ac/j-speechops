# Speech Data Curation Policy

## Policy identity

The default policy is `speech-curation-v1`. Every `SpeechCurationReport` records this version so regenerated datasets can be traced to the rules that routed them.

## Baseline thresholds

| Signal | Default | Outcome when triggered |
|---|---:|---|
| Minimum duration | 0.5 seconds | REVIEW below threshold |
| Maximum duration | 30.0 seconds | REVIEW above threshold |
| Minimum transcript density | 1.0 non-whitespace characters/second | REVIEW below threshold |
| Maximum transcript density | 20.0 non-whitespace characters/second | REVIEW above threshold |
| Minimum Japanese script ratio | 0.30 | REVIEW below threshold |
| Exact repetition count | 3 | REVIEW when the whole compact transcript repeats a unit this many times |
| Maximum repeated unit length | 20 characters | Limits units considered by the repetition heuristic |
| Require cleared rights | true | UNKNOWN -> REVIEW; RESTRICTED -> DROP |

These values are an operational baseline, not linguistic standards. They are centralized in `CurationPolicy` and configurable through the CLI.

## Hard DROP rules

- `empty_transcript`: `asr_text` is null, empty, or whitespace-only after the internal analysis view.
- `audio_missing`: the canonical path does not resolve to an accessible file.
- `audio_unreadable`: the file exists but cannot be read for hashing.
- `rights_restricted`: source rights are explicitly restricted.

Any hard-drop issue wins over simultaneous review issues.

## REVIEW rules

- `rights_unknown`: rights have not been cleared and the default policy requires clearance.
- `duration_too_short` / `duration_too_long`: declared duration falls outside the baseline.
- `duration_unavailable`: the manifest has no declared duration.
- `transcript_density_anomaly`: non-whitespace character count divided by positive declared duration falls outside the baseline.
- `low_japanese_script_ratio`: Hiragana, Katakana, and Kanji comprise less than 30% of Unicode letter/number characters.
- `excessive_text_repetition`: after removing whitespace and Unicode punctuation, the entire text is an exact repetition of a short unit.
- `duplicate_audio`: two or more analyzed records have identical SHA-256 file hashes; every member is reviewed.

Low Japanese-script ratio is deliberately code-switch tolerant: Latin product names, URLs, model names, and numbers are legitimate Japanese content and are never dropped by this signal. Equal transcript strings do not constitute duplicates.

## PASS, REVIEW, and DROP

- PASS: no policy rule triggered; exported to `usable_speech.jsonl`.
- REVIEW: one or more ambiguous warning signals triggered; retained only in the complete curated manifest.
- DROP: at least one explicit critical condition triggered; retained only in the complete curated manifest with evidence.

There is no numeric quality score and no confidence-to-correctness mapping. Stage 4 does not mutate or normalize `asr_text`, and it does not populate `normalized_text`, `reading_kana`, or `phonemes`.

**Curation heuristics are data-routing signals, not ground-truth ASR accuracy measurements.**

**PASS means no current curation rule was triggered. It does not guarantee transcript correctness.**
