# Japanese Reading Policy

## Baseline provider

Stage 6 uses `OpenJTalkReadingProvider`, a model-neutral `ReadingProvider` implementation backed by the frontend portion of pyopenjtalk-plus 0.4.1.post9. It requests joined Katakana with `kana=True` and does not request or store phonemes.

For reproducibility, Marine, tsqyomi, Sudachi Kanji-reading correction, and the optional “何” prediction model are explicitly disabled. Provider-side normalization is also disabled because `JapaneseNormalizer` owns that layer.

## Override precedence

The JSON configuration contains a version, source, and unique `surface -> reading_kana` entries. Matching uses deterministic longest-surface-first replacement:

```text
explicit project override
  -> substituted Kana input
  -> OpenJTalk frontend
  -> final reading_kana
```

The report preserves the unmodified OpenJTalk baseline, final reading, applied surfaces/readings, occurrence counts, source, and SHA-256 of the exact override file. Overrides are not recursively reapplied.

## Interpretation and limits

The output is a baseline generated reading. It is not ground truth and does not prove pronunciation correctness. Proper names such as `佐伯`, place names, organizations, acronyms, brands, abbreviations, counters, dates, and novel terms can have multiple valid readings or dictionary errors.

OpenJTalk handles the pinned baseline cases for ordinary Japanese, Kanji, Katakana, integers, amounts, percentages, dates, and times reproducibly. These observed results must not be generalized into a complete Japanese number/calendar engine.

OpenJTalk treats a comma as punctuation. Before the Stage 6.1 patch, `3,500円` therefore produced `サン，ゴヒャクエン`. The normalizer now removes commas only from strictly thousands-grouped integer tokens before the provider runs, so the provider receives `3500円` and produces `サンゼンゴヒャクエン`. Malformed grouping such as `12,34円` remains unchanged and visible in both normalization and reading results.

Stage 6 reports unresolved output rather than copying raw Latin text into `reading_kana`. Stage 7 may route risky or ambiguous records to REVIEW, but Stage 6 keeps `curation_decision=undecided`.
