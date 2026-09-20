# Pronunciation Risk Watchlist

## Purpose

`config/japanese_pronunciation_risks.json` is a small, explicit, project-maintained list of surfaces whose pronunciation needs confirmation. Each entry records `surface`, `category`, `reason`, and an optional note. The bundled entries are examples/demos, not a complete Japanese people/place/brand dictionary and not automatic NER.

The exact file bytes receive a SHA-256 fingerprint stored in every curation report and batch summary. Duplicate surfaces or invalid entries fail initialization.

## Watchlist versus reading override

These files encode different knowledge:

```text
Pronunciation risk watchlist:
  "this surface may be ambiguous; route it to REVIEW"

Reading override:
  "a reviewer confirmed this exact reading; apply it in Stage 6"
```

Stage 7 searches raw and normalized text, reports one match per configured surface with its occurrence count, and routes unresolved matches to REVIEW. It does not guess entities or pronunciations.

An exact matching Stage 6 `AppliedReadingOverride` can suppress the warning only when the companion preparation report proves the override covered every detected occurrence. String similarity or the mere existence of an override config entry is not enough. Applied override details and their configuration fingerprint are copied from Stage 6 provenance; Stage 7 never recomputes them.

## Feedback loop

For example, a configured `佐伯` risk routes to REVIEW even if OpenJTalk produced `サエキ`. If a specialist confirms `サイキ`, add `佐伯 -> サイキ` to the Stage 6 override file, rerun preparation, and recurate. The new applied-override evidence resolves the matching watchlist warning. Stage 7 itself never changes `サエキ` to `サイキ`.
