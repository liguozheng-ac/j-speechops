# TTS Text Curation Policy

## Policy boundary

`tts-text-curation-v1` uses discrete issues with evidence and the fixed priority:

```text
CRITICAL -> DROP
WARNING  -> REVIEW
none or INFO only -> PASS
```

Therefore `DROP > REVIEW > PASS`. No heterogeneous signals are combined into a risk, pronunciation, or quality score.

## DROP rules

- `missing_normalized_text`: a READING_PREPARED record has no usable normalized text.
- `missing_reading`: a READING_PREPARED record has no usable reading.
- `rights_restricted`: rights explicitly prohibit downstream use.
- `invalid_control_character`: text contains a non-whitespace control/format character outside the emoji-format allowance.

DROP means the record must not enter synthesis under the current policy. Uncertainty alone does not cause DROP.

## REVIEW rules

- `rights_unknown`: use rights are not cleared.
- `unresolved_reading_token`: Latin text remains in `reading_kana`.
- `unresolved_numeric_reading`: decimal digits remain in `reading_kana`.
- `remaining_kanji_in_reading`: Kanji remains in the Katakana-based reading.
- `unexpected_reading_symbol`: a character falls outside the lightweight reading baseline.
- `pronunciation_risk_term`: an unresolved watchlist surface occurs in raw or normalized text.
- `emoji_requires_review`: spoken/omission behavior requires product policy or a specialist.

Latin letters and digits in raw/normalized text are not risks by themselves. `JR`, `AI`, `Wi-Fi`, amounts, dates, and times may pass when the prepared reading is resolved. Legal thousands grouping fixed by Stage 6.1 is not a Stage 7 signal. Malformed grouping is not reparsed here; it routes only if the resulting reading triggers another rule.

Changes in the Stage 6 frontend regression fixture belong to CI/release validation. They are never converted into per-sample Stage 7 REVIEW issues.

Ordinary Japanese punctuation, long-vowel marks, spaces, and the documented symbol baseline are allowed. The inspector is a deterministic Unicode heuristic, not language identification or reading evaluation.

## INFO evidence

- `normalization_applied`: Stage 6 changed character form.
- `override_applied`: Stage 6 proved that a confirmed override was applied.

INFO does not change a PASS decision. An applied override is evidence that risk was addressed, not a reason for review.

## Interpretation

- PASS: no current rule triggered; not human-verified pronunciation.
- REVIEW: expected human-routing output, not a pipeline failure.
- DROP: current production policy prohibits progression.

Curation never mutates reading. Confirmed corrections return upstream to the Stage 6 pronunciation-override configuration.
