# Generated audio content QA policy

## Identity

- Policy: `generated-audio-content-qa-v1`
- Reading comparison: `japanese-reading-comparison-v1`
- Decision priority: `DROP > REVIEW > PASS`

The policy fingerprint is the deterministic SHA-256 of the canonical JSON policy configuration and is written into every result and batch summary.

## Decision rules

`DROP` is reserved for objective artifact failures: missing or empty files, undecodable WAV data, non-finite or silent samples, a Stage 8 SHA-256 mismatch, invalid stream metadata, or material sample-rate/channel/duration disagreement.

`REVIEW` is used when a valid artifact lacks exact canonical reading agreement. An empty Whisper transcript and any canonical reading mismatch are warnings. Neither CER nor ASR disagreement can cause DROP by itself.

`PASS` requires all of the following:

- artifact integrity checks succeeded;
- canonical expected and observed readings are exactly equal;
- no WARNING or CRITICAL issue exists.

PASS means content-QA-passed under current machine checks. It does not establish naturalness, pitch accent, prosody, speaker similarity, human preference, or release fitness.

An execution or provenance failure uses `status=failed` and `decision=undecided`. A missing `expected_reading_kana`, recoverable per-sample ASR exception, or reading-preparation failure is not silently guessed or converted to a release decision.

## Japanese comparison

Whisper text is normalized through the existing `JapaneseNormalizer`, converted to Kana through the existing OpenJTalk reading provider, and then compared to Stage 8 `expected_reading_kana`.

The comparison canonicalizer performs only presentation normalization:

- Unicode NFKC;
- Hiragana to Katakana;
- whitespace removal;
- Unicode punctuation removal.

It preserves long vowels, sokuon, moraic nasal, voicing, semi-voicing, and small Kana. Kanji is never compared directly to expected Kana. Stage 6 pronunciation overrides are not applied to the ASR transcript.

Character-level Levenshtein distance and `CER = distance / len(canonical expected)` are diagnostic evidence only. There is no CER PASS threshold and no overall quality score.

## Idempotency

The stable `qa_id` hashes the Stage 8 run ID, audio hash, expected stream metadata, and expected reading together with the QA policy fingerprint, ASR backend/model/config fingerprint, OpenJTalk provider fingerprint, and comparison-canonicalizer version. A completed result is reused only when the same QA identity exists, its audio path still identifies the current artifact, and the current audio hash still matches Stage 8 provenance. `--force` reruns QA. The CLI loads Whisper lazily, so an all-skipped rerun does not initialize the model.
