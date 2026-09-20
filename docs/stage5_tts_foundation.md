# Stage 5 — TTS Production Foundation

## Scope

Stage 5 starts the independent TTS Production Pipeline:

```text
Raw Text
  -> Text Data Contract
  -> Japanese Normalization
  -> Reading / G2P
  -> Text Curation
  -> TTS Generation
```

Only the Text Data Contract is implemented. Stage 5 does not normalize Japanese, infer readings or phonemes, install a G2P system, download a TTS model, synthesize audio, or evaluate output.

The completed Speech Data Pipeline keeps `SpeechSample`; TTS source text uses `TTSTextSample`. They have no inheritance relationship, and ASR is not required upstream of TTS.

## Plain-text ingestion

The CLI accepts strict UTF-8 plain text with one physical line per sample. Universal `LF` or `CRLF` line endings are removed. Every other character—including leading/trailing spaces on a nonblank line, full-width punctuation, digits, emoji, Latin text, and Japanese-English mixtures—is preserved in `raw_text`.

An empty file or any empty/whitespace-only line raises `TextIngestionError` with the source filename and, when applicable, line number. Stage 5 does not silently skip invalid lines.

Default IDs are deterministic and human-readable:

```text
<SANITIZED_DATASET_ID>_<ONE_BASED_LINE_NUMBER:06d>
```

For example, `hospitality-demo` produces `HOSPITALITY_DEMO_000001`. A caller may provide a stable prefix. Dataset IDs containing no ASCII letters or digits use `TEXT_<DATASET_HASH>` as a deterministic fallback.

## CLI

```powershell
python -m j_speech_ops.ingest_text `
  --input examples/hospitality_text.txt `
  --dataset-id hospitality-demo `
  --language ja `
  --locale ja-JP `
  --source-name japanese-work-rehearsal `
  --rights-status cleared `
  --intended-use tts_production `
  --domain hospitality `
  --id-prefix HOSPITALITY `
  --output data/manifests/tts_text_samples.jsonl
```

The output is canonical `tts_text_samples.jsonl`. Running the same input with the same configuration produces identical IDs, ordering, fields, and serialized bytes. Canonical samples deliberately have no current-time field.

## Operational limits

- Only plain-text line ingestion is implemented; JSONL import adapters and general ETL are out of scope.
- A blank line stops the batch instead of being skipped.
- Source paths record the path supplied to ingestion; callers seeking portable manifests should use stable relative paths.
- Dataset-wide descriptive metadata is deferred until a TTS-specific need is established.
- No `SynthesisRun` exists yet. The stable text sample ID is the future one-to-many reference boundary.
