# TTS Text Contract

## Canonical object

`TTSTextSample` represents source text before synthesis. It is a separate frozen Pydantic v2 model—not a subtype or partial instance of `SpeechSample`.

```text
TTSTextSample
├── schema_version
├── sample_id / dataset_id
├── language / locale
├── text
│   ├── raw_text
│   ├── normalized_text
│   ├── reading_kana
│   └── phonemes
├── source
│   ├── source_type
│   ├── source_name
│   ├── source_id
│   └── source_path
├── rights_status
├── intended_use / domain
├── state
│   ├── stage
│   ├── curation_decision
│   └── decision_reasons
└── metadata
```

`rights_status` and `curation_decision` reuse the general Stage 1 enums without introducing speech inheritance. `source_type` is one of `manual`, `file`, `dataset`, `application`, `generated`, or `unknown`. Intended use is a lightweight `tts_production`, `benchmark`, `demo`, `application`, or `unknown` value; domain remains optional free text.

## Text fidelity and immutability

`raw_text` must contain at least one non-whitespace character. Its original characters are retained exactly after ingestion removes the physical line ending. The full canonical object is frozen, so later stages use copy-on-write and must retain `raw_text` while populating derived fields.

At Stage 5, `normalized_text`, `reading_kana`, and `phonemes` remain null. Stage 6 populates the first two through separate copy-on-write transitions while leaving `phonemes` null. Stage 7 changes only `state.stage`, `state.curation_decision`, and `state.decision_reasons`; prepared text remains identical. A tuple represents future phonemes internally for immutability and serializes to a JSON array. No Japanese analysis traces are stored on the canonical object; they remain in operational reports.

## Independent lifecycle

```text
raw -> normalized -> reading_prepared -> curated
```

Only idempotent or adjacent forward transitions are valid. Stage 5 creates `raw + undecided` records. Stage 7 performs the final `reading_prepared -> curated` edge and exports only `curated + pass` records to the synthesis-ready manifest.

## Manifest and schema

`tts_text_samples.jsonl` contains exactly one validated `TTSTextSample` per UTF-8 line. Reader errors identify the file, one-based line, and sample ID when it can be parsed. Duplicate sample IDs are invalid. Read/write round trips preserve model equality and ordering.

The exported schema is `schemas/tts_text_sample.schema.json`. SpeechSample and DatasetMetadata schemas remain byte-for-byte unchanged.

## Synthesis boundary

Canonical text contains no speaker, voice, reference audio, model, model version, temperature, seed, or output audio path. A future `SynthesisRun` will reference `TTSTextSample.sample_id`, allowing one canonical sentence to produce any number of outputs with different configurations.
