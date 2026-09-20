# Data Schema

Pydantic models are the source of truth. JSON field names use lower snake case, and enum values are lowercase strings. Unknown fields are rejected so misspellings do not silently enter the canonical manifest. `schema_version` is currently fixed at `1.0`.

## SpeechSample

`SpeechSample` is one source-data record. `sample_id` identifies it for later associations, while `dataset_id` identifies its owning dataset. `language` is the broad language tag and `locale` may hold a regional tag. The nested `audio`, `text`, `speaker`, `source`, `lineage`, and `state` objects keep concerns explicit. `split` controls dataset use. `metadata` is reserved for JSON-compatible extensions.

The object does not contain synthesis runs, generated audio, model identity, evaluation metrics, pronunciation findings, or release decisions. Future objects can reference it by `sample_id`.

## AudioInfo

`path` is the logical or filesystem path recorded in the manifest. `format`, `duration_sec`, `sample_rate_hz`, and `channels` are optional because records may be created before audio preparation. Numeric constraints reject negative duration and non-positive rates or channel counts, but no file is opened and no claimed value is verified in Stage 1.

## TextInfo

- `reference_text`: annotation supplied by a person or source dataset.
- `asr_text`: reserved for a later transcription result.
- `normalized_text`: reserved for later Japanese text normalization.
- `reading_kana`: reserved for a later reading representation.
- `phonemes`: reserved for later G2P output.

All text representations are optional because processing may be incomplete. Stage 1 performs none of the transformations.

## SpeakerInfo

`speaker_id` is an optional dataset-scoped identifier. Stage 1 deliberately excludes embeddings, personas, and detailed speaker profiles.

## SourceInfo

`source_name` is required. `source_record_id` maps the sample back to the source dataset, and `source_url` can record a discovery location. `license` records the declared license. `rights_status` is `unknown`, `cleared`, or `restricted`; its default is `unknown`, so availability never implies training permission.

## LineageInfo

`parent_sample_id` links a derived record to its direct parent. A segment uses `segment_start_sec` and `segment_end_sec` together, requires a parent, rejects negative offsets, and requires end to be greater than or equal to start. The parent cannot be the sample itself. These fields only record lineage; they do not perform segmentation.

## ProcessingState

`stage` describes completed source-data processing: `ingested`, `audio_prepared`, `transcribed`, `text_prepared`, or `curated`. `curation_decision` independently records `undecided`, `pass`, `review`, or `drop`. `decision_reasons` is a free-form list of short reason strings. Dataset curation uses `drop`; future model-output acceptance may use a separate `reject` decision outside this object.

## Dataset split

`split` is one of `unsplit`, `train`, `validation`, `test`, or `benchmark`. `benchmark` is distinct from ordinary test data so critical regression cases are not accidentally treated as a general evaluation split.

## DatasetMetadata

`DatasetMetadata` describes a manifest as a whole. It contains a stable `dataset_id`, display `name`, dataset `version`, optional `description`, `default_language`, optional dataset `license`, provenance `source`, and the `manifest` location. Its extensible `metadata` follows the same strict JSON-compatibility rule as sample metadata.

## Metadata compatibility

Metadata values may contain JSON null, booleans, strings, integers, finite floats, arrays, and string-keyed objects recursively. Sets, tuples, custom instances, non-string keys, NaN, and infinity are rejected during model validation.

