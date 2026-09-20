# Stage 1 Foundation

## Scope

Stage 1 establishes the shared data contract for J-SpeechOps. It models source speech samples, dataset metadata, provenance and rights, segmentation lineage, processing progress, and curation decisions. It also provides JSONL IO, lifecycle transitions, generated JSON Schema, examples, and tests.

The implementation is model-agnostic. It describes data and state but does not inspect physical audio or execute speech processing.

## Implemented components

- Pydantic v2 schemas in `src/j_speech_ops/schemas.py`
- Canonical JSONL read/write in `src/j_speech_ops/manifest.py`
- Explicit forward lifecycle transitions in `src/j_speech_ops/lifecycle.py`
- JSON Schema export in `scripts/export_schemas.py`
- Example manifest and dataset metadata in `examples/`
- Schema, lifecycle, lineage, curation, and manifest tests in `tests/`

## Out of scope

Stage 1 contains no audio decoding or validation, VAD or segmentation execution, ASR, Japanese normalization, kana conversion, G2P, TTS, evaluation metrics, pronunciation QA, model output, release decision, UI, database, model download, or model dependency.

## Running the project

From the repository root:

```powershell
python -m pip install -e ".[dev]"
python -m pytest -q
python scripts/export_schemas.py
```

## Manifest behavior

The canonical manifest is UTF-8 JSONL. Every physical line must contain exactly one JSON value that validates as a `SpeechSample`; blank lines are therefore invalid rather than silently skipped. `iter_manifest()` parses lazily. `write_manifest()` writes compact one-line objects and returns the number of records written.

Malformed JSON and Pydantic validation failures are wrapped as `ManifestValidationError`. The message and structured attributes include the manifest path and one-based line number.

## Known limitations

- Schema fields describe claimed audio properties; they are not compared with an audio file.
- The writer replaces its destination directly and does not provide transaction or append semantics.
- IDs are non-empty strings but are not governed by a project-wide naming convention yet.
- `decision_reasons` remains schema-level free-form for compatibility; Stage 4 emits documented policy issue codes.
- Lineage represents one direct parent and time interval; multi-parent derivations are outside Stage 1.
