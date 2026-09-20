# Stage 2 Audio Preparation

## Scope

Stage 2 turns source `SpeechSample` records into validated, uniformly encoded, traceable speech segments. It implements audio validation, decoding, mono conversion, 16 kHz resampling, Silero VAD, segmentation, parent lifecycle updates, child sample construction, batch manifests, operational reports, and a minimal CLI.

ASR and all text inference remain outside this stage.

## Audio validation and preparation

SoundFile decodes source audio as a two-dimensional float32 array. The validator checks that the path is an existing file, decoding succeeds, the frame and channel counts are positive, the sample rate is positive, duration is positive, and every sample is finite.

Channels are converted to mono by their arithmetic mean. TorchAudio then resamples the mono tensor to 16 kHz when needed. Processing stays float32 in memory. Every segment is written through SoundFile as mono PCM_16 WAV. The source file is never opened for writing.

WAV is the only format promised by the Stage 2 MVP. SoundFile may decode other formats supported by the installed libsndfile, but they are not part of the verified interface.

## VAD and segmentation

`SileroVAD` is a narrow adapter over the pip package's `load_silero_vad()` and `get_speech_timestamps()` APIs. It loads the PyTorch model locally and runs on CPU. It accepts a one-dimensional 16 kHz tensor and returns `SpeechRegion` values with half-open sample-index boundaries. Seconds are derived from indices for reporting and lineage.

Every region becomes a separate file. Segment IDs use a stable one-based suffix such as `JP_0001__seg0001`. Files are not concatenated because doing so would destroy original timing context. Unsafe filename characters in a parent ID are replaced and accompanied by an eight-character hash to avoid obvious collisions while retaining deterministic output.

## Lifecycle and text policy

Successful decode and VAD execution advance a parent from `ingested` to `audio_prepared`. A missing, corrupt, unreadable, empty, or non-finite source remains at its current stage, is marked `drop`, and receives a stable reason code.

If preparation succeeds but VAD detects no speech, the parent is `audio_prepared` and `drop` with `no_speech_detected`. That distinction records that the technical preparation completed even though the sample is unsuitable for the next stage.

Every written child starts at `audio_prepared` with an `undecided` curation decision. Children do not inherit any parent text fields, even when there is only one region, because VAD does not prove alignment. Provenance, rights, dataset identity, speaker, language, locale, and split are preserved.

## Batch error handling

Expected data problems raise `AudioValidationError` with a stable code. The batch converts that exception into a dropped parent and a structured report error, then continues. VAD contract violations, invalid lifecycle input, filesystem output failures, and other unexpected exceptions are not suppressed.

Current reason codes include:

- `audio_missing`
- `audio_unreadable`
- `audio_decode_failed`
- `audio_empty`
- `audio_invalid_sample_rate`
- `audio_invalid_channels`
- `audio_non_finite`
- `no_speech_detected`

## Outputs

Each run rewrites three JSONL files:

- `audio_prepared_sources.jsonl`: one updated parent per input record.
- `audio_segments.jsonl`: only child segments; this avoids sending parent and child audio together to a future ASR stage.
- `audio_preparation_report.jsonl`: one operational result per input parent, including source and prepared properties, exact VAD config, backend version, segment count, speech duration, speech ratio, child IDs, warnings, and errors.

The segment audio lives under `interim/audio_segments/`. Derived paths stored in the child manifest are POSIX-style paths relative to the dataset root. Re-running with the same input and config overwrites deterministic segment files and rewrites manifests, so rows do not accumulate.

## Running

```powershell
.\.venv\Scripts\python.exe -m j_speech_ops.audio_prepare `
    --manifest input.jsonl `
    --dataset-root . `
    --output-dir data
```

The CLI prints input, prepared, dropped, segment, and error totals.

## Limitations

- VAD boundaries are acoustic regions, not transcript alignments.
- No denoising, dereverberation, loudness normalization, enhancement, or clipping restoration is performed.
- There is no ASR or semantic validation.
- Output replacement is deterministic but not transactional across all files.
- Old unreferenced segment files from a run with different inputs are not garbage-collected; manifests remain authoritative.
- VAD parameters are a baseline and require later empirical evaluation on representative Japanese speech.

