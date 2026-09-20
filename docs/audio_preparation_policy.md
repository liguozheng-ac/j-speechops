# Audio Preparation Policy

## Canonical prepared format

| Property | Policy |
|---|---|
| Sample rate | 16,000 Hz |
| Channels | 1 |
| In-memory samples | float32 |
| File container | WAV |
| File subtype | PCM_16 |
| Mono conversion | Arithmetic mean across all channels |
| Resampling | `torchaudio.functional.resample` |

Raw source audio is immutable. All written audio is derived data under `data/interim/audio_segments/` or the equivalent configured output root.

## VAD baseline

| Parameter | Value |
|---|---:|
| `threshold` | 0.5 |
| `min_speech_duration_ms` | 250 |
| `min_silence_duration_ms` | 100 |
| `speech_pad_ms` | 30 |
| `max_speech_duration_s` | 30.0 |

The 30-second cap is an operational guard against producing unusually long downstream ASR units. These values are reproducible starting parameters, not claims about optimal Japanese speech detection. Every processing report stores the effective values.

## Segment naming and timing

The normal child ID is `<parent_sample_id>__segNNNN`, with a one-based, zero-padded index in chronological VAD order. A safe parent ID therefore produces paths such as `data/interim/audio_segments/JP_0001__seg0001.wav`.

Waveform slicing uses half-open integer sample ranges `[start_sample, end_sample)`. Lineage seconds are calculated from those exact indices. Regions remain separate and are not concatenated.

## No-speech policy

When audio validation and preparation succeed but VAD returns no regions:

- parent stage: `audio_prepared`
- parent curation decision: `drop`
- decision reason: `no_speech_detected`
- children: none
- report status: `dropped`
- processing error count: zero; the report carries a warning

No other speech-ratio threshold triggers automatic curation in Stage 2.

## Text inheritance

Children receive an empty `TextInfo`. Parent `reference_text`, `asr_text`, normalized text, reading, and phonemes are not copied. Audio segmentation alone cannot establish which text span belongs to a region, including the single-region case.

