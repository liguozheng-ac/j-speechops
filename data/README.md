# Data directory

No real audio is committed. Stage 2 uses the following local, ignored layout:

```text
data/raw/                         immutable source audio
data/interim/audio_segments/     derived mono 16 kHz PCM_16 WAV
data/manifests/                   updated parents and segment samples
data/reports/                     audio preparation, ASR, and curation reports
```

Large local audio, intermediate datasets, model weights, and generated audio must remain outside version control. Tiny automated-test WAV files are generated under pytest temporary directories rather than committed as binary fixtures.

Stage 3 adds `data/manifests/transcribed_segments.jsonl` and `data/reports/asr_run_report.jsonl`. Model caches belong under ignored `models/` or `work/`, never under tracked data.

Stage 4 adds `data/manifests/curated_segments.jsonl`, the PASS-only `data/manifests/usable_speech.jsonl`, and `data/reports/speech_curation_report.jsonl`. These are generated artifacts and remain ignored.

Stage 5 accepts UTF-8 source text under `data/raw/text/` and writes `data/manifests/tts_text_samples.jsonl`. This manifest contains `TTSTextSample` records and belongs to the independent TTS Production Pipeline; it contains no audio, speaker, ASR, or synthesis-run state.

Stage 6 writes the intermediate `data/manifests/normalized_text_samples.jsonl`, final `data/manifests/reading_prepared_text_samples.jsonl`, and operational `data/reports/text_preparation_report.jsonl`. These generated files remain ignored. No audio is produced.

Stage 7 writes `data/manifests/curated_tts_text_samples.jsonl`, PASS-only `data/manifests/synthesis_ready_text_samples.jsonl`, and `data/reports/tts_text_curation_report.jsonl`.

Stage 8 consumes the PASS-only manifest plus Stage 6/7 reports without modifying them. Its ignored output directory contains `synthesis_runs.jsonl`, success-only `generated_audio_manifest.jsonl`, `tts_generation_summary.json`, and native-rate generated WAV files. Model weights remain outside the repository.
