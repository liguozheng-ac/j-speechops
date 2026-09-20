# Third-Party Components

Versions below are the packages resolved in the verified Windows/Python 3.14.7 environments through Stage 8. Core and TTS dependencies remain isolated.

| Component | Installed version | Purpose | Project | License | Classification |
|---|---:|---|---|---|---|
| Silero VAD | 6.2.2 | Speech-region inference | https://github.com/snakers4/silero-vad | MIT | Borrow |
| PyTorch | 2.9.1+cpu | Tensor runtime and Silero execution | https://pytorch.org | BSD-3-Clause | Borrow |
| TorchAudio | 2.9.1+cpu | Tensor-side resampling | https://github.com/pytorch/audio | BSD-2-Clause | Borrow |
| SoundFile | 0.14.0 | Audio decode, metadata, and PCM_16 WAV encode | https://github.com/bastibe/python-soundfile | BSD-3-Clause | Borrow |
| ONNX Runtime | 1.30.0 | Import-time compatibility dependency of silero-vad 6.2.2 sequence API | https://onnxruntime.ai | MIT | Borrow (compatibility only) |
| faster-whisper | 1.2.1 | Whisper inference API and feature/decoding pipeline | https://github.com/SYSTRAN/faster-whisper | MIT | Borrow |
| CTranslate2 | 4.8.2 | CUDA inference runtime | https://opennmt.net/CTranslate2/ | MIT | Borrow |
| Whisper large-v3 conversion | `Systran/faster-whisper-large-v3` | Stage 3 Japanese baseline model | https://huggingface.co/Systran/faster-whisper-large-v3 | MIT | Borrow |
| NVIDIA cuBLAS CUDA 12 | 12.9.2.10 | Windows GPU matrix operations required by CTranslate2 | https://developer.nvidia.com/cuda-zone | NVIDIA software license | Borrow |
| NVIDIA cuDNN CUDA 12 | 9.26.0.51 | Windows GPU neural-network runtime required by CTranslate2 | https://developer.nvidia.com/cudnn | NVIDIA software license | Borrow |
| jaconv | 0.5.0 | Unicode, width, Kana, and selected punctuation normalization | https://github.com/ikegami-yukino/jaconv | MIT | Borrow |
| pyopenjtalk-plus | 0.4.1.post9 | OpenJTalk frontend Python binding and bundled customized dictionary | https://github.com/tsukumijima/pyopenjtalk-plus | MIT for Python wrapper; bundled components below | Borrow |
| Modified Open JTalk | bundled with pyopenjtalk-plus | Japanese morphological/frontend implementation | https://github.com/r9y9/open_jtalk | Modified BSD | Borrow, frontend only |
| Customized OpenJTalk/NAIST-jdic dictionary | bundled with pyopenjtalk-plus wheel | System dictionary used by the reading frontend | pyopenjtalk-plus `pyopenjtalk/dictionary/` | Combined notices include NAIST BSD, UniDic Consortium BSD-style, and Open JTalk Modified BSD terms | Borrow |
| SudachiPy | 0.6.11, transitive | Required package dependency of pyopenjtalk-plus | https://github.com/WorksApplications/sudachi.rs/tree/develop/python | Apache-2.0 | Installed transitively; reading correction disabled |
| SudachiDict-core | 20260723, transitive | Required dictionary dependency of pyopenjtalk-plus | https://github.com/WorksApplications/SudachiDict | Apache-2.0 | Installed transitively; not selected as project baseline |
| Qwen3-TTS | 0.1.1 | Isolated CustomVoice inference API | https://github.com/QwenLM/Qwen3-TTS | Apache-2.0 | Borrow; `.venv-tts` only |
| Qwen3-TTS 12Hz 1.7B CustomVoice | checkpoint | Stage 8 Japanese/Ono_Anna baseline | https://huggingface.co/Qwen/Qwen3-TTS-12Hz-1.7B-CustomVoice | Apache-2.0 | Borrow; external model directory |
| PyTorch CUDA | 2.11.0+cu128 | Qwen GPU tensor runtime | https://pytorch.org | BSD-3-Clause | Borrow; `.venv-tts` only |
| TorchAudio CUDA | 2.11.0+cu128 | Qwen runtime dependency | https://github.com/pytorch/audio | BSD-2-Clause | Borrow; `.venv-tts` only |

The Stage 2 adapter calls Silero's pip APIs with `onnx=False`, so actual VAD inference uses the PyTorch CPU backend. ONNX Runtime is declared because importing `silero_vad` 6.2.2 otherwise fails when its package initializer imports the sequence API; J-SpeechOps does not select the ONNX VAD backend.

Stage 3 uses CTranslate2 CUDA directly; the CPU-only PyTorch build remains specific to Stage 2 Silero and does not imply ASR CPU execution. On Windows, cuBLAS and cuDNN wheels are installed inside the virtual environment. Their `bin` directories are exposed only to the active Python process. NVIDIA packages remain subject to their bundled license terms.

pyopenjtalk-plus is a derivative of r9y9/pyopenjtalk that integrates improvements from multiple forks and imports as `pyopenjtalk`. Its wheel bundles a customized dictionary based on mecab-naist-jdic with additional UniDic-derived material; the bundled `dictionary/COPYING` contains the applicable NAIST, UniDic Consortium, and Open JTalk notices. The wheel also contains the HTS Voice “Mei” under Creative Commons Attribution 3.0, but Stage 6 never invokes HTSEngine or speech synthesis. Marine and tsqyomi are not installed. Although SudachiPy/Core are mandatory package dependencies, J-SpeechOps explicitly sets `use_sudachi_kanji_yomi=False`; they are not exposed as a second ReadingProvider.

## Borrow / Adapt / Build

### Borrow

- Silero's trained VAD model and inference functions
- PyTorch tensor runtime
- TorchAudio resampling implementation
- SoundFile/libsndfile audio I/O
- faster-whisper's Whisper decoding implementation
- CTranslate2 CUDA runtime
- Whisper large-v3 converted model weights
- NVIDIA cuBLAS and cuDNN runtime libraries
- Python standard-library `hashlib` and Unicode metadata for Stage 4 signals
- Pydantic v2 and Python standard-library JSON/text/path utilities for the Stage 5 contract and ingestion
- jaconv character normalization and the pyopenjtalk-plus OpenJTalk frontend
- Qwen3-TTS 1.7B CustomVoice, PyTorch CUDA, and SoundFile for isolated synthesis

### Adapt

- Centralized Silero baseline parameters
- Decode, channel averaging, and resampling order
- Separate-region segmentation policy
- Portable derived-data layout and deterministic naming
- Fixed-language Japanese decoding configuration
- large-v3 CUDA/float16 operational baseline
- Process-local Windows CUDA DLL discovery
- Japanese-aware Unicode script, density, rights, duration, repetition, and duplicate-routing baselines
- Existing JSONL error-localization conventions adapted for text manifests
- Versioned character normalization, explicit pronunciation overrides, business-domain regression cases, and strict text lifecycle integration
- Fixed Japanese language/Ono_Anna voice, SDPA/BF16 runtime baseline, confirmed-override synthesis input, and stable seed/config provenance

### Build

- Audio validation contract and reason codes
- Audio preparation orchestration
- `SpeechRegion` representation and boundary checks
- Parent lifecycle and child `SpeechSample` integration
- Provenance and lineage preservation
- Per-sample batch error isolation
- `AudioPreparationReport` and report JSONL
- Source and child manifest generation
- CLI, synthetic audio tests, fake-VAD tests, and integration smoke test
- Model-neutral `ASRAdapter`, `ASRResult`, and `ASRSegment`
- Faster-whisper result mapping and lifecycle integration
- ASR batch retry/skip/force behavior
- `transcribed_segments.jsonl` and `asr_run_report.jsonl`
- Structured ASR data errors and fail-fast infrastructure errors
- ASR unit tests and real Japanese CUDA integration smoke test
- Versioned `CurationPolicy`, `CurationIssue`, and `SpeechCurationReport`
- Deterministic curation engine, usable-manifest export, CLI, summary, and synthetic tests
- Independent immutable `TTSTextSample` and strict text lifecycle
- Deterministic UTF-8 line ingestion, stable IDs, text-manifest IO, CLI, schema export, examples, and tests
- `JapaneseNormalizer`, `ReadingProvider`, and `OpenJTalkReadingProvider`
- Layered text-preparation batch pipeline, reports, override provenance/fingerprint, CLI, and 23-case regression fixture
- Model-neutral TTS contracts, subprocess adapter, synthesis planner/text builder, `SynthesisRun`, success-only generated-audio manifest, WAV validation/hashing, Stage 8 CLI, and GPU integration test
