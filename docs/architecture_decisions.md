# Architecture Decisions

## ADR 1 — JSONL is the canonical manifest

**Decision:** Store one `SpeechSample` JSON object per UTF-8 line.

**Rationale:** JSONL supports streaming, record-level diagnostics, nested typed data, ordinary source-control review, and incremental processing without loading a whole dataset. CSV cannot faithfully preserve the nested contract. Model-specific formats will be derived through future adapters rather than becoming canonical.

## ADR 2 — Processing and curation state are separate

**Decision:** Keep `stage` and `curation_decision` as independent fields in `ProcessingState`.

**Rationale:** Processing progress answers what operations have occurred; curation answers whether the sample should be retained or reviewed. Combining them would create ambiguous states and make re-review difficult. Dataset curation uses `drop`, distinct from a future model-output `reject` decision.

## ADR 3 — SpeechSample lifecycle ends at CURATED

**Decision:** The last `ProcessingStage` is `curated`.

**Rationale:** `SpeechSample` represents source speech data. Synthesis and evaluation are repeatable runs with their own identity and lifecycle, potentially many per sample. Putting those states on the source sample would collapse one-to-many relationships and mix data readiness with model results.

## ADR 4 — Schema validation does not inspect audio

**Decision:** Validate declared field types and numeric bounds only; do not check file existence, decoding, duration, sample rate, or channels.

**Rationale:** A portable manifest may refer to storage unavailable on the validation machine, and schema validation must stay deterministic and free of audio dependencies. Physical audio validation belongs to Stage 2.

## ADR 5 — BENCHMARK is a dedicated split

**Decision:** Preserve `benchmark` alongside train, validation, and test.

**Rationale:** Critical Japanese regression cases have a stable quality-gate purpose and should not be conflated with a general test set. The explicit value reduces accidental training use and clarifies reporting.

## ADR 6 — Transitions follow explicit pipeline edges

**Decision:** Allow an idempotent same-stage operation or an explicitly declared forward edge. The Speech Data path permits `transcribed -> curated`; the optional text-preparation path remains `transcribed -> text_prepared -> curated`. Reject undeclared forward skips and all backward transitions.

**Rationale:** Explicit edges make missing processing evidence visible without falsely requiring Japanese normalization in the Speech Data Pipeline. Same-stage transitions are safe for retryable jobs. J-SpeechOps does not implement rollback or a workflow engine.

## ADR 7 — Strict, extensible records

**Decision:** Reject unknown typed fields while providing a strictly JSON-compatible `metadata` object.

**Rationale:** Canonical fields should fail fast on spelling or version mistakes. Controlled extension remains possible without accepting Python-only objects that cannot round-trip through JSONL.

## ADR 8 — SoundFile owns Stage 2 WAV I/O

**Decision:** Use SoundFile for decoding, metadata inspection, and PCM_16 WAV writing. Do not use TorchAudio's load/save functions.

**Rationale:** Audio I/O and tensor transformations remain separate, and the pipeline does not acquire a TorchCodec or FFmpeg dependency for its WAV MVP. SoundFile also exposes the libsndfile properties needed for validation.

## ADR 9 — Prepared audio is 16 kHz mono

**Decision:** Average channels to mono, resample to 16 kHz, process as float32, and write PCM_16 WAV.

**Rationale:** This is Silero's supported and efficient input rate and gives the future ASR boundary a single reproducible representation. Arithmetic channel mean is deterministic, generalizes to more than two channels, and is directly testable.

## ADR 10 — Load Silero through its pip API

**Decision:** Use the installed package's `load_silero_vad()` and `get_speech_timestamps()` functions with the PyTorch backend. Do not use `torch.hub`.

**Rationale:** The package API avoids runtime GitHub repository downloads and makes the dependency version explicit. Silero 6.2.2 also requires ONNX Runtime to import its sequence module, even though this pipeline selects `onnx=False`; the compatibility dependency is declared and documented.

## ADR 11 — VAD is CPU-first

**Decision:** Require CPU tensors at the VAD adapter boundary.

**Rationale:** VAD is lightweight, CPU behavior is reproducible across the target Windows environment, and GPU configuration would add complexity without a demonstrated Stage 2 need.

## ADR 12 — Speech regions remain separate

**Decision:** Write one child WAV and `SpeechSample` per VAD region instead of concatenating regions.

**Rationale:** Separate regions preserve source timing and exact lineage, avoid artificial joins, and retain context for future alignment and human review.

## ADR 13 — Parent and child manifests are separate

**Decision:** Emit `audio_prepared_sources.jsonl` for updated parents and `audio_segments.jsonl` for children.

**Rationale:** The future ASR input is unambiguously the segment manifest. A combined file could cause both a full parent recording and its segments to be transcribed, duplicating content.

## ADR 14 — VAD children do not inherit parent text

**Decision:** Construct every segment child with an empty `TextInfo`.

**Rationale:** Acoustic speech detection does not align a parent transcript to region boundaries. Copying the full transcript—even to one detected region—would assert evidence Stage 2 does not possess.

## ADR 15 — Expected data failures are isolated per sample

**Decision:** Convert known decode and validation failures to a dropped parent plus structured report error, then continue the batch. Let unexpected exceptions propagate.

**Rationale:** One bad recording should not block unrelated data, while broad exception suppression would hide programming defects and corrupt operational evidence. No-speech is recorded as a warning and dropped curation outcome after successful preparation.

## ADR 16 — J-SpeechOps has three independent pipelines

**Decision:** Document Speech Data, TTS Production, and Quality & Release as related but independently entered pipelines. Place Stages 3 and 4 only in Speech Data.

The documented top-level boundaries are:

1. Speech Data Pipeline
2. TTS Production Pipeline
3. Quality & Release Pipeline

**Rationale:** ASR produces transcripts from source speech, while TTS can begin with an independent text dataset. Treating ASR as a mandatory TTS upstream step would misrepresent both data lineage and product architecture. Evaluation is a separate quality responsibility rather than the whole system.

## ADR 17 — ASR has a model-neutral boundary

**Decision:** Batch processing depends on `ASRAdapter` and J-SpeechOps `ASRResult`/`ASRSegment`, not on faster-whisper objects.

**Rationale:** The baseline can later be compared with a SenseVoice adapter using the same canonical samples, reports, and orchestration. It also keeps the core callable from a future API or product layer without embedding model-specific types.

## ADR 18 — Stage 3 is fixed to Japanese CUDA/float16

**Decision:** Require `language=ja`, CUDA device 0 by default, and float16. Do not silently fall back to CPU.

**Rationale:** This stage establishes one reproducible baseline for the target RTX 5080 environment. A hidden fallback would make runtime and performance evidence misleading. Unsupported hardware or runtime libraries are infrastructure failures and stop the batch.

## ADR 19 — Faster-whisper VAD remains disabled

**Decision:** Always call faster-whisper with `vad_filter=False`.

**Rationale:** Stage 2 owns VAD boundaries and segmentation lineage. A second VAD pass could silently trim or remove audio and break the meaning of the recorded parent offsets.

## ADR 20 — ASR writes only asr_text

**Decision:** On success, mutate only `text.asr_text` and advance `audio_prepared` to `transcribed`.

**Rationale:** Reference annotations and future normalization, reading, and phoneme fields have distinct provenance. ASR must not overwrite them or imply that transcription is text curation.

## ADR 21 — Skip existing transcripts unless forced

**Decision:** Skip a transcribed input by default and reuse a prior output transcript for the same sample ID and audio path. `--force` explicitly reruns inference; output files are rewritten.

**Rationale:** large-v3 inference is expensive, and reruns must not duplicate rows or overwrite evidence accidentally. Explicit force makes replacement intentional while the lifecycle transition remains idempotent.

## ADR 22 — Operational ASR results remain outside SpeechSample

**Decision:** Keep backend, model config, detected-language confidence, segment count, and errors in `ASRRunReport`. Store only aggregate `asr_text` on the canonical sample.

**Rationale:** Operational evidence describes one processing run and may vary across models. Keeping it separate preserves the canonical sample identity and supports future fair backend comparisons without adding model-specific fields.

## ADR 23 — Stage 3 does not evaluate ASR

**Decision:** Do not compute CER, WER, Kana-CER, rankings, or acceptance gates.

**Rationale:** The current responsibility is transcript production. Evaluation requires a separate policy for reference provenance, normalization, and acceptance thresholds and belongs to later quality work.

## ADR 24 — Speech curation precedes usable-data export

**Decision:** Run Stage 4 after ASR and export a separate `usable_speech.jsonl` containing PASS records only.

**Rationale:** The complete curated manifest must preserve REVIEW and DROP evidence, while downstream consumers need an unambiguous operational subset. The export is called usable speech, not a training manifest, because no fine-tuning workflow exists yet.

## ADR 25 — Ambiguous curation signals route to REVIEW

**Decision:** Reserve DROP for empty transcripts, inaccessible audio, and explicitly restricted rights. Route duration, density, script-ratio, repetition, unknown-rights, and duplicate signals to REVIEW.

**Rationale:** These heuristics identify risk but cannot establish transcript correctness or unusability. Conservative routing prevents irreversible over-cleaning and leaves room for a future human-review process.

## ADR 26 — Curation has no aggregate quality score

**Decision:** Emit discrete issues, evidence, and a priority decision (`DROP > REVIEW > PASS`); do not combine heterogeneous signals into a numeric score.

**Rationale:** Rights, duration, character density, script mix, and duplication have no justified common linear scale. A score would create false precision and obscure the actual reason for routing.

## ADR 27 — Rights participate in data routing

**Decision:** DROP records with restricted rights and, under the default policy, REVIEW records whose rights are unknown.

**Rationale:** Technical fitness does not override authorization to use data. Unknown rights are ambiguous rather than proven prohibited, so conservative review is appropriate.

## ADR 28 — Exact audio duplicates use SHA-256 content hashes

**Decision:** Hash readable segment files with standard-library SHA-256 and route every member of an equal-hash group to REVIEW. Do not deduplicate on transcript text.

**Rationale:** A content hash provides deterministic exact-byte identity without a new dependency. Repeated transcripts can occur naturally, and even exact audio repetitions may be intentional, so Stage 4 does not auto-drop a member.

## ADR 29 — Transcript normalization remains out of scope

**Decision:** Analyze a temporary whitespace/punctuation view where needed but never write it to `asr_text`, `normalized_text`, or any other text field.

**Rationale:** Curation observes and routes canonical data. Japanese normalization has different provenance and belongs to the independent TTS Production Pipeline.

## ADR 30 — PASS is policy silence, not ASR correctness

**Decision:** Define PASS as no current curation rule being triggered.

**Rationale:** Stage 4 has no human reference or accuracy evaluation. Its heuristics are operational signals and cannot establish that a transcript is correct.

## ADR 31 — Speech and TTS text use separate canonical contracts

**Decision:** Model source speech as `SpeechSample` and pre-synthesis text as `TTSTextSample`. Neither inherits from the other, and a text record never uses a null or placeholder audio object.

**Rationale:** TTS input can originate from authored scripts, applications, datasets, or files without any recording. Sharing an inheritance hierarchy would make speech-only fields appear relevant to text and weaken both contracts.

## ADR 32 — TTS input does not require upstream ASR

**Decision:** Let the TTS Production Pipeline begin directly with raw text and a `TTSTextSample` manifest.

**Rationale:** ASR is one possible provenance path, not a TTS prerequisite. The Speech Data and TTS Production pipelines remain independently enterable and may be related later through IDs and provenance.

## ADR 33 — Canonical raw text is immutable

**Decision:** Make `TTSTextSample` and its nested text payload frozen. Later stages must create a validated copy that retains `raw_text` alongside derived fields.

**Rationale:** Exact source text is audit evidence. Overwriting it with normalized or reading text would lose provenance and make transformations impossible to review or reproduce.

## ADR 34 — Synthesis configuration is not canonical text data

**Decision:** Exclude model names, versions, voices, speakers, seeds, decoding settings, and output paths from `TTSTextSample`.

**Rationale:** These values describe a future synthesis execution rather than the source text. Keeping them separate avoids model coupling and lets the text contract remain stable.

## ADR 35 — One text sample may have many synthesis runs

**Decision:** Preserve `sample_id` as the future foreign-key boundary and store no unique output-audio relationship on the text sample.

**Rationale:** The same sentence may be synthesized with multiple models, voices, seeds, or settings. Future `SynthesisRun` records can reference one text ID without rewriting canonical text.

## ADR 36 — TTS text lifecycle is independent and strict

**Decision:** Define `raw -> normalized -> reading_prepared -> curated`, allowing idempotence or one explicit forward edge only.

**Rationale:** Audio preparation and transcription states have no meaning for raw TTS text. Strict forward transitions expose missing transformation evidence and prevent Stage 5 from claiming that normalization or reading preparation occurred.

## ADR 37 — Normalization and reading are separate layers

**Decision:** Materialize a `normalized` canonical boundary before invoking a reading provider and preserve separate `NormalizationResult` and `ReadingResult` evidence.

**Rationale:** Character-form normalization and grapheme-to-reading conversion have different policies and failure modes. A reading failure must leave a truthful normalized record rather than hiding both operations behind one black box.

## ADR 38 — Reading Kana is the current canonical frontend output

**Decision:** Store a Katakana-based baseline in `reading_kana` while keeping backend details and rule traces in `TextPreparationReport`.

**Rationale:** Kana is inspectable and can be consumed independently by future TTS backends. It avoids prematurely locking the canonical contract to a model-specific phoneme inventory.

## ADR 39 — Phonemes remain unset

**Decision:** Leave `TTSTextContent.phonemes` null in Stage 6 even though OpenJTalk can emit a phoneme sequence.

**Rationale:** A phoneme output is meaningful only with an inventory, scheme, backend, and version. The current contract lacks that provenance, so storing bare phonemes would be ambiguous.

## ADR 40 — Explicit pronunciation overrides precede dictionary reading

**Decision:** Replace configured surfaces with reviewed Kana before the OpenJTalk frontend, record every applied override, and fingerprint the exact override file.

**Rationale:** Names, brands, acronyms, and new words can be ambiguous or out of vocabulary. Explicit project evidence must take precedence while remaining auditable and reproducible.

## ADR 41 — Reading generation is TTS-model independent

**Decision:** Hide pyopenjtalk-plus behind `ReadingProvider` and generate no audio or synthesis configuration.

**Rationale:** Qwen3-TTS, CosyVoice, or another future backend should be able to consume the same prepared text. Frontend provenance must not be inferred from a synthesis model's behavior.

## ADR 42 — Stage 6 uses deterministic offline frontend settings

**Decision:** Use pyopenjtalk-plus with `kana=True` while explicitly disabling Marine, tsqyomi, Sudachi reading correction, and the optional “何” prediction model. Apply external normalization only once before the provider.

**Rationale:** The first baseline must run offline and reproduce exact results. Additional statistical or alternate reading systems would introduce unversioned behavior and violate the single-provider Stage 6 scope.

## ADR 43 — Text curation never mutates prepared text

**Decision:** Stage 7 may change only the text sample lifecycle stage, curation decision, and decision reasons. It never rewrites raw, normalized, reading, or phoneme fields.

**Rationale:** Risk routing is operational evidence, not a pronunciation-correction engine. Mutating prepared content would erase Stage 6 provenance and make human review unauditable.

## ADR 44 — Pronunciation ambiguity routes to human review

**Decision:** Treat explicit pronunciation-risk terms as WARNING and route them to REVIEW, never automatically DROP them or guess an alternative reading.

**Rationale:** A deterministic first version cannot reliably infer the identity or intended reading of a name. REVIEW preserves potentially valid data while requesting qualified confirmation.

## ADR 45 — Risk watchlists and reading overrides remain separate

**Decision:** Store unconfirmed ambiguity in a Stage 7 risk watchlist and confirmed readings in the Stage 6 override configuration. A warning is suppressed only by applied-override provenance for the same surface and occurrences.

**Rationale:** Risk and confirmed knowledge have opposite meanings. Combining them would make it impossible to distinguish “needs review” from “reviewed and resolved.”

## ADR 46 — Confirmed pronunciation corrections return upstream

**Decision:** Human-confirmed corrections are added as Stage 6 pronunciation overrides, followed by preparation and recuration; Stage 7 does not patch reading output.

**Rationale:** This maintains one owner for reading generation and a clean evidence chain from confirmation through applied override to curation.

## ADR 47 — TTS-text PASS is policy silence, not ground truth

**Decision:** Define Stage 7 PASS as no current CRITICAL or WARNING issue. Do not publish a pronunciation or quality score.

**Rationale:** Deterministic structural and watchlist checks cannot prove linguistic correctness or predict future synthesized-audio quality.

## ADR 48 — Only PASS text is synthesis-ready

**Decision:** Export a model-neutral `synthesis_ready_text_samples.jsonl` containing only PASS records while preserving PASS, REVIEW, and DROP in the complete curated manifest.

**Rationale:** Future synthesis needs an unambiguous canonical input without losing review/drop evidence or coupling the dataset to a specific TTS model.

## ADR 49 — TTS generation is separate from the text lifecycle

**Decision:** Keep `TTSTextSample` at `CURATED + PASS`; never add synthesis status, model, speaker, seed, latency, or output audio to it.

**Rationale:** Generation is a repeatable operation, not a new state of canonical source text. Keeping operational results separate preserves the established schema and lifecycle.

## ADR 50 — One text may produce many synthesis runs

**Decision:** Represent each backend/model/speaker/config/seed/text-strategy combination as an independent `SynthesisRun` linked by `sample_id`.

**Rationale:** Different voices, settings, and future backends must coexist without replacing one another or rewriting the source text.

## ADR 51 — Core and Qwen use a JSON subprocess boundary

**Decision:** Core writes a model-neutral batch job, invokes `.venv-tts\\Scripts\\python.exe`, and validates a result document. Only `runtime/qwen3_tts_worker.py` imports `torch` and `qwen_tts` for synthesis.

**Rationale:** Qwen's CUDA/Transformers dependency set must not mutate the established Core environment. A subprocess keeps the boundary explicit while avoiding an unnecessary HTTP service.

## ADR 52 — Normalized text is the synthesis basis

**Decision:** Begin with `normalized_text` and replace only surfaces confirmed by Stage 6 `applied_overrides` provenance. Retain full `reading_kana` as expected-reading evidence rather than sending it as the default sentence.

**Rationale:** A generated frontend reading is not equivalent to an approved pronunciation rewrite. Local confirmed substitutions preserve Japanese orthography everywhere else.

## ADR 53 — Override occurrence counts are integrity checks

**Decision:** Require every confirmed override surface to occur exactly as many times as Stage 6 recorded; otherwise create a sample-level planning failure.

**Rationale:** Guessing around stale or contradictory provenance could send unintended text to a model and conceal an upstream data-integrity defect.

## ADR 54 — Stage 8 pins effective Qwen generation parameters

**Decision:** Explicitly record and pass the locally inspected sampling, top-k/top-p, temperature, repetition-penalty, sub-talker, and maximum-token values. Use Japanese, Ono_Anna, BF16, CUDA device 0, SDPA, no instruction, and non-streaming mode.

**Rationale:** The package merges checkpoint values and hard defaults. Saying only “default” would not preserve the configuration actually used.

## ADR 55 — Seed and run identity provide reproducible provenance

**Decision:** Derive a per-sample seed from a recorded base seed and SHA-256 of `sample_id`. Hash the sample, backend, model, speaker, config fingerprint, seed, synthesis strategy, and actual synthesis text into `run_id`.

**Rationale:** Stable provenance enables idempotent retry and one-to-many runs. It does not promise byte-identical CUDA output across runtime or library upgrades.

## ADR 56 — Successful audio requires physical validation

**Decision:** Decode every generated WAV, require positive stream metadata, finite and non-silent samples, and record its SHA-256. Export only validated successes to `generated_audio_manifest.jsonl`.

**Rationale:** A successful model call is insufficient evidence that a usable artifact was written. Later QA needs an explicit trusted handoff boundary.

## ADR 57 — Model loads once and failures have two scopes

**Decision:** Load Qwen once per worker batch. Record ordinary request failures and continue; fail the batch on worker initialization, CUDA loss, or an unusable runtime.

**Rationale:** Reloading a 1.7B model per sample is wasteful, while continuing after infrastructure corruption would produce misleading partial evidence.

## ADR 58 — Stage 8 generates but does not evaluate

**Decision:** Record runtime measurements and artifact validity only. Do not compute ASR round trips, pronunciation scores, speaker similarity, MOS, pitch accent, or release decisions.

**Rationale:** Those policies belong to the separate Quality & Release Pipeline and require evaluation evidence that synthesis cannot provide itself.
