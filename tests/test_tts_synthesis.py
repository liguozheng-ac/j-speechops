from __future__ import annotations

import math
import struct
import wave
from pathlib import Path

import pytest

from j_speech_ops.japanese_reading import (
    AppliedReadingOverride,
    ReadingProviderInfo,
    ReadingResult,
)
from j_speech_ops.schemas import CurationDecision, RightsStatus
from j_speech_ops.text_manifest import write_text_manifest
from j_speech_ops.text_preparation import (
    TextPreparationDependencies,
    TextPreparationReport,
    TextPreparationStatus,
    write_text_preparation_reports,
)
from j_speech_ops.tts_synthesis import (
    AUDIO_POST_PROCESSING_POLICY_VERSION,
    AudioPostProcessingPolicy,
    GeneratedAudioArtifact,
    Qwen3TTSRuntimeAdapter,
    RuntimeBatchJob,
    RuntimeBatchResult,
    RuntimeResultStatus,
    RuntimeSampleResult,
    SynthesisConfig,
    SynthesisPlanningError,
    SynthesisRunStatus,
    TTSSynthesisPipeline,
    TTSRuntimeInfrastructureError,
    build_run_id,
    build_synthesis_text,
    config_fingerprint,
    derive_seed,
    iter_synthesis_runs,
    validate_generated_wav,
)
from j_speech_ops.tts_text import (
    TTSTextContent,
    TTSTextSample,
    TTSTextSource,
    TTSTextStage,
)
from j_speech_ops.tts_text_curation import (
    TTSTextCurationReport,
    TTSTextCurationStatus,
    write_tts_text_curation_reports,
)


PROVIDER = ReadingProviderInfo(
    provider="test-provider",
    provider_version="1",
    package_name="test",
    package_version="1",
)


def make_sample(
    sample_id: str,
    *,
    normalized: str = "料金は3500円です。",
    reading: str = "リョーキンワサンゼンゴヒャクエンデス。",
    decision: CurationDecision = CurationDecision.PASS,
) -> TTSTextSample:
    return TTSTextSample(
        sample_id=sample_id,
        dataset_id="stage8-test",
        language="ja",
        locale="ja-JP",
        text=TTSTextContent(
            raw_text=normalized,
            normalized_text=normalized,
            reading_kana=reading,
        ),
        source=TTSTextSource(source_name="test", source_id=sample_id),
        rights_status=RightsStatus.CLEARED,
        intended_use="tts_production",
        state={
            "stage": TTSTextStage.CURATED.value,
            "curation_decision": decision.value,
            "decision_reasons": (),
        },
    )


def make_reports(
    sample: TTSTextSample,
    *,
    overrides: tuple[AppliedReadingOverride, ...] = (),
) -> tuple[TextPreparationReport, TTSTextCurationReport]:
    reading = ReadingResult(
        sample_id=sample.sample_id,
        provider=PROVIDER.provider,
        provider_version=PROVIDER.provider_version,
        input_text=sample.text.normalized_text or "",
        baseline_reading_kana=sample.text.reading_kana or "",
        reading_kana=sample.text.reading_kana or "",
        override_count=sum(item.occurrences for item in overrides),
        applied_overrides=overrides,
        override_fingerprint_sha256="a" * 64,
    )
    preparation = TextPreparationReport(
        sample_id=sample.sample_id,
        policy_version="test-preparation-v1",
        status=TextPreparationStatus.READING_PREPARED,
        input_stage=TTSTextStage.RAW,
        output_stage=TTSTextStage.READING_PREPARED,
        reading=reading,
        dependencies=TextPreparationDependencies(
            jaconv_version="0.5.0", reading_provider=PROVIDER
        ),
    )
    curation = TTSTextCurationReport(
        sample_id=sample.sample_id,
        policy_version="test-curation-v1",
        policy_fingerprint_sha256="b" * 64,
        status=TTSTextCurationStatus.CURATED,
        input_stage=TTSTextStage.READING_PREPARED,
        output_stage=TTSTextStage.CURATED,
        decision=CurationDecision.PASS,
        raw_text=sample.text.raw_text,
        normalized_text=sample.text.normalized_text,
        reading_kana=sample.text.reading_kana,
        preparation_policy_version=preparation.policy_version,
        preparation_status=preparation.status,
        reading_provider=PROVIDER,
        applied_overrides=overrides,
        override_fingerprint_sha256="a" * 64,
        watchlist_version="test-watchlist-v1",
        watchlist_source="tests",
        watchlist_fingerprint_sha256="c" * 64,
    )
    return preparation, curation


def write_wav(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    frames = [
        int(4000 * math.sin(2 * math.pi * 440 * index / 24000))
        for index in range(2400)
    ]
    with wave.open(str(path), "wb") as handle:
        handle.setnchannels(1)
        handle.setsampwidth(2)
        handle.setframerate(24000)
        handle.writeframes(struct.pack(f"<{len(frames)}h", *frames))


class FakeRuntime:
    def __init__(self, failed_samples: set[str] | None = None) -> None:
        self.failed_samples = failed_samples or set()
        self.calls = 0
        self.jobs: list[RuntimeBatchJob] = []

    def synthesize(self, job: RuntimeBatchJob, *, output_root: Path) -> RuntimeBatchResult:
        self.calls += 1
        self.jobs.append(job)
        results = []
        for request in job.requests:
            if request.sample_id in self.failed_samples:
                results.append(
                    RuntimeSampleResult(
                        run_id=request.run_id,
                        status=RuntimeResultStatus.FAILED,
                        error_code="synthetic_failure",
                        error_message="intentional test failure",
                    )
                )
                continue
            write_wav(Path(request.output_audio_path))
            results.append(
                RuntimeSampleResult(
                    run_id=request.run_id,
                    status=RuntimeResultStatus.SUCCESS,
                    generation_time_sec=0.2,
                )
            )
        return RuntimeBatchResult(
            backend_version="test-backend-1",
            model_load_time_sec=0.1,
            gpu_name="synthetic GPU",
            gpu_total_vram_bytes=16 * 1024**3,
            peak_allocated_vram_bytes=4 * 1024**3,
            peak_reserved_vram_bytes=5 * 1024**3,
            supported_languages=("japanese",),
            supported_speakers=("ono_anna",),
            results=tuple(results),
        )


class BrokenRuntime:
    def synthesize(self, job: RuntimeBatchJob, *, output_root: Path) -> RuntimeBatchResult:
        raise TTSRuntimeInfrastructureError("model initialization failed")


def run_pipeline(
    tmp_path: Path,
    samples: list[TTSTextSample],
    runtime,
    *,
    overrides: dict[str, tuple[AppliedReadingOverride, ...]] | None = None,
    config: SynthesisConfig | None = None,
    force: bool = False,
):
    manifest = tmp_path / "input.jsonl"
    prep_path = tmp_path / "preparation.jsonl"
    curation_path = tmp_path / "curation.jsonl"
    prepared = []
    curated = []
    for sample in samples:
        prep, curation = make_reports(
            sample, overrides=(overrides or {}).get(sample.sample_id, ())
        )
        prepared.append(prep)
        curated.append(curation)
    write_text_manifest(manifest, samples)
    write_text_preparation_reports(prep_path, prepared)
    write_tts_text_curation_reports(curation_path, curated)
    pipeline = TTSSynthesisPipeline(
        output_root=tmp_path / "output",
        config=config or SynthesisConfig(model_path="D:/models/qwen"),
        runtime_adapter=runtime,
    )
    return pipeline.run(manifest, prep_path, curation_path, force=force)


def test_synthesis_text_without_override_preserves_normalized_orthography() -> None:
    plan = build_synthesis_text("料金は3500円です。", "リョーキン...", ())
    assert plan.synthesis_text == "料金は3500円です。"
    assert plan.synthesis_text != plan.expected_reading_kana


def test_confirmed_override_rewrites_only_surface_and_all_occurrences() -> None:
    override = AppliedReadingOverride(
        surface="佐伯", reading_kana="サイキ", source="reviewed", occurrences=2
    )
    plan = build_synthesis_text(
        "佐伯様から佐伯様へ。", "サイキサマカラサイキサマエ。", (override,)
    )
    assert plan.synthesis_text == "サイキ様からサイキ様へ。"
    assert plan.applied_pronunciation_overrides == (override,)


def test_override_occurrence_mismatch_is_planning_failure() -> None:
    override = AppliedReadingOverride(
        surface="佐伯", reading_kana="サイキ", source="reviewed", occurrences=2
    )
    with pytest.raises(SynthesisPlanningError, match="occurrence mismatch"):
        build_synthesis_text("佐伯様です。", "サイキサマデス。", (override,))


def test_seed_and_run_identity_capture_seed_speaker_and_config() -> None:
    first = SynthesisConfig(model_path="D:/models/qwen", base_seed=10)
    other_seed = SynthesisConfig(model_path="D:/models/qwen", base_seed=11)
    other_speaker = first.model_copy(update={"speaker": "Future_Speaker"})
    seed_a = derive_seed("sample", first.base_seed)
    seed_b = derive_seed("sample", other_seed.base_seed)
    assert seed_a != seed_b
    run_a = build_run_id(
        sample_id="sample", config=first, seed=seed_a, synthesis_text="本文"
    )
    run_b = build_run_id(
        sample_id="sample", config=other_seed, seed=seed_b, synthesis_text="本文"
    )
    run_c = build_run_id(
        sample_id="sample", config=other_speaker, seed=seed_a, synthesis_text="本文"
    )
    assert len({run_a, run_b, run_c}) == 3
    assert config_fingerprint(first) != config_fingerprint(other_speaker)


def test_post_roll_policy_defaults_and_changes_run_identity() -> None:
    default = SynthesisConfig(model_path="D:/models/qwen")
    disabled = default.model_copy(
        update={"audio_post_processing": AudioPostProcessingPolicy(duration_ms=0)}
    )
    extended = default.model_copy(
        update={"audio_post_processing": AudioPostProcessingPolicy(duration_ms=500)}
    )
    assert default.audio_post_processing.duration_ms == 300
    assert default.audio_post_processing.policy_version == (
        AUDIO_POST_PROCESSING_POLICY_VERSION
    )
    seed = derive_seed("sample", default.base_seed)
    identities = {
        build_run_id(
            sample_id="sample", config=config, seed=seed, synthesis_text="本文"
        )
        for config in (disabled, default, extended)
    }
    assert len(identities) == 3


def test_runtime_request_round_trip() -> None:
    payload = RuntimeBatchJob(
        backend="fake",
        model_name="fake/model",
        model_path="D:/fake",
        speaker="speaker",
        language="Japanese",
        device="cuda:0",
        dtype="bfloat16",
        attention_implementation="sdpa",
        generation_parameters={},
        requests=(),
    )
    assert RuntimeBatchJob.model_validate_json(payload.model_dump_json()) == payload


def test_one_text_produces_valid_run_artifact_and_manifests(tmp_path: Path) -> None:
    sample = make_sample("sample-1")
    runtime = FakeRuntime()
    outputs = run_pipeline(tmp_path, [sample], runtime)
    runs = list(iter_synthesis_runs(outputs.synthesis_runs_manifest))
    assert runtime.calls == 1
    assert len(runtime.jobs[0].requests) == 1
    assert len(runs) == 1
    assert runs[0].status is SynthesisRunStatus.SUCCESS
    assert runs[0].synthesis_text == sample.text.normalized_text
    assert runs[0].expected_reading_kana == sample.text.reading_kana
    assert runs[0].audio_sha256
    assert runs[0].audio_post_processing is not None
    assert runs[0].audio_post_processing.type == "tail_padding"
    assert runs[0].audio_post_processing.duration_ms == 300
    assert len(runs[0].audio_post_processing.policy_fingerprint_sha256) == 64
    artifact = GeneratedAudioArtifact.model_validate_json(
        outputs.generated_audio_manifest.read_text(encoding="utf-8")
    )
    assert artifact.audio_post_processing == runs[0].audio_post_processing
    assert outputs.generated_audio_manifest.read_text(encoding="utf-8").count("\n") == 1
    assert outputs.summary.succeeded_runs == 1


def test_existing_valid_run_skips_and_force_regenerates(tmp_path: Path) -> None:
    sample = make_sample("sample-1")
    runtime = FakeRuntime()
    run_pipeline(tmp_path, [sample], runtime)
    skipped = run_pipeline(tmp_path, [sample], runtime)
    assert runtime.calls == 1
    assert skipped.summary.skipped_runs == 1
    assert skipped.summary.model_load_time_sec == 0.1
    assert skipped.summary.gpu_name == "synthetic GPU"
    forced = run_pipeline(tmp_path, [sample], runtime, force=True)
    assert runtime.calls == 2
    assert forced.summary.attempted_runs == 1


def test_same_text_with_different_seed_preserves_two_runs(tmp_path: Path) -> None:
    sample = make_sample("sample-1")
    runtime = FakeRuntime()
    first = SynthesisConfig(model_path="D:/models/qwen", base_seed=10)
    second = SynthesisConfig(model_path="D:/models/qwen", base_seed=11)
    run_pipeline(tmp_path, [sample], runtime, config=first)
    outputs = run_pipeline(tmp_path, [sample], runtime, config=second)
    runs = list(iter_synthesis_runs(outputs.synthesis_runs_manifest))
    assert len(runs) == 2
    assert len({run.seed for run in runs}) == 2
    assert len({run.run_id for run in runs}) == 2


def test_sample_failure_isolated_and_later_sample_continues(tmp_path: Path) -> None:
    failed = make_sample("failed")
    succeeded = make_sample("succeeded")
    outputs = run_pipeline(
        tmp_path, [failed, succeeded], FakeRuntime({"failed"})
    )
    runs = {run.sample_id: run for run in iter_synthesis_runs(outputs.synthesis_runs_manifest)}
    assert runs["failed"].status is SynthesisRunStatus.FAILED
    assert runs["succeeded"].status is SynthesisRunStatus.SUCCESS
    assert outputs.summary.failed_runs == 1
    assert outputs.summary.succeeded_runs == 1


def test_planning_failure_isolated_and_override_reaches_runtime(tmp_path: Path) -> None:
    bad = make_sample("bad", normalized="佐伯様です。", reading="サイキサマデス。")
    good = make_sample("good", normalized="JR新宿駅です。", reading="ジェイアールシンジュクエキデス。")
    overrides = {
        "bad": (
            AppliedReadingOverride(
                surface="佐伯", reading_kana="サイキ", source="reviewed", occurrences=2
            ),
        ),
        "good": (
            AppliedReadingOverride(
                surface="JR", reading_kana="ジェイアール", source="reviewed", occurrences=1
            ),
        ),
    }
    runtime = FakeRuntime()
    outputs = run_pipeline(tmp_path, [bad, good], runtime, overrides=overrides)
    assert outputs.summary.planning_failed_runs == 1
    assert len(runtime.jobs[0].requests) == 1
    assert runtime.jobs[0].requests[0].synthesis_text == "ジェイアール新宿駅です。"


def test_non_pass_input_is_rejected_before_runtime(tmp_path: Path) -> None:
    runtime = FakeRuntime()
    sample = make_sample("review", decision=CurationDecision.REVIEW)
    with pytest.raises(ValueError, match=r"CURATED \+ PASS"):
        run_pipeline(tmp_path, [sample], runtime)
    assert runtime.calls == 0


def test_worker_initialization_failure_fails_fast(tmp_path: Path) -> None:
    with pytest.raises(TTSRuntimeInfrastructureError, match="initialization"):
        run_pipeline(tmp_path, [make_sample("sample")], BrokenRuntime())


def test_wav_validation_rejects_silence(tmp_path: Path) -> None:
    path = tmp_path / "silent.wav"
    with wave.open(str(path), "wb") as handle:
        handle.setnchannels(1)
        handle.setsampwidth(2)
        handle.setframerate(24000)
        handle.writeframes(b"\x00\x00" * 100)
    with pytest.raises(ValueError, match="silent"):
        validate_generated_wav(path)


def test_missing_tts_python_fails_without_core_fallback(tmp_path: Path) -> None:
    adapter = Qwen3TTSRuntimeAdapter(
        python_executable=tmp_path / "missing-python.exe",
        worker_script=tmp_path / "worker.py",
    )
    job = RuntimeBatchJob(
        backend="qwen3-tts",
        model_name="model",
        model_path="D:/model",
        speaker="Ono_Anna",
        language="Japanese",
        device="cuda:0",
        dtype="bfloat16",
        attention_implementation="sdpa",
        generation_parameters={},
        requests=(),
    )
    with pytest.raises(TTSRuntimeInfrastructureError, match="TTS Python"):
        adapter.synthesize(job, output_root=tmp_path)
