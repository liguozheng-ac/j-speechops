from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import pytest
import soundfile as sf

from j_speech_ops.asr import (
    ASRConfig,
    ASRInfrastructureError,
    ASRResult,
    ASRRuntimeInfo,
    ASRSegment,
    LazyFasterWhisperAdapter,
)
from j_speech_ops.generated_audio_qa import (
    GeneratedAudioQAPipeline,
    GeneratedAudioQAPolicy,
    QADecision,
    QAExecutionStatus,
    canonicalize_japanese_reading,
    iter_qa_results,
    levenshtein_distance,
)
from j_speech_ops.japanese_normalization import JapaneseNormalizer
from j_speech_ops.japanese_reading import (
    ReadingProviderInfo,
    ReadingResult,
)
from j_speech_ops.qa_generated_audio import build_parser
from j_speech_ops.tts_synthesis import (
    GeneratedAudioArtifact,
    write_generated_audio_manifest,
)


class FakeASR:
    def __init__(self, responses: dict[str, str | Exception] | None = None) -> None:
        self.responses = responses or {}
        self.calls: list[Path] = []

    @property
    def runtime_info(self) -> ASRRuntimeInfo:
        return ASRRuntimeInfo(
            backend_name="fake-whisper",
            backend_version="1.0",
            model_name="large-v3",
            device="cuda",
            compute_type="float16",
            language="ja",
            effective_config={
                "beam_size": 5,
                "temperature": 0.0,
                "vad_filter": False,
                "local_files_only": True,
                "model_path": "D:/machine-local/model",
            },
        )

    def transcribe(self, audio_path: Path) -> ASRResult:
        self.calls.append(audio_path)
        response = self.responses.get(audio_path.name, "コンニチワ。")
        if isinstance(response, Exception):
            raise response
        segments = (
            [ASRSegment(start_sec=0.0, end_sec=0.1, text=response)]
            if response.strip()
            else []
        )
        return ASRResult(
            text=response,
            language="ja",
            language_probability=0.99,
            duration_sec=0.1,
            segments=segments,
        )


class FakeReadingProvider:
    def __init__(self, readings: dict[str, str] | None = None) -> None:
        self.readings = readings or {}
        self.calls: list[str] = []
        self._info = ReadingProviderInfo(
            provider="fake-openjtalk",
            provider_version="1",
            package_name="fake",
            package_version="1",
        )

    @property
    def info(self) -> ReadingProviderInfo:
        return self._info

    def prepare(self, sample_id, text, overrides) -> ReadingResult:
        self.calls.append(text)
        reading = self.readings.get(text, text)
        return ReadingResult(
            sample_id=sample_id,
            provider=self.info.provider,
            provider_version=self.info.provider_version,
            input_text=text,
            baseline_reading_kana=reading,
            reading_kana=reading,
            override_count=0,
            override_fingerprint_sha256=overrides.fingerprint_sha256,
        )


def write_wav(
    path: Path,
    *,
    sample_rate: int = 24_000,
    channels: int = 1,
    silent: bool = False,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    frames = 2_400
    signal = np.zeros(frames, dtype=np.float32)
    if not silent:
        positions = np.arange(frames, dtype=np.float32)
        signal = 0.2 * np.sin(2 * np.pi * 440 * positions / sample_rate)
    audio = signal if channels == 1 else np.column_stack([signal] * channels)
    sf.write(path, audio, sample_rate, format="WAV", subtype="PCM_16")


def make_artifact(
    sample_id: str,
    path: Path,
    *,
    expected: str = "コンニチワ。",
    audio_sha256: str | None = None,
    sample_rate_hz: int = 24_000,
    channels: int = 1,
    duration_sec: float = 0.1,
) -> GeneratedAudioArtifact:
    digest = audio_sha256
    if digest is None:
        digest = hashlib.sha256(path.read_bytes()).hexdigest()
    return GeneratedAudioArtifact(
        run_id=hashlib.sha256(f"run:{sample_id}".encode()).hexdigest(),
        sample_id=sample_id,
        input_text="こんにちは。",
        expected_reading_kana=expected,
        synthesis_text="こんにちは。",
        audio_path=str(path),
        backend="qwen3-tts",
        backend_version="0.1.1",
        model_name="Qwen/Qwen3-TTS-12Hz-1.7B-CustomVoice",
        speaker="Ono_Anna",
        language="Japanese",
        sample_rate_hz=sample_rate_hz,
        channels=channels,
        duration_sec=duration_sec,
        audio_sha256=digest,
    )


def run_pipeline(
    tmp_path: Path,
    artifacts: list[GeneratedAudioArtifact],
    *,
    responses: dict[str, str | Exception] | None = None,
    readings: dict[str, str] | None = None,
    force: bool = False,
    output_name: str = "qa",
):
    manifest = tmp_path / "generated_audio_manifest.jsonl"
    write_generated_audio_manifest(manifest, artifacts)
    adapter = FakeASR(responses)
    provider = FakeReadingProvider(readings)
    pipeline = GeneratedAudioQAPipeline(
        output_root=tmp_path / output_name,
        adapter=adapter,
        normalizer=JapaneseNormalizer(),
        reading_provider=provider,
        asr_model_load_time_sec=1.25,
        gpu_name="fake-gpu",
    )
    outputs = pipeline.run(manifest, force=force)
    return outputs, adapter, provider, pipeline, manifest


def one_result(outputs) -> object:
    return list(iter_qa_results(outputs.results_manifest))[0]


def test_missing_audio_is_objective_drop(tmp_path: Path) -> None:
    artifact = make_artifact(
        "missing", tmp_path / "missing.wav", audio_sha256="a" * 64
    )
    outputs, adapter, _, _, _ = run_pipeline(tmp_path, [artifact])
    result = one_result(outputs)
    assert result.status is QAExecutionStatus.COMPLETED
    assert result.decision is QADecision.DROP
    assert result.issues[0].code == "audio_missing"
    assert adapter.calls == []


def test_hash_mismatch_is_drop(tmp_path: Path) -> None:
    audio = tmp_path / "changed.wav"
    write_wav(audio)
    artifact = make_artifact("hash", audio, audio_sha256="0" * 64)
    outputs, adapter, _, _, _ = run_pipeline(tmp_path, [artifact])
    result = one_result(outputs)
    assert result.decision is QADecision.DROP
    assert "audio_hash_mismatch" in result.decision_reasons
    assert result.observed_audio_sha256 != result.expected_audio_sha256
    assert adapter.calls == []


def test_empty_or_silent_audio_is_drop(tmp_path: Path) -> None:
    empty = tmp_path / "empty.wav"
    empty.write_bytes(b"")
    silent = tmp_path / "silent.wav"
    write_wav(silent, silent=True)
    artifacts = [
        make_artifact("empty", empty, audio_sha256="a" * 64),
        make_artifact("silent", silent),
    ]
    outputs, _, _, _, _ = run_pipeline(tmp_path, artifacts)
    results = {item.sample_id: item for item in iter_qa_results(outputs.results_manifest)}
    assert results["empty"].decision is QADecision.DROP
    assert results["empty"].issues[0].code == "audio_empty"
    assert results["silent"].decision is QADecision.DROP
    assert results["silent"].issues[0].code == "audio_silent"


def test_valid_non_silent_wav_continues_and_records_hash(tmp_path: Path) -> None:
    audio = tmp_path / "valid.wav"
    write_wav(audio)
    artifact = make_artifact("valid", audio)
    outputs, adapter, _, _, _ = run_pipeline(tmp_path, [artifact])
    result = one_result(outputs)
    assert result.decision is QADecision.PASS
    assert adapter.calls == [audio]
    assert result.observed_audio_sha256 == artifact.audio_sha256
    assert result.audio_integrity.observed_sample_rate_hz == 24_000
    assert result.audio_integrity.observed_channels == 1


@pytest.mark.parametrize(
    ("field", "value", "issue"),
    [
        ("sample_rate_hz", 16_000, "audio_sample_rate_mismatch"),
        ("channels", 2, "audio_channel_count_mismatch"),
        ("duration_sec", 1.0, "audio_duration_mismatch"),
    ],
)
def test_manifest_metadata_mismatch_is_drop(
    tmp_path: Path, field: str, value: int | float, issue: str
) -> None:
    audio = tmp_path / f"{field}.wav"
    write_wav(audio)
    artifact = make_artifact("metadata", audio).model_copy(update={field: value})
    outputs, _, _, _, _ = run_pipeline(tmp_path, [artifact])
    result = one_result(outputs)
    assert result.decision is QADecision.DROP
    assert issue in result.decision_reasons


@pytest.mark.parametrize(
    ("expected", "observed"),
    [
        ("コンニチワ。", "コンニチワ。"),
        ("コンニチワ。", "コンニチワ、"),
        ("コンニチハ", "こんにちは"),
    ],
)
def test_presentation_only_reading_differences_pass(
    tmp_path: Path, expected: str, observed: str
) -> None:
    audio = tmp_path / "presentation.wav"
    write_wav(audio)
    artifact = make_artifact("presentation", audio, expected=expected)
    outputs, _, _, _, _ = run_pipeline(
        tmp_path,
        [artifact],
        responses={audio.name: observed},
    )
    result = one_result(outputs)
    assert result.decision is QADecision.PASS
    assert result.edit_distance == 0
    assert result.cer == 0.0


def test_true_one_character_difference_is_review(tmp_path: Path) -> None:
    audio = tmp_path / "one-character.wav"
    write_wav(audio)
    artifact = make_artifact("one-char", audio, expected="コンニチワ")
    outputs, _, _, _, _ = run_pipeline(
        tmp_path, [artifact], responses={audio.name: "コンニチハ"}
    )
    result = one_result(outputs)
    assert result.decision is QADecision.REVIEW
    assert result.edit_distance == 1
    assert result.cer == pytest.approx(1 / 5)
    assert "content_reading_mismatch" in result.decision_reasons


def test_empty_asr_transcript_is_review_not_drop(tmp_path: Path) -> None:
    audio = tmp_path / "empty-asr.wav"
    write_wav(audio)
    outputs, _, _, _, _ = run_pipeline(
        tmp_path,
        [make_artifact("empty-asr", audio)],
        responses={audio.name: "  "},
    )
    result = one_result(outputs)
    assert result.status is QAExecutionStatus.COMPLETED
    assert result.decision is QADecision.REVIEW
    assert "asr_empty_transcript" in result.decision_reasons
    assert "content_reading_mismatch" in result.decision_reasons


def test_large_mismatch_is_review_not_drop(tmp_path: Path) -> None:
    audio = tmp_path / "large-mismatch.wav"
    write_wav(audio)
    outputs, _, _, _, _ = run_pipeline(
        tmp_path,
        [make_artifact("large", audio, expected="ゼンゼンチガウブン")],
        responses={audio.name: "コンニチワ"},
    )
    result = one_result(outputs)
    assert result.decision is QADecision.REVIEW
    assert result.cer is not None and result.cer > 0.5
    assert all(issue.severity.value != "critical" for issue in result.issues)


def test_kanji_asr_text_is_converted_to_reading_before_comparison(
    tmp_path: Path,
) -> None:
    audio = tmp_path / "kanji.wav"
    write_wav(audio)
    transcript = "新宿駅までご案内いたします。"
    reading = "シンジュクエキマデゴアンナイイタシマス。"
    outputs, _, provider, _, _ = run_pipeline(
        tmp_path,
        [make_artifact("kanji", audio, expected=reading)],
        responses={audio.name: transcript},
        readings={transcript: reading},
    )
    result = one_result(outputs)
    assert provider.calls == [transcript]
    assert result.asr_transcript == transcript
    assert result.observed_reading_kana == reading
    assert result.decision is QADecision.PASS


def test_one_sample_failure_does_not_stop_later_samples(tmp_path: Path) -> None:
    first = tmp_path / "first.wav"
    second = tmp_path / "second.wav"
    write_wav(first)
    write_wav(second)
    outputs, adapter, _, _, _ = run_pipeline(
        tmp_path,
        [make_artifact("first", first), make_artifact("second", second)],
        responses={first.name: ValueError("bad packet"), second.name: "コンニチワ。"},
    )
    results = {item.sample_id: item for item in iter_qa_results(outputs.results_manifest)}
    assert results["first"].status is QAExecutionStatus.FAILED
    assert results["first"].decision is QADecision.UNDECIDED
    assert results["second"].decision is QADecision.PASS
    assert len(adapter.calls) == 2


def test_asr_model_initialization_failure_fails_batch(tmp_path: Path) -> None:
    audio = tmp_path / "infrastructure.wav"
    write_wav(audio)
    manifest = tmp_path / "generated_audio_manifest.jsonl"
    write_generated_audio_manifest(manifest, [make_artifact("infra", audio)])
    def broken_factory(config: ASRConfig) -> FakeASR:
        raise ASRInfrastructureError("model initialization failed")

    pipeline = GeneratedAudioQAPipeline(
        output_root=tmp_path / "qa",
        adapter=LazyFasterWhisperAdapter(
            ASRConfig(), adapter_factory=broken_factory
        ),
        normalizer=JapaneseNormalizer(),
        reading_provider=FakeReadingProvider(),
    )
    with pytest.raises(ASRInfrastructureError, match="model initialization failed"):
        pipeline.run(manifest)


def test_same_qa_id_skips_and_force_reruns(tmp_path: Path) -> None:
    audio = tmp_path / "idempotent.wav"
    write_wav(audio)
    artifact = make_artifact("idempotent", audio)
    manifest = tmp_path / "generated_audio_manifest.jsonl"
    write_generated_audio_manifest(manifest, [artifact])
    adapter = FakeASR()
    pipeline = GeneratedAudioQAPipeline(
        output_root=tmp_path / "qa",
        adapter=adapter,
        normalizer=JapaneseNormalizer(),
        reading_provider=FakeReadingProvider(),
    )
    first = pipeline.run(manifest)
    second = pipeline.run(manifest)
    forced = pipeline.run(manifest, force=True)
    assert first.summary.skipped_samples == 0
    assert second.summary.skipped_samples == 1
    assert forced.summary.skipped_samples == 0
    assert len(adapter.calls) == 2


def test_idempotent_rerun_does_not_initialize_whisper(tmp_path: Path) -> None:
    audio = tmp_path / "lazy.wav"
    write_wav(audio)
    manifest = tmp_path / "generated_audio_manifest.jsonl"
    write_generated_audio_manifest(manifest, [make_artifact("lazy", audio)])
    factory_calls: list[FakeASR] = []

    def factory(config: ASRConfig) -> FakeASR:
        adapter = FakeASR()
        factory_calls.append(adapter)
        return adapter

    first_adapter = LazyFasterWhisperAdapter(ASRConfig(), adapter_factory=factory)
    GeneratedAudioQAPipeline(
        output_root=tmp_path / "qa",
        adapter=first_adapter,
        normalizer=JapaneseNormalizer(),
        reading_provider=FakeReadingProvider(),
    ).run(manifest)
    assert len(factory_calls) == 1

    second_adapter = LazyFasterWhisperAdapter(ASRConfig(), adapter_factory=factory)
    second = GeneratedAudioQAPipeline(
        output_root=tmp_path / "qa",
        adapter=second_adapter,
        normalizer=JapaneseNormalizer(),
        reading_provider=FakeReadingProvider(),
    ).run(manifest)
    assert second.summary.skipped_samples == 1
    assert second.summary.asr_model_load_time_sec == 0.0
    assert len(factory_calls) == 1


def test_downstream_manifests_are_strictly_routed(tmp_path: Path) -> None:
    pass_audio = tmp_path / "pass.wav"
    review_audio = tmp_path / "review.wav"
    write_wav(pass_audio)
    write_wav(review_audio)
    missing_audio = tmp_path / "drop.wav"
    artifacts = [
        make_artifact("pass", pass_audio),
        make_artifact("review", review_audio),
        make_artifact("drop", missing_audio, audio_sha256="b" * 64),
    ]
    outputs, _, _, _, _ = run_pipeline(
        tmp_path,
        artifacts,
        responses={review_audio.name: "サヨナラ"},
    )
    pass_rows = [json.loads(line) for line in outputs.pass_manifest.read_text(encoding="utf-8").splitlines()]
    review_rows = [json.loads(line) for line in outputs.review_queue.read_text(encoding="utf-8").splitlines()]
    assert [row["sample_id"] for row in pass_rows] == ["pass"]
    assert [row["sample_id"] for row in review_rows] == ["review"]
    assert all(row["sample_id"] != "drop" for row in pass_rows + review_rows)
    assert outputs.summary.passed_samples == 1
    assert outputs.summary.review_samples == 1
    assert outputs.summary.dropped_samples == 1


def test_missing_expected_reading_is_failed_undecided_and_continues(
    tmp_path: Path,
) -> None:
    missing_audio = tmp_path / "missing-reading.wav"
    valid_audio = tmp_path / "valid-reading.wav"
    write_wav(missing_audio)
    write_wav(valid_audio)
    missing = make_artifact("missing-reading", missing_audio).model_dump(mode="json")
    missing.pop("expected_reading_kana")
    valid = make_artifact("valid-reading", valid_audio).model_dump(mode="json")
    manifest = tmp_path / "generated_audio_manifest.jsonl"
    manifest.write_text(
        json.dumps(missing, ensure_ascii=False)
        + "\n"
        + json.dumps(valid, ensure_ascii=False)
        + "\n",
        encoding="utf-8",
    )
    pipeline = GeneratedAudioQAPipeline(
        output_root=tmp_path / "qa",
        adapter=FakeASR(),
        normalizer=JapaneseNormalizer(),
        reading_provider=FakeReadingProvider(),
    )
    outputs = pipeline.run(manifest)
    results = {item.sample_id: item for item in iter_qa_results(outputs.results_manifest)}
    assert results["missing-reading"].status is QAExecutionStatus.FAILED
    assert results["missing-reading"].decision is QADecision.UNDECIDED
    assert results["missing-reading"].expected_reading_kana is None
    assert results["valid-reading"].decision is QADecision.PASS


def test_canonicalizer_preserves_phonological_distinctions() -> None:
    assert canonicalize_japanese_reading("しんじゅく、えき。") == "シンジュクエキ"
    assert canonicalize_japanese_reading("カー") != canonicalize_japanese_reading("カ")
    assert canonicalize_japanese_reading("カッタ") != canonicalize_japanese_reading("カタ")
    assert canonicalize_japanese_reading("カン") != canonicalize_japanese_reading("カ")
    assert canonicalize_japanese_reading("ガ") != canonicalize_japanese_reading("カ")


def test_levenshtein_handles_empty_and_is_deterministic() -> None:
    assert levenshtein_distance("", "") == 0
    assert levenshtein_distance("カナ", "") == 2
    assert levenshtein_distance("カナ", "カニ") == 1
    assert levenshtein_distance("キョー", "キヨ") == 2


def test_policy_change_produces_distinct_qa_identity(tmp_path: Path) -> None:
    audio = tmp_path / "identity.wav"
    write_wav(audio)
    artifact = make_artifact("identity", audio)
    first, _, _, first_pipeline, _ = run_pipeline(
        tmp_path, [artifact], output_name="first"
    )
    manifest = tmp_path / "generated_audio_manifest.jsonl"
    second_pipeline = GeneratedAudioQAPipeline(
        output_root=tmp_path / "second",
        adapter=FakeASR(),
        normalizer=JapaneseNormalizer(),
        reading_provider=FakeReadingProvider(),
        policy=GeneratedAudioQAPolicy(duration_tolerance_sec=0.002),
    )
    second = second_pipeline.run(manifest)
    assert first_pipeline.policy_fingerprint != second_pipeline.policy_fingerprint
    assert one_result(first).qa_id != one_result(second).qa_id


def test_cli_requires_explicit_model_and_has_no_tts_runtime_argument() -> None:
    parser = build_parser()
    args = parser.parse_args(
        [
            "--manifest",
            "generated_audio_manifest.jsonl",
            "--model-path",
            "X:/external-models/Whisper/faster-whisper-large-v3",
            "--force",
        ]
    )
    assert args.model_path == Path("X:/external-models/Whisper/faster-whisper-large-v3")
    assert args.force is True
    assert "tts" not in vars(args)
