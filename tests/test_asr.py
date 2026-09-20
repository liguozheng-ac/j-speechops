from pathlib import Path
from types import SimpleNamespace

import pytest
from pydantic import ValidationError

import j_speech_ops.asr as asr_module
from j_speech_ops.asr import (
    ASRConfig,
    ASRInfrastructureError,
    FasterWhisperAdapter,
)


class FakeWhisperModel:
    init_calls: list[tuple[tuple, dict]] = []

    def __init__(self, *args, **kwargs) -> None:
        self.init_calls.append((args, kwargs))
        self.transcribe_calls: list[tuple[str, dict]] = []

    def transcribe(self, audio_path: str, **kwargs):
        self.transcribe_calls.append((audio_path, kwargs))
        segments = iter(
            [
                SimpleNamespace(
                    start=0.0,
                    end=1.25,
                    text=" 今日は晴れです。",
                    avg_logprob=-0.2,
                    no_speech_prob=0.01,
                )
            ]
        )
        info = SimpleNamespace(language="ja", language_probability=0.99, duration=1.25)
        return segments, info


def patch_cuda(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(asr_module, "_prepare_windows_cuda_runtime", lambda: [])
    monkeypatch.setattr(asr_module.ctranslate2, "get_cuda_device_count", lambda: 1)
    monkeypatch.setattr(
        asr_module.ctranslate2,
        "get_supported_compute_types",
        lambda device, index: {"float16", "float32"},
    )


def test_asr_config_fixes_japanese_cuda_policy() -> None:
    config = ASRConfig()
    assert config.model_name == "large-v3"
    assert config.device == "cuda"
    assert config.compute_type == "float16"
    assert config.language == "ja"
    assert config.vad_filter is False

    with pytest.raises(ValidationError):
        ASRConfig(device="cpu")
    with pytest.raises(ValidationError):
        ASRConfig(language="en")
    with pytest.raises(ValidationError):
        ASRConfig(vad_filter=True)
    with pytest.raises(ValidationError):
        ASRConfig(model_name="small")


def test_faster_whisper_adapter_loads_once_and_maps_result(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    patch_cuda(monkeypatch)
    FakeWhisperModel.init_calls.clear()
    adapter = FasterWhisperAdapter(ASRConfig(), model_factory=FakeWhisperModel)

    first = adapter.transcribe(Path("first.wav"))
    second = adapter.transcribe(Path("second.wav"))

    assert len(FakeWhisperModel.init_calls) == 1
    assert first.text == "今日は晴れです。"
    assert first.language == "ja"
    assert first.segments[0].start_sec == 0.0
    assert first.segments[0].end_sec == 1.25
    assert second.text == first.text
    assert len(adapter._model.transcribe_calls) == 2
    _, kwargs = adapter._model.transcribe_calls[0]
    assert kwargs["language"] == "ja"
    assert kwargs["vad_filter"] is False


def test_cuda_unavailable_fails_before_model_initialization(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(asr_module, "_prepare_windows_cuda_runtime", lambda: [])
    monkeypatch.setattr(asr_module.ctranslate2, "get_cuda_device_count", lambda: 0)
    called = False

    def factory(*args, **kwargs):
        nonlocal called
        called = True

    with pytest.raises(ASRInfrastructureError, match="CUDA device 0 is unavailable"):
        FasterWhisperAdapter(ASRConfig(), model_factory=factory)
    assert called is False


def test_unsupported_compute_type_fails_before_model_initialization(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(asr_module, "_prepare_windows_cuda_runtime", lambda: [])
    monkeypatch.setattr(asr_module.ctranslate2, "get_cuda_device_count", lambda: 1)
    monkeypatch.setattr(
        asr_module.ctranslate2,
        "get_supported_compute_types",
        lambda device, index: {"float32"},
    )
    with pytest.raises(ASRInfrastructureError, match="does not support"):
        FasterWhisperAdapter(ASRConfig(), model_factory=FakeWhisperModel)


def test_cuda_inference_runtime_error_fails_fast(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    patch_cuda(monkeypatch)

    class BrokenModel:
        def __init__(self, *args, **kwargs) -> None:
            pass

        def transcribe(self, audio_path: str, **kwargs):
            def broken_segments():
                raise RuntimeError("cuDNN unavailable")
                yield

            return broken_segments(), SimpleNamespace(language="ja")

    adapter = FasterWhisperAdapter(ASRConfig(), model_factory=BrokenModel)
    with pytest.raises(ASRInfrastructureError, match="CUDA inference failed"):
        adapter.transcribe(Path("sample.wav"))
