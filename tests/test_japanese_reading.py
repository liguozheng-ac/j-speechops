from __future__ import annotations

import json
from pathlib import Path

import pytest

import j_speech_ops.japanese_reading as reading_module
from j_speech_ops.japanese_normalization import JapaneseNormalizer
from j_speech_ops.japanese_reading import (
    OpenJTalkReadingProvider,
    ReadingInfrastructureError,
    ReadingOverrides,
    ReadingSampleError,
)


ROOT = Path(__file__).resolve().parents[1]


def test_openjtalk_regression_fixture() -> None:
    cases = json.loads(
        (ROOT / "tests" / "fixtures" / "japanese_text_frontend_regression.json").read_text(
            encoding="utf-8"
        )
    )
    normalizer = JapaneseNormalizer()
    provider = OpenJTalkReadingProvider()
    overrides = ReadingOverrides.empty()
    for index, case in enumerate(cases):
        normalized = normalizer.normalize(case["raw_text"])
        reading = provider.prepare(str(index), normalized.normalized_text, overrides)
        assert normalized.normalized_text == case["normalized_text"]
        assert reading.reading_kana == case["reading_kana"]


def test_provider_configuration_is_openjtalk_only() -> None:
    info = OpenJTalkReadingProvider().info
    assert info.provider == "openjtalk-frontend"
    assert info.package_name == "pyopenjtalk-plus"
    assert info.package_version == "0.4.1.post9"
    assert info.dictionary_identity is not None
    assert info.dictionary_version is None
    assert info.settings["run_marine"] is False
    assert info.settings["use_tsqyomi"] is False
    assert info.settings["use_sudachi_kanji_yomi"] is False
    assert info.settings["predict_nani"] is False


def test_explicit_override_wins_and_is_traceable() -> None:
    provider = OpenJTalkReadingProvider()
    overrides = ReadingOverrides.load(
        ROOT / "config" / "japanese_reading_overrides.json"
    )
    result = provider.prepare("override", "ChatGPTとWi-Fi", overrides)
    assert result.baseline_reading_kana == "シーエイチエーティージーピーティートワイファイ"
    assert result.reading_kana == "チャットジーピーティートワイファイ"
    assert result.override_count == 2
    assert {item.surface for item in result.applied_overrides} == {
        "ChatGPT",
        "Wi-Fi",
    }
    assert len(result.override_fingerprint_sha256 or "") == 64
    assert all(item.source for item in result.applied_overrides)


def test_unresolved_emoji_only_reading_is_not_a_success() -> None:
    with pytest.raises(ReadingSampleError):
        OpenJTalkReadingProvider().prepare("emoji", "😊", ReadingOverrides.empty())


def test_backend_initialization_failure_is_fail_fast(monkeypatch) -> None:
    def fail_import(name: str):
        raise ImportError(name)

    monkeypatch.setattr(reading_module.importlib, "import_module", fail_import)
    with pytest.raises(ReadingInfrastructureError):
        OpenJTalkReadingProvider()
