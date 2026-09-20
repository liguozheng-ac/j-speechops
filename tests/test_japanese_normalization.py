from __future__ import annotations

import pytest

from j_speech_ops.japanese_normalization import (
    JapaneseNormalizer,
    NormalizationSampleError,
)


@pytest.mark.parametrize(
    ("raw_text", "expected"),
    [
        ("３５００円", "3500円"),
        ("ﾁｪｯｸｲﾝ", "チェックイン"),
        ("ＡＩとChatGPT", "AIとChatGPT"),
        ("Ｗｉ－Ｆｉ", "Wi-Fi"),
        ("「予約」、確認！", "「予約」、確認!"),
        ("日本語   English\t混在", "日本語 English 混在"),
        ("テスト😊", "テスト😊"),
    ],
)
def test_normalization_baseline(raw_text: str, expected: str) -> None:
    result = JapaneseNormalizer().normalize(raw_text)
    assert result.raw_text == raw_text
    assert result.normalized_text == expected
    assert result.jaconv_version == "0.5.0"


def test_normalization_records_transformations_deterministically() -> None:
    normalizer = JapaneseNormalizer()
    first = normalizer.normalize("  Ｗｉ－Ｆｉ  １５時  ")
    second = normalizer.normalize("  Ｗｉ－Ｆｉ  １５時  ")
    assert first == second
    assert first.transformations == (
        "jaconv_unicode_width_normalization",
        "latin_hyphen_normalization",
        "whitespace_normalization",
    )


@pytest.mark.parametrize(
    ("raw_text", "expected"),
    [
        ("3,500円", "3500円"),
        ("12,000円", "12000円"),
        ("123,456円", "123456円"),
        ("1,234,567円", "1234567円"),
        ("３，５００円", "3500円"),
    ],
)
def test_normalization_removes_valid_numeric_thousands_separators(
    raw_text: str, expected: str
) -> None:
    result = JapaneseNormalizer().normalize(raw_text)
    assert result.normalized_text == expected
    assert "numeric_thousands_separator_removed" in result.transformations


@pytest.mark.parametrize(
    "raw_text",
    [
        "12,34円",
        "3.5%",
        "こんにちは,世界",
        "A,B",
        "1,234.5",
    ],
)
def test_normalization_preserves_non_grouping_commas_and_decimals(
    raw_text: str,
) -> None:
    result = JapaneseNormalizer().normalize(raw_text)
    assert result.normalized_text == raw_text
    assert "numeric_thousands_separator_removed" not in result.transformations


def test_normalization_rejects_empty_text() -> None:
    with pytest.raises(NormalizationSampleError):
        JapaneseNormalizer().normalize(" \t ")
