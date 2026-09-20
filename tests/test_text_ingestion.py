from __future__ import annotations

import hashlib
from pathlib import Path

import pytest

from j_speech_ops.text_ingestion import (
    TextIngestionConfig,
    TextIngestionError,
    ingest_text_file,
    ingest_text_to_manifest,
)
from j_speech_ops.text_manifest import iter_text_manifest


def config() -> TextIngestionConfig:
    return TextIngestionConfig(
        dataset_id="hospitality-demo",
        source_name="hospitality.txt",
        rights_status="cleared",
        intended_use="tts_production",
        domain="hospitality",
    )


def test_ingestion_preserves_text_and_creates_stable_ids(tmp_path: Path) -> None:
    source = tmp_path / "hospitality.txt"
    source.write_bytes(
        "  佐伯様、ご予約は15時30分でございます。  \r\n"
        "Wi-Fiのパスワードをご確認ください。\r\n".encode("utf-8")
    )
    first = ingest_text_file(source, config())
    second = ingest_text_file(source, config())
    assert first == second
    assert [sample.sample_id for sample in first] == [
        "HOSPITALITY_DEMO_000001",
        "HOSPITALITY_DEMO_000002",
    ]
    assert first[0].text.raw_text == "  佐伯様、ご予約は15時30分でございます。  "
    assert first[1].text.raw_text == "Wi-Fiのパスワードをご確認ください。"
    assert first[0].source.source_id == "line:1"


@pytest.mark.parametrize("content", ["", "   \n", "有効です。\n\t\u3000\n"])
def test_ingestion_rejects_empty_input_or_line(tmp_path: Path, content: str) -> None:
    source = tmp_path / "bad.txt"
    source.write_text(content, encoding="utf-8")
    with pytest.raises(TextIngestionError):
        ingest_text_file(source, config())


def test_ingestion_manifest_is_deterministic_and_reloadable(tmp_path: Path) -> None:
    source = tmp_path / "input.txt"
    source.write_text(
        "料金は3,500円です。\n9月21日のご予約でございます。\n",
        encoding="utf-8",
    )
    output = tmp_path / "data" / "manifests" / "tts_text_samples.jsonl"
    first = ingest_text_to_manifest(source, output, config())
    first_hash = hashlib.sha256(output.read_bytes()).hexdigest()
    second = ingest_text_to_manifest(source, output, config())
    second_hash = hashlib.sha256(output.read_bytes()).hexdigest()
    assert first_hash == second_hash
    assert first.samples == second.samples
    assert list(iter_text_manifest(output)) == list(first.samples)
    assert first.sample_count == 2


def test_non_ascii_dataset_id_gets_deterministic_fallback_prefix(tmp_path: Path) -> None:
    source = tmp_path / "input.txt"
    source.write_text("こんにちは。\n", encoding="utf-8")
    japanese_config = TextIngestionConfig(
        dataset_id="接客",
        source_name="input.txt",
    )
    first = ingest_text_file(source, japanese_config)[0].sample_id
    second = ingest_text_file(source, japanese_config)[0].sample_id
    assert first == second
    assert first.startswith("TEXT_")


def test_repository_japanese_examples_match_canonical_manifest() -> None:
    root = Path(__file__).resolve().parents[1]
    raw_lines = (root / "examples" / "hospitality_text.txt").read_text(
        encoding="utf-8"
    ).splitlines()
    samples = list(
        iter_text_manifest(root / "examples" / "tts_text_samples.jsonl")
    )
    assert len(samples) == 5
    assert [sample.text.raw_text for sample in samples] == raw_lines
    assert all(sample.text.normalized_text is None for sample in samples)
