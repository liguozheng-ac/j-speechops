from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest

from j_speech_ops.text_manifest import (
    TextManifestValidationError,
    iter_text_manifest,
    write_text_manifest,
)
from j_speech_ops.tts_text import TTSTextContent, TTSTextSample, TTSTextSource


def make_sample(sample_id: str = "TEXT_000001") -> TTSTextSample:
    return TTSTextSample(
        sample_id=sample_id,
        dataset_id="text-demo",
        text=TTSTextContent(raw_text="料金は3,500円です。"),
        source=TTSTextSource(source_type="file", source_name="demo.txt"),
    )


def test_text_manifest_round_trip(tmp_path: Path) -> None:
    path = tmp_path / "tts_text_samples.jsonl"
    expected = [make_sample()]
    assert write_text_manifest(path, expected) == 1
    assert list(iter_text_manifest(path)) == expected
    assert len(path.read_text(encoding="utf-8").splitlines()) == 1


def test_text_manifest_error_localizes_line_and_sample_id(tmp_path: Path) -> None:
    path = tmp_path / "invalid.jsonl"
    path.write_text(
        '{"sample_id":"KNOWN_ID","dataset_id":"demo"}\n', encoding="utf-8"
    )
    with pytest.raises(TextManifestValidationError) as captured:
        list(iter_text_manifest(path))
    assert captured.value.line_number == 1
    assert captured.value.sample_id == "KNOWN_ID"
    assert str(path) in str(captured.value)


def test_text_manifest_rejects_duplicate_sample_ids(tmp_path: Path) -> None:
    path = tmp_path / "duplicate.jsonl"
    encoded = make_sample().model_dump_json()
    path.write_text(f"{encoded}\n{encoded}\n", encoding="utf-8")
    with pytest.raises(TextManifestValidationError, match="duplicate sample_id"):
        list(iter_text_manifest(path))


def test_tts_schema_export_matches_model_and_speech_schema_is_unchanged() -> None:
    root = Path(__file__).resolve().parents[1]
    exported = json.loads(
        (root / "schemas" / "tts_text_sample.schema.json").read_text(encoding="utf-8")
    )
    assert exported == TTSTextSample.model_json_schema()
    speech_hash = hashlib.sha256(
        (root / "schemas" / "speech_sample.schema.json").read_bytes()
    ).hexdigest()
    assert speech_hash == "dec2c98402e4433df64e0b0f2280a73c7ee8cf3ee03232d7bcc8ec2ef7b70435"
