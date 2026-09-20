from pathlib import Path

import pytest

from j_speech_ops.manifest import ManifestValidationError, iter_manifest, write_manifest
from j_speech_ops.schemas import DatasetMetadata, SpeechSample


def make_sample() -> SpeechSample:
    return SpeechSample.model_validate(
        {
            "sample_id": "JP_0001",
            "dataset_id": "demo-ja",
            "audio": {"path": "audio/JP_0001.wav", "duration_sec": 2.5},
            "text": {"reference_text": "東京都千代田区に向かいます。"},
            "source": {"source_name": "demo", "rights_status": "cleared"},
            "metadata": {"reviewed": True, "tags": ["clean", "studio"]},
        }
    )


def test_manifest_round_trip(tmp_path: Path) -> None:
    path = tmp_path / "manifest.jsonl"
    expected = [make_sample()]
    assert write_manifest(path, expected) == 1
    assert list(iter_manifest(path)) == expected
    assert len(path.read_text(encoding="utf-8").splitlines()) == 1


def test_malformed_json_after_valid_record_reports_exact_line(tmp_path: Path) -> None:
    path = tmp_path / "broken.jsonl"
    write_manifest(path, [make_sample()])
    with path.open("a", encoding="utf-8") as handle:
        handle.write('{"sample_id":\n')
    with pytest.raises(ManifestValidationError) as captured:
        list(iter_manifest(path))
    assert str(path) in str(captured.value)
    assert captured.value.line_number == 2
    assert "malformed JSON" in str(captured.value)


def test_invalid_sample_reports_filename_and_line(tmp_path: Path) -> None:
    path = tmp_path / "invalid.jsonl"
    path.write_text('{"sample_id":"only-an-id"}\n', encoding="utf-8")
    with pytest.raises(ManifestValidationError) as captured:
        list(iter_manifest(path))
    assert str(path) in str(captured.value)
    assert captured.value.line_number == 1
    assert "invalid SpeechSample" in str(captured.value)


def test_repository_examples_validate() -> None:
    project_root = Path(__file__).resolve().parents[1]
    samples = list(iter_manifest(project_root / "examples" / "sample_manifest.jsonl"))
    dataset = DatasetMetadata.model_validate_json(
        (project_root / "examples" / "dataset.json").read_text(encoding="utf-8")
    )
    assert len(samples) == 3
    assert {sample.dataset_id for sample in samples} == {dataset.dataset_id}
    assert dataset.manifest == "sample_manifest.jsonl"
