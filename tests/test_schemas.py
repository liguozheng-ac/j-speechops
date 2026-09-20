import pytest
from pydantic import ValidationError

from j_speech_ops.schemas import (
    CurationDecision,
    DatasetSplit,
    LineageInfo,
    ProcessingState,
    SourceInfo,
    SpeechSample,
)


def sample_payload() -> dict:
    return {
        "sample_id": "JP_0001",
        "dataset_id": "demo-ja",
        "audio": {"path": "audio/JP_0001.wav"},
        "source": {"source_name": "demo"},
    }


def test_valid_speech_sample() -> None:
    sample = SpeechSample.model_validate(sample_payload())
    assert sample.sample_id == "JP_0001"
    assert sample.split is DatasetSplit.UNSPLIT


def test_missing_sample_id_fails() -> None:
    payload = sample_payload()
    del payload["sample_id"]
    with pytest.raises(ValidationError):
        SpeechSample.model_validate(payload)


def test_invalid_enum_fails() -> None:
    payload = sample_payload()
    payload["split"] = "development"
    with pytest.raises(ValidationError):
        SpeechSample.model_validate(payload)


@pytest.mark.parametrize("bad_value", [{"x"}, ("tuple",), object(), float("nan")])
def test_non_json_metadata_fails(bad_value: object) -> None:
    payload = sample_payload()
    payload["metadata"] = {"bad": bad_value}
    with pytest.raises(ValidationError):
        SpeechSample.model_validate(payload)


@pytest.mark.parametrize("decision", list(CurationDecision))
def test_curation_decisions_validate_and_serialize(decision: CurationDecision) -> None:
    state = ProcessingState(curation_decision=decision)
    assert state.model_dump(mode="json")["curation_decision"] == decision.value


def test_source_rights_default_is_explicit() -> None:
    source = SourceInfo(source_name="demo")
    assert source.model_dump(mode="json")["rights_status"] == "unknown"


def test_valid_segmentation_lineage() -> None:
    lineage = LineageInfo(
        parent_sample_id="JP_PARENT", segment_start_sec=0.5, segment_end_sec=1.75
    )
    assert lineage.segment_end_sec == 1.75


def test_segmentation_end_before_start_fails() -> None:
    with pytest.raises(ValidationError):
        LineageInfo(
            parent_sample_id="JP_PARENT", segment_start_sec=2.0, segment_end_sec=1.0
        )


def test_negative_segmentation_start_fails() -> None:
    with pytest.raises(ValidationError):
        LineageInfo(
            parent_sample_id="JP_PARENT", segment_start_sec=-0.1, segment_end_sec=1.0
        )

