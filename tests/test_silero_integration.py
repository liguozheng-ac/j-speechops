import pytest
import torch

from j_speech_ops.vad import SileroVAD, VADConfig


@pytest.mark.integration
def test_installed_silero_model_loads_and_runs_inference() -> None:
    vad = SileroVAD(VADConfig())
    regions = vad.detect(torch.zeros(16_000, dtype=torch.float32), 16_000)

    assert vad.backend_name == "silero-vad"
    assert vad.backend_version
    assert isinstance(regions, list)
    assert all(0 <= region.start_sample < region.end_sample <= 16_000 for region in regions)

