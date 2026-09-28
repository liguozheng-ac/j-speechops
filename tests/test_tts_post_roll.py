from __future__ import annotations

import importlib.util
from pathlib import Path

import numpy as np
import pytest


WORKER = Path(__file__).parents[1] / "runtime" / "qwen3_tts_worker.py"
SPEC = importlib.util.spec_from_file_location("qwen3_tts_worker", WORKER)
assert SPEC is not None and SPEC.loader is not None
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


@pytest.mark.parametrize(
    ("duration_ms", "expected_padding_frames"),
    ((0, 0), (300, 7_200), (500, 12_000)),
)
def test_tail_padding_preserves_samples_and_appends_exact_zeros(
    duration_ms: int, expected_padding_frames: int
) -> None:
    original = np.linspace(-0.5, 0.5, 2_400, dtype=np.float32)

    processed = MODULE.append_tail_padding(original, 24_000, duration_ms)

    assert processed.dtype == original.dtype
    assert len(processed) == len(original) + expected_padding_frames
    assert np.array_equal(processed[: len(original)], original)
    assert np.count_nonzero(processed[len(original) :]) == 0
    assert not np.shares_memory(processed, original)


def test_tail_padding_preserves_channel_shape() -> None:
    original = np.ones((10, 2), dtype=np.float64)
    processed = MODULE.append_tail_padding(original, 1_000, 100)
    assert processed.shape == (110, 2)
    assert np.array_equal(processed[:10], original)
    assert np.array_equal(processed[10:], np.zeros((100, 2)))


def test_tail_padding_rejects_invalid_inputs() -> None:
    with pytest.raises(ValueError, match="must not be negative"):
        MODULE.append_tail_padding(np.ones(10), 24_000, -1)
    with pytest.raises(ValueError, match="sample rate"):
        MODULE.append_tail_padding(np.ones(10), 0, 300)
