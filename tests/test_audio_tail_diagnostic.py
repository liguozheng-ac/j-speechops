from __future__ import annotations

import importlib.util
from pathlib import Path

import numpy as np
import soundfile as sf


SCRIPT = (
    Path(__file__).parents[1]
    / "tools"
    / "diagnostics"
    / "analyze_audio_tail.py"
)
SPEC = importlib.util.spec_from_file_location("audio_tail_diagnostic", SCRIPT)
assert SPEC is not None and SPEC.loader is not None
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


def test_tail_analysis_distinguishes_silence_from_hard_end(tmp_path: Path) -> None:
    sample_rate = 24_000
    active = np.full(sample_rate, 0.1, dtype=np.float64)
    with_silence = np.concatenate([active, np.zeros(sample_rate // 5)])
    silent_path = tmp_path / "silence.wav"
    hard_path = tmp_path / "hard.wav"
    sf.write(silent_path, with_silence, sample_rate, subtype="PCM_16")
    sf.write(hard_path, active, sample_rate, subtype="PCM_16")

    silent = MODULE.analyze_audio_tail(silent_path)
    hard = MODULE.analyze_audio_tail(hard_path)

    assert silent["trailing_silence_sec"] == 0.2
    assert silent["tail_classification"] == "substantial_trailing_silence"
    assert hard["trailing_silence_sec"] == 0.0
    assert hard["tail_classification"] == "no_detectable_trailing_silence"


def test_padding_preserves_original_samples_and_appends_exact_silence(
    tmp_path: Path,
) -> None:
    sample_rate = 24_000
    source = tmp_path / "source.wav"
    destination = tmp_path / "padded.wav"
    audio = np.linspace(-0.25, 0.25, 2_400, dtype=np.float64)
    sf.write(source, audio, sample_rate, subtype="PCM_16")

    MODULE.write_silence_padded_copy(source, destination, padding_ms=300)
    before, _ = sf.read(source, dtype="float64", always_2d=True)
    after, _ = sf.read(destination, dtype="float64", always_2d=True)

    assert np.array_equal(after[: len(before)], before)
    assert len(after) == len(before) + 7_200
    assert np.count_nonzero(after[len(before) :]) == 0


def test_diagnostic_does_not_change_canonical_schema_files() -> None:
    import hashlib

    root = Path(__file__).parents[1]
    speech = hashlib.sha256(
        (root / "schemas" / "speech_sample.schema.json").read_bytes()
    ).hexdigest().upper()
    text = hashlib.sha256(
        (root / "schemas" / "tts_text_sample.schema.json").read_bytes()
    ).hexdigest().upper()
    assert speech == "DEC2C98402E4433DF64E0B0F2280A73C7EE8CF3EE03232D7BCC8EC2EF7B70435"
    assert text == "169E87DFA910A64BEF2E1200EB94927C96857065C76629D7FB07CB107DE74EDE"
