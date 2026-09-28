"""Temporary Stage 8 WAV-tail diagnostic; not part of the production pipeline."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import numpy as np
import soundfile as sf


def _dbfs(value: float) -> float | None:
    if value <= 0.0:
        return None
    return 20.0 * math.log10(value)


def _window_metrics(mono: np.ndarray, sample_rate: int, milliseconds: int) -> dict:
    count = min(len(mono), max(1, round(sample_rate * milliseconds / 1000)))
    tail = mono[-count:]
    rms = float(np.sqrt(np.mean(np.square(tail, dtype=np.float64))))
    peak = float(np.max(np.abs(tail)))
    return {
        "window_ms": milliseconds,
        "sample_count": count,
        "rms": rms,
        "rms_dbfs": _dbfs(rms),
        "peak": peak,
        "peak_dbfs": _dbfs(peak),
    }


def analyze_audio_tail(path: Path, *, silence_dbfs: float = -50.0) -> dict:
    resolved = path.resolve()
    info = sf.info(resolved)
    audio, sample_rate = sf.read(resolved, dtype="float64", always_2d=True)
    if len(audio) == 0:
        raise ValueError(f"empty audio: {resolved}")
    mono = np.max(np.abs(audio), axis=1)
    absolute = np.abs(audio)
    peak = float(np.max(absolute))
    rms = float(np.sqrt(np.mean(np.square(audio, dtype=np.float64))))
    threshold = 10.0 ** (silence_dbfs / 20.0)
    audible = np.flatnonzero(mono > threshold)
    nonzero = np.flatnonzero(mono > 0.0)
    last_audible = int(audible[-1]) if len(audible) else None
    last_nonzero = int(nonzero[-1]) if len(nonzero) else None
    trailing_samples = len(mono) if last_audible is None else len(mono) - last_audible - 1
    exact_zero_trailing = (
        len(mono) if last_nonzero is None else len(mono) - last_nonzero - 1
    )
    trailing_seconds = trailing_samples / sample_rate
    if trailing_seconds < 0.010:
        tail_classification = "no_detectable_trailing_silence"
    elif trailing_seconds < 0.100:
        tail_classification = "short_trailing_silence"
    else:
        tail_classification = "substantial_trailing_silence"
    final_frame_peak = float(mono[-1])
    return {
        "path": str(resolved),
        "filename": resolved.name,
        "format": info.format,
        "subtype": info.subtype,
        "sample_rate_hz": sample_rate,
        "channels": int(audio.shape[1]),
        "frames": int(audio.shape[0]),
        "duration_sec": len(audio) / sample_rate,
        "file_size_bytes": resolved.stat().st_size,
        "peak_amplitude": peak,
        "peak_dbfs": _dbfs(peak),
        "rms_energy": rms,
        "rms_dbfs": _dbfs(rms),
        "silence_threshold_dbfs": silence_dbfs,
        "silence_threshold_amplitude": threshold,
        "last_non_silent_sample_index": last_audible,
        "last_non_silent_position_sec": (
            last_audible / sample_rate if last_audible is not None else None
        ),
        "trailing_silence_samples": trailing_samples,
        "trailing_silence_sec": trailing_seconds,
        "exact_zero_trailing_samples": exact_zero_trailing,
        "exact_zero_trailing_sec": exact_zero_trailing / sample_rate,
        "final_frame_peak_amplitude": final_frame_peak,
        "final_frame_peak_dbfs": _dbfs(final_frame_peak),
        "tail_classification": tail_classification,
        "tail_windows": [
            _window_metrics(mono, sample_rate, milliseconds)
            for milliseconds in (10, 20, 50, 100, 300)
        ],
    }


def write_silence_padded_copy(
    source: Path,
    destination: Path,
    *,
    padding_ms: int,
) -> Path:
    if padding_ms <= 0:
        raise ValueError("padding must be positive")
    info = sf.info(source)
    audio, sample_rate = sf.read(source, dtype="float64", always_2d=True)
    padding_frames = round(sample_rate * padding_ms / 1000)
    padded = np.concatenate(
        [audio, np.zeros((padding_frames, audio.shape[1]), dtype=np.float64)], axis=0
    )
    destination.parent.mkdir(parents=True, exist_ok=True)
    sf.write(destination, padded, sample_rate, format="WAV", subtype=info.subtype)
    written, written_rate = sf.read(destination, dtype="float64", always_2d=True)
    if written_rate != sample_rate or len(written) != len(audio) + padding_frames:
        raise RuntimeError("padded WAV length verification failed")
    if not np.array_equal(written[: len(audio)], audio):
        raise RuntimeError("padding changed original PCM samples")
    if np.any(written[len(audio) :] != 0.0):
        raise RuntimeError("padding region is not silent")
    return destination


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Inspect existing WAV tails without changing production artifacts"
    )
    parser.add_argument("paths", nargs="+", type=Path)
    parser.add_argument("--silence-dbfs", type=float, default=-50.0)
    parser.add_argument("--json-output", type=Path)
    parser.add_argument("--padding-ms", nargs="*", type=int, default=[])
    parser.add_argument("--padding-output-dir", type=Path)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    reports = [
        analyze_audio_tail(path, silence_dbfs=args.silence_dbfs) for path in args.paths
    ]
    if args.padding_ms:
        if args.padding_output_dir is None:
            raise SystemExit("--padding-output-dir is required with --padding-ms")
        for path, report in zip(args.paths, reports, strict=True):
            copies = []
            for milliseconds in args.padding_ms:
                destination = (
                    args.padding_output_dir
                    / f"{path.stem}.tailpad-{milliseconds}ms.wav"
                )
                write_silence_padded_copy(
                    path, destination, padding_ms=milliseconds
                )
                copies.append(str(destination.resolve()))
            report["padded_copies"] = copies
    rendered = json.dumps(reports, ensure_ascii=False, indent=2) + "\n"
    if args.json_output:
        args.json_output.parent.mkdir(parents=True, exist_ok=True)
        args.json_output.write_text(rendered, encoding="utf-8")
    print(rendered, end="")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
