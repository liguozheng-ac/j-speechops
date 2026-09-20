"""Command-line entry point for Stage 2 audio preparation."""

from __future__ import annotations

import argparse
from pathlib import Path

from .audio_preparation import AudioPreparationConfig, AudioPreparationPipeline
from .vad import SileroVAD, VADConfig


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Prepare and segment speech audio with Silero VAD")
    parser.add_argument("--manifest", required=True, type=Path, help="Input SpeechSample JSONL")
    parser.add_argument("--dataset-root", type=Path, default=Path("."), help="Base for relative audio paths")
    parser.add_argument("--output-dir", type=Path, default=Path("data"), help="Stage 2 output root inside dataset root")
    parser.add_argument("--threshold", type=float, default=0.5)
    parser.add_argument("--min-speech-duration-ms", type=int, default=250)
    parser.add_argument("--min-silence-duration-ms", type=int, default=100)
    parser.add_argument("--speech-pad-ms", type=int, default=30)
    parser.add_argument("--max-speech-duration-s", type=float, default=30.0)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    vad_config = VADConfig(
        threshold=args.threshold,
        min_speech_duration_ms=args.min_speech_duration_ms,
        min_silence_duration_ms=args.min_silence_duration_ms,
        speech_pad_ms=args.speech_pad_ms,
        max_speech_duration_s=args.max_speech_duration_s,
    )
    config = AudioPreparationConfig(vad=vad_config)
    vad = SileroVAD(vad_config)
    dataset_root = args.dataset_root.resolve()
    output_root = args.output_dir
    if not output_root.is_absolute():
        output_root = dataset_root / output_root
    pipeline = AudioPreparationPipeline(
        dataset_root=dataset_root,
        output_root=output_root,
        vad=vad,
        config=config,
    )
    outputs = pipeline.run(args.manifest)
    summary = outputs.summary
    print(f"Input samples:     {summary.input_samples}")
    print(f"Prepared:          {summary.prepared}")
    print(f"Dropped:           {summary.dropped}")
    print(f"Segments created:  {summary.segments_created}")
    print(f"Errors:             {summary.errors}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
