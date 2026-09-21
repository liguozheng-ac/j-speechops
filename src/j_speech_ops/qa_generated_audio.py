"""Core CLI for Stage 9 generated-audio content QA."""

from __future__ import annotations

import argparse
from pathlib import Path

from .asr import ASRConfig, LazyFasterWhisperAdapter
from .generated_audio_qa import GeneratedAudioQAPipeline, detect_nvidia_gpu_name
from .japanese_normalization import JapaneseNormalizer
from .japanese_reading import OpenJTalkReadingProvider


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Validate generated WAVs and collect Whisper round-trip evidence"
    )
    parser.add_argument("--manifest", required=True, type=Path)
    parser.add_argument("--model-path", required=True, type=Path)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("outputs/stage9_generated_audio_qa"),
    )
    parser.add_argument("--force", action="store_true")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    config = ASRConfig(
        model_path=str(args.model_path.resolve()),
        local_files_only=True,
    )
    adapter = LazyFasterWhisperAdapter(config)
    gpu_name = detect_nvidia_gpu_name(config.device_index)
    outputs = GeneratedAudioQAPipeline(
        output_root=args.output_dir,
        adapter=adapter,
        normalizer=JapaneseNormalizer(),
        reading_provider=OpenJTalkReadingProvider(),
        gpu_name=gpu_name,
    ).run(args.manifest, force=args.force)
    summary = outputs.summary
    print(f"Total Samples:    {summary.total_samples}")
    print(f"Attempted:        {summary.attempted_samples}")
    print(f"Skipped:          {summary.skipped_samples}")
    print(f"PASS:             {summary.passed_samples}")
    print(f"REVIEW:           {summary.review_samples}")
    print(f"DROP:             {summary.dropped_samples}")
    print(f"FAILED:           {summary.failed_samples}")
    print(f"Model Load (sec): {summary.asr_model_load_time_sec:.3f}")
    print(f"Batch Time (sec): {summary.batch_total_time_sec:.3f}")
    print(f"Results:          {outputs.results_manifest}")
    print(f"QA-pass Audio:    {outputs.pass_manifest}")
    print(f"Review Queue:     {outputs.review_queue}")
    print(f"Summary:          {outputs.summary_file}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
