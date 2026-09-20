"""Command-line entry point for Stage 3 Japanese ASR."""

from __future__ import annotations

import argparse
from pathlib import Path

from .asr import ASRConfig, FasterWhisperAdapter
from .asr_pipeline import ASRBatchPipeline


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Transcribe Stage 2 speech segments with Japanese Whisper large-v3"
    )
    parser.add_argument("--manifest", required=True, type=Path)
    parser.add_argument("--dataset-root", type=Path, default=Path("."))
    parser.add_argument("--output-dir", type=Path, default=Path("data"))
    parser.add_argument("--model", choices=["large-v3"], default="large-v3")
    parser.add_argument("--device", choices=["cuda"], default="cuda")
    parser.add_argument("--device-index", type=int, default=0)
    parser.add_argument("--compute-type", choices=["float16"], default="float16")
    parser.add_argument("--language", choices=["ja"], default="ja")
    parser.add_argument("--beam-size", type=int, default=5)
    parser.add_argument("--download-root", type=Path)
    parser.add_argument("--local-files-only", action="store_true")
    parser.add_argument("--force", action="store_true")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    dataset_root = args.dataset_root.resolve()
    output_root = args.output_dir
    if not output_root.is_absolute():
        output_root = dataset_root / output_root
    config = ASRConfig(
        model_name=args.model,
        device=args.device,
        device_index=args.device_index,
        compute_type=args.compute_type,
        language=args.language,
        beam_size=args.beam_size,
        download_root=str(args.download_root) if args.download_root else None,
        local_files_only=args.local_files_only,
    )
    adapter = FasterWhisperAdapter(config)
    pipeline = ASRBatchPipeline(
        dataset_root=dataset_root,
        output_root=output_root,
        adapter=adapter,
    )
    outputs = pipeline.run(args.manifest, force=args.force)
    summary = outputs.summary
    print(f"Input samples:  {summary.input_samples}")
    print(f"Transcribed:    {summary.transcribed}")
    print(f"Skipped:        {summary.skipped}")
    print(f"Failed:         {summary.failed}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
