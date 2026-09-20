"""CLI for Stage 6 Japanese text normalization and reading preparation."""

from __future__ import annotations

import argparse
from pathlib import Path

from .japanese_normalization import JapaneseNormalizer
from .japanese_reading import OpenJTalkReadingProvider, ReadingOverrides
from .text_preparation import TextPreparationPipeline


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Normalize Japanese TTS text and prepare OpenJTalk readings"
    )
    parser.add_argument("--manifest", required=True, type=Path)
    parser.add_argument("--output-dir", type=Path, default=Path("data"))
    parser.add_argument("--overrides", type=Path)
    parser.add_argument("--force", action="store_true")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    overrides = (
        ReadingOverrides.load(args.overrides)
        if args.overrides is not None
        else ReadingOverrides.empty()
    )
    pipeline = TextPreparationPipeline(
        output_root=args.output_dir,
        normalizer=JapaneseNormalizer(),
        reading_provider=OpenJTalkReadingProvider(),
        overrides=overrides,
    )
    outputs = pipeline.run(args.manifest, force=args.force)
    summary = outputs.summary
    print(f"Total Samples:    {summary.total_samples}")
    print(f"Reading Prepared: {summary.reading_prepared}")
    print(f"Normalized Only:  {summary.normalized_only}")
    print(f"Skipped:          {summary.skipped}")
    print(f"Failed:           {summary.failed}")
    print(f"Normalized:       {outputs.normalized_manifest}")
    print(f"Reading Prepared: {outputs.reading_prepared_manifest}")
    print(f"Report:           {outputs.report_file}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
