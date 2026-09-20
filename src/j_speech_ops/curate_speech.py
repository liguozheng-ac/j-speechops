"""Command-line entry point for Stage 4 speech-data curation."""

from __future__ import annotations

import argparse
from pathlib import Path

from .curation import CurationPolicy, SpeechCurationPipeline


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Route transcribed speech samples through Stage 4 curation"
    )
    parser.add_argument("--manifest", required=True, type=Path)
    parser.add_argument("--dataset-root", type=Path, default=Path("."))
    parser.add_argument("--output-dir", type=Path, default=Path("data"))
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--min-duration-sec", type=float, default=0.5)
    parser.add_argument("--max-duration-sec", type=float, default=30.0)
    parser.add_argument("--min-chars-per-sec", type=float, default=1.0)
    parser.add_argument("--max-chars-per-sec", type=float, default=20.0)
    parser.add_argument("--min-japanese-script-ratio", type=float, default=0.3)
    parser.add_argument("--repetition-min-count", type=int, default=3)
    parser.add_argument("--repetition-max-unit-chars", type=int, default=20)
    parser.add_argument(
        "--allow-unknown-rights",
        action="store_true",
        help="Do not route UNKNOWN rights to REVIEW; RESTRICTED always drops",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    dataset_root = args.dataset_root.resolve()
    output_root = args.output_dir
    if not output_root.is_absolute():
        output_root = dataset_root / output_root
    policy = CurationPolicy(
        min_duration_sec=args.min_duration_sec,
        max_duration_sec=args.max_duration_sec,
        min_chars_per_sec=args.min_chars_per_sec,
        max_chars_per_sec=args.max_chars_per_sec,
        min_japanese_script_ratio=args.min_japanese_script_ratio,
        repetition_min_count=args.repetition_min_count,
        repetition_max_unit_chars=args.repetition_max_unit_chars,
        require_cleared_rights=not args.allow_unknown_rights,
    )
    outputs = SpeechCurationPipeline(
        dataset_root=dataset_root,
        output_root=output_root,
        policy=policy,
    ).run(args.manifest, force=args.force)
    summary = outputs.summary
    print(f"Total Samples:   {summary.total_samples}")
    print(f"PASS:            {summary.pass_count}")
    print(f"REVIEW:          {summary.review_count}")
    print(f"DROP:            {summary.drop_count}")
    print(f"Total Duration:  {summary.total_audio_duration_sec:.3f} sec")
    print(f"PASS Duration:   {summary.pass_duration_sec:.3f} sec")
    print(f"REVIEW Duration: {summary.review_duration_sec:.3f} sec")
    print(f"DROP Duration:   {summary.drop_duration_sec:.3f} sec")
    if summary.top_issue_codes:
        rendered = ", ".join(
            f"{item.code}={item.count}" for item in summary.top_issue_codes
        )
        print(f"Top Issues:      {rendered}")
    else:
        print("Top Issues:      none")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

