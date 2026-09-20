"""CLI for Stage 7 Japanese TTS text curation."""

from __future__ import annotations

import argparse
from pathlib import Path

from .pronunciation_risks import PronunciationRiskWatchlist
from .tts_text_curation import TTSTextCurationPipeline


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Route prepared Japanese TTS text through Stage 7 curation"
    )
    parser.add_argument("--manifest", required=True, type=Path)
    parser.add_argument("--prep-report", required=True, type=Path)
    parser.add_argument("--risk-watchlist", required=True, type=Path)
    parser.add_argument("--output-dir", type=Path, default=Path("data"))
    parser.add_argument("--force", action="store_true")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    outputs = TTSTextCurationPipeline(
        output_root=args.output_dir,
        watchlist=PronunciationRiskWatchlist.load(args.risk_watchlist),
    ).run(args.manifest, args.prep_report, force=args.force)
    summary = outputs.summary
    print(f"Total Samples:               {summary.total_samples}")
    print(f"PASS:                        {summary.pass_count}")
    print(f"REVIEW:                      {summary.review_count}")
    print(f"DROP:                        {summary.drop_count}")
    print(f"Failed:                      {summary.failed_count}")
    if summary.top_issue_codes:
        rendered = ", ".join(
            f"{item.code}={item.count}" for item in summary.top_issue_codes
        )
        print(f"Top Issues:                  {rendered}")
    else:
        print("Top Issues:                  none")
    print(
        "Override-applied Samples:    "
        f"{summary.override_applied_sample_count}"
    )
    print(
        "Pronunciation-risk Samples:  "
        f"{summary.pronunciation_risk_sample_count}"
    )
    print(f"Curated:                     {outputs.curated_manifest}")
    print(f"Synthesis Ready:             {outputs.synthesis_ready_manifest}")
    print(f"Report:                      {outputs.report_file}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
