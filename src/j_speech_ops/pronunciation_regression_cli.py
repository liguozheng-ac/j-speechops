"""CLI for Stage 10A pronunciation evidence and regression operations."""

from __future__ import annotations

import argparse
from pathlib import Path

from .pronunciation_regression import (
    PronunciationEvidencePipeline,
    capture_pronunciation_baseline,
    compare_pronunciation_regression,
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Build Japanese pronunciation evidence and regression artifacts"
    )
    commands = parser.add_subparsers(dest="command", required=True)

    build = commands.add_parser("build", help="build Stage 10A pronunciation evidence")
    build.add_argument("--artifact-manifest", required=True, type=Path)
    build.add_argument("--synthesis-runs", required=True, type=Path)
    build.add_argument("--qa-results", required=True, type=Path)
    build.add_argument("--qa-pass-manifest", required=True, type=Path)
    build.add_argument("--curation-report", required=True, type=Path)
    build.add_argument(
        "--output-dir",
        type=Path,
        default=Path("outputs/stage10a_pronunciation_regression"),
    )

    capture = commands.add_parser(
        "capture-baseline", help="capture a persistent regression baseline"
    )
    capture.add_argument("--evidence", required=True, type=Path)
    capture.add_argument("--output", required=True, type=Path)

    compare = commands.add_parser(
        "compare", help="compare current evidence with a captured baseline"
    )
    compare.add_argument("--evidence", required=True, type=Path)
    compare.add_argument("--baseline", required=True, type=Path)
    compare.add_argument(
        "--output-dir",
        type=Path,
        default=Path("outputs/stage10a_pronunciation_regression"),
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.command == "build":
        outputs = PronunciationEvidencePipeline(output_root=args.output_dir).build(
            artifact_manifest=args.artifact_manifest,
            synthesis_runs_manifest=args.synthesis_runs,
            qa_results_manifest=args.qa_results,
            qa_pass_manifest=args.qa_pass_manifest,
            curation_report=args.curation_report,
        )
        print(f"Evidence Records:  {outputs.total}")
        print(f"Review Candidates: {outputs.review_count}")
        print(f"Evidence:          {outputs.evidence_manifest}")
        print(f"Review Queue:      {outputs.review_candidates}")
        return 0
    if args.command == "capture-baseline":
        count = capture_pronunciation_baseline(
            evidence_manifest=args.evidence,
            output_path=args.output,
        )
        print(f"Baseline Cases: {count}")
        print(f"Baseline:       {args.output.resolve()}")
        return 0
    outputs = compare_pronunciation_regression(
        evidence_manifest=args.evidence,
        baseline_manifest=args.baseline,
        output_root=args.output_dir,
    )
    summary = outputs.summary
    print(f"Compared Cases:         {summary.compared_cases}")
    print(f"No Regression Detected: {summary.no_regression_detected}")
    print(f"Review Candidates:      {summary.review_candidates}")
    print(f"Provenance-only Changes: {summary.provenance_only_changes}")
    print(f"Results:                {outputs.results_manifest}")
    print(f"Review Queue:           {outputs.review_candidates}")
    print(f"Summary:                {outputs.summary_file}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
