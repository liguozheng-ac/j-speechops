"""CLI for Stage 10B human review and release-gate operations."""

from __future__ import annotations

import argparse
from datetime import datetime
from pathlib import Path

from .human_review_release import (
    HumanReviewDecision,
    ReviewReasonCode,
    ReworkTarget,
    build_release_manifest,
    evaluate_release_gate,
    prepare_human_review_queue,
    record_human_review,
)


def _timestamp(value: str) -> datetime:
    try:
        result = datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError as exc:
        raise argparse.ArgumentTypeError("timestamp must be ISO-8601") from exc
    if result.utcoffset() is None:
        raise argparse.ArgumentTypeError("timestamp must include a timezone")
    return result


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Prepare human review and evaluate the conservative release gate"
    )
    commands = parser.add_subparsers(dest="command", required=True)

    prepare = commands.add_parser("prepare-review", help="build the mandatory review queue")
    prepare.add_argument("--artifact-manifest", required=True, type=Path)
    prepare.add_argument("--synthesis-runs", required=True, type=Path)
    prepare.add_argument("--qa-results", required=True, type=Path)
    prepare.add_argument("--pronunciation-evidence", required=True, type=Path)
    prepare.add_argument("--regression-results", required=True, type=Path)
    prepare.add_argument(
        "--output-dir", type=Path, default=Path("outputs/stage10b_human_review_release")
    )

    record = commands.add_parser("record-review", help="append a real human decision")
    record.add_argument("--queue", required=True, type=Path)
    record.add_argument("--records", required=True, type=Path)
    record.add_argument("--artifact-id", required=True)
    record.add_argument("--reviewer-id", required=True)
    record.add_argument(
        "--decision", required=True, choices=[item.value for item in HumanReviewDecision]
    )
    record.add_argument(
        "--reason",
        action="append",
        default=[],
        choices=[item.value for item in ReviewReasonCode],
    )
    record.add_argument(
        "--resolve-issue",
        action="append",
        default=[],
        help="machine review issue ID explicitly resolved by this human review",
    )
    record.add_argument("--notes")
    record.add_argument("--rework-target", choices=[item.value for item in ReworkTarget])
    record.add_argument("--reviewed-at", type=_timestamp)

    evaluate = commands.add_parser(
        "evaluate-release", help="evaluate current reviews against the release policy"
    )
    evaluate.add_argument("--queue", required=True, type=Path)
    evaluate.add_argument("--records", required=True, type=Path)
    evaluate.add_argument(
        "--output-dir", type=Path, default=Path("outputs/stage10b_human_review_release")
    )

    manifest = commands.add_parser(
        "build-release-manifest", help="write entries that passed the release gate"
    )
    manifest.add_argument("--queue", required=True, type=Path)
    manifest.add_argument("--records", required=True, type=Path)
    manifest.add_argument("--gate-results", required=True, type=Path)
    manifest.add_argument("--output", required=True, type=Path)
    manifest.add_argument("--released-at", type=_timestamp)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.command == "prepare-review":
        outputs = prepare_human_review_queue(
            artifact_manifest=args.artifact_manifest,
            synthesis_runs_manifest=args.synthesis_runs,
            qa_results_manifest=args.qa_results,
            pronunciation_evidence_manifest=args.pronunciation_evidence,
            regression_results_manifest=args.regression_results,
            output_root=args.output_dir,
        )
        print(f"Review Queue Items: {outputs.summary.total_items}")
        print(f"Pending Items:      {outputs.summary.pending_items}")
        print(f"Queue:              {outputs.queue_manifest}")
        print(f"Records:            {outputs.records_manifest}")
        print(f"Summary:            {outputs.summary_file}")
        return 0
    if args.command == "record-review":
        record = record_human_review(
            queue_manifest=args.queue,
            records_manifest=args.records,
            artifact_run_id=args.artifact_id,
            reviewer_id=args.reviewer_id,
            decision=HumanReviewDecision(args.decision),
            reason_codes=(ReviewReasonCode(value) for value in args.reason),
            resolved_issue_ids=args.resolve_issue,
            notes=args.notes,
            rework_target=(ReworkTarget(args.rework_target) if args.rework_target else None),
            reviewed_at=args.reviewed_at,
        )
        print(f"Review ID: {record.review_id}")
        print(f"Decision:  {record.decision.value}")
        print(f"Artifact:  {record.artifact_run_id}")
        return 0
    if args.command == "evaluate-release":
        outputs = evaluate_release_gate(
            queue_manifest=args.queue,
            records_manifest=args.records,
            output_root=args.output_dir,
        )
        print(f"Candidates:   {outputs.summary.total_candidates}")
        print(f"Released:     {outputs.summary.released}")
        print(f"Not Released: {outputs.summary.not_released}")
        print(f"Gate Results: {outputs.results_manifest}")
        print(f"Summary:      {outputs.summary_file}")
        return 0
    count = build_release_manifest(
        queue_manifest=args.queue,
        records_manifest=args.records,
        gate_results_manifest=args.gate_results,
        output_path=args.output,
        released_at=args.released_at,
    )
    print(f"Released Manifest Entries: {count}")
    print(f"Manifest:                  {args.output.resolve()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
