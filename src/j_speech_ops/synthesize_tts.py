"""Core CLI for Stage 8 model-neutral TTS generation."""

from __future__ import annotations

import argparse
from pathlib import Path

from .tts_synthesis import (
    Qwen3TTSRuntimeAdapter,
    SynthesisConfig,
    TTSSynthesisPipeline,
)


PROJECT_ROOT = Path(__file__).resolve().parents[2]


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Generate validated Japanese WAVs from CURATED + PASS text"
    )
    parser.add_argument("--manifest", required=True, type=Path)
    parser.add_argument("--prep-report", required=True, type=Path)
    parser.add_argument("--curation-report", required=True, type=Path)
    parser.add_argument("--model-path", required=True, type=Path)
    parser.add_argument(
        "--output-dir", type=Path, default=Path("outputs/stage8_tts_generation")
    )
    parser.add_argument(
        "--tts-python",
        type=Path,
        default=PROJECT_ROOT / ".venv-tts" / "Scripts" / "python.exe",
    )
    parser.add_argument("--base-seed", type=int, default=20260921)
    parser.add_argument("--force", action="store_true")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    config = SynthesisConfig(
        model_path=str(args.model_path.resolve()), base_seed=args.base_seed
    )
    outputs = TTSSynthesisPipeline(
        output_root=args.output_dir,
        config=config,
        runtime_adapter=Qwen3TTSRuntimeAdapter(
            python_executable=args.tts_python,
            worker_script=PROJECT_ROOT / "runtime" / "qwen3_tts_worker.py",
        ),
    ).run(
        args.manifest,
        args.prep_report,
        args.curation_report,
        force=args.force,
    )
    summary = outputs.summary
    print(f"Total Samples:      {summary.total_samples}")
    print(f"Attempted Runs:     {summary.attempted_runs}")
    print(f"Successful Runs:    {summary.succeeded_runs}")
    print(f"Failed Runs:        {summary.failed_runs}")
    print(f"Skipped Runs:       {summary.skipped_runs}")
    print(f"Planning Failures:  {summary.planning_failed_runs}")
    print(f"Synthesis Runs:     {outputs.synthesis_runs_manifest}")
    print(f"Generated Audio:    {outputs.generated_audio_manifest}")
    print(f"Summary:            {outputs.summary_file}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
