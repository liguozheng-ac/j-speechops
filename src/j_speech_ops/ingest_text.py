"""CLI for deterministic Stage 5 plain-text ingestion."""

from __future__ import annotations

import argparse
from pathlib import Path

from .schemas import RightsStatus
from .text_ingestion import TextIngestionConfig, ingest_text_to_manifest
from .tts_text import TTSIntendedUse, TextSourceType


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Ingest one UTF-8 text line per canonical TTS text sample"
    )
    parser.add_argument("--input", required=True, type=Path)
    parser.add_argument("--dataset-id", required=True)
    parser.add_argument("--language", default="ja")
    parser.add_argument("--locale", default="ja-JP")
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("data/manifests/tts_text_samples.jsonl"),
    )
    parser.add_argument("--source-name")
    parser.add_argument(
        "--source-type",
        choices=[item.value for item in TextSourceType],
        default=TextSourceType.FILE.value,
    )
    parser.add_argument(
        "--rights-status",
        choices=[item.value for item in RightsStatus],
        default=RightsStatus.UNKNOWN.value,
    )
    parser.add_argument(
        "--intended-use",
        choices=[item.value for item in TTSIntendedUse],
        default=TTSIntendedUse.TTS_PRODUCTION.value,
    )
    parser.add_argument("--domain")
    parser.add_argument("--id-prefix")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    config = TextIngestionConfig(
        dataset_id=args.dataset_id,
        language=args.language,
        locale=args.locale,
        source_name=args.source_name or args.input.name,
        source_type=args.source_type,
        rights_status=args.rights_status,
        intended_use=args.intended_use,
        domain=args.domain,
        id_prefix=args.id_prefix,
    )
    result = ingest_text_to_manifest(args.input, args.output, config)
    print(f"Input:    {result.input_path}")
    print(f"Output:   {result.output_manifest}")
    print(f"Samples:  {result.sample_count}")
    print(f"First ID: {result.samples[0].sample_id}")
    print(f"Last ID:  {result.samples[-1].sample_id}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
