"""Read and write the canonical SpeechSample JSONL manifest."""

from __future__ import annotations

import json
from collections.abc import Iterable, Iterator
from os import PathLike
from pathlib import Path

from pydantic import ValidationError

from .schemas import SpeechSample


class ManifestValidationError(ValueError):
    """A malformed or schema-invalid JSONL record with source location."""

    def __init__(self, path: Path, line_number: int, detail: str) -> None:
        self.path = path
        self.line_number = line_number
        self.detail = detail
        super().__init__(f"{path}:{line_number}: {detail}")


def iter_manifest(path: str | PathLike[str]) -> Iterator[SpeechSample]:
    """Yield validated samples from a UTF-8 JSONL manifest."""
    manifest_path = Path(path)
    with manifest_path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            try:
                payload = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ManifestValidationError(
                    manifest_path, line_number, f"malformed JSON: {exc.msg}"
                ) from exc
            try:
                yield SpeechSample.model_validate(payload)
            except ValidationError as exc:
                raise ManifestValidationError(
                    manifest_path, line_number, f"invalid SpeechSample: {exc}"
                ) from exc


def write_manifest(
    path: str | PathLike[str], samples: Iterable[SpeechSample]
) -> int:
    """Write validated samples as one compact JSON object per line; return record count."""
    manifest_path = Path(path)
    count = 0
    with manifest_path.open("w", encoding="utf-8", newline="\n") as handle:
        for sample in samples:
            validated = SpeechSample.model_validate(sample)
            handle.write(validated.model_dump_json())
            handle.write("\n")
            count += 1
    return count

