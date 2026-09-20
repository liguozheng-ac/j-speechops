"""JSONL IO for canonical TTSTextSample manifests."""

from __future__ import annotations

import json
from collections.abc import Iterable, Iterator
from os import PathLike
from pathlib import Path

from pydantic import ValidationError

from .tts_text import TTSTextSample


class TextManifestValidationError(ValueError):
    def __init__(
        self,
        path: Path,
        line_number: int,
        detail: str,
        sample_id: str | None = None,
    ) -> None:
        self.path = path
        self.line_number = line_number
        self.sample_id = sample_id
        self.detail = detail
        identity = f" [sample_id={sample_id}]" if sample_id is not None else ""
        super().__init__(f"{path}:{line_number}{identity}: {detail}")


def iter_text_manifest(path: str | PathLike[str]) -> Iterator[TTSTextSample]:
    manifest_path = Path(path)
    sample_ids: set[str] = set()
    with manifest_path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            try:
                payload = json.loads(line)
            except json.JSONDecodeError as exc:
                raise TextManifestValidationError(
                    manifest_path, line_number, f"malformed JSON: {exc.msg}"
                ) from exc
            candidate_id = (
                payload.get("sample_id") if isinstance(payload, dict) else None
            )
            sample_id = candidate_id if isinstance(candidate_id, str) else None
            try:
                sample = TTSTextSample.model_validate(payload)
            except ValidationError as exc:
                raise TextManifestValidationError(
                    manifest_path,
                    line_number,
                    f"invalid TTSTextSample: {exc}",
                    sample_id,
                ) from exc
            if sample.sample_id in sample_ids:
                raise TextManifestValidationError(
                    manifest_path,
                    line_number,
                    "duplicate sample_id",
                    sample.sample_id,
                )
            sample_ids.add(sample.sample_id)
            yield sample


def write_text_manifest(
    path: str | PathLike[str], samples: Iterable[TTSTextSample]
) -> int:
    manifest_path = Path(path)
    validated: list[TTSTextSample] = []
    sample_ids: set[str] = set()
    for item in samples:
        sample = TTSTextSample.model_validate(item)
        if sample.sample_id in sample_ids:
            raise ValueError(f"duplicate TTSTextSample sample_id: {sample.sample_id}")
        sample_ids.add(sample.sample_id)
        validated.append(sample)

    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    with manifest_path.open("w", encoding="utf-8", newline="\n") as handle:
        for sample in validated:
            handle.write(sample.model_dump_json())
            handle.write("\n")
    return len(validated)
