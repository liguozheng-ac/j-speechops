"""Export JSON Schema from the Pydantic source of truth."""

from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from j_speech_ops.schemas import DatasetMetadata, SpeechSample  # noqa: E402
from j_speech_ops.tts_text import TTSTextSample  # noqa: E402


def main() -> None:
    schema_dir = ROOT / "schemas"
    schema_dir.mkdir(parents=True, exist_ok=True)
    exports = {
        "speech_sample.schema.json": SpeechSample.model_json_schema(),
        "dataset_metadata.schema.json": DatasetMetadata.model_json_schema(),
        "tts_text_sample.schema.json": TTSTextSample.model_json_schema(),
    }
    for filename, schema in exports.items():
        destination = schema_dir / filename
        destination.write_text(
            json.dumps(schema, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
        )
        print(destination.relative_to(ROOT))


if __name__ == "__main__":
    main()
