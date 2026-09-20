"""Isolated Qwen3-TTS worker used only by the .venv-tts subprocess."""

from __future__ import annotations

import argparse
import importlib.metadata
import json
import os
import sys
import time
import traceback
from pathlib import Path


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run one isolated Qwen3-TTS batch")
    parser.add_argument("--job", required=True, type=Path)
    parser.add_argument("--result", required=True, type=Path)
    return parser


def _is_fatal_runtime_error(exc: Exception) -> bool:
    rendered = f"{type(exc).__name__}: {exc}".casefold()
    markers = (
        "cuda error",
        "cuda out of memory",
        "device-side assert",
        "cublas",
        "cudnn",
        "driver",
    )
    return any(marker in rendered for marker in markers)


def run(job_path: Path, result_path: Path) -> int:
    try:
        import soundfile as sf
        import torch
        from qwen_tts import Qwen3TTSModel

        job = json.loads(job_path.read_text(encoding="utf-8"))
        if job["backend"] != "qwen3-tts":
            raise ValueError(f"unsupported backend: {job['backend']}")
        if job["device"] != "cuda:0":
            raise ValueError("Stage 8 baseline requires device cuda:0")
        if job["dtype"] != "bfloat16":
            raise ValueError("Stage 8 baseline requires bfloat16")
        if job["attention_implementation"] != "sdpa":
            raise ValueError("Stage 8 baseline requires SDPA")
        if job.get("instruct") is not None:
            raise ValueError("Stage 8 baseline does not permit style instructions")
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA is unavailable; CPU fallback is forbidden")
        if not torch.cuda.is_bf16_supported():
            raise RuntimeError("BF16 is unavailable; fallback is forbidden")

        device = torch.device("cuda:0")
        properties = torch.cuda.get_device_properties(device)
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats(device)
        torch.cuda.synchronize(device)
        load_started = time.perf_counter()
        model = Qwen3TTSModel.from_pretrained(
            job["model_path"],
            device_map="cuda:0",
            dtype=torch.bfloat16,
            attn_implementation="sdpa",
        )
        torch.cuda.synchronize(device)
        load_time = time.perf_counter() - load_started
        parameter_devices = {str(parameter.device) for parameter in model.model.parameters()}
        if not parameter_devices or any(
            not value.startswith("cuda") for value in parameter_devices
        ):
            raise RuntimeError(
                f"model parameters are not wholly on CUDA: {sorted(parameter_devices)}"
            )
        languages = [str(value) for value in (model.get_supported_languages() or [])]
        speakers = [str(value) for value in (model.get_supported_speakers() or [])]
        if job["language"].casefold() not in {value.casefold() for value in languages}:
            raise ValueError(f"unsupported language: {job['language']}")
        if job["speaker"].casefold() not in {value.casefold() for value in speakers}:
            raise ValueError(f"unsupported speaker: {job['speaker']}")

        results = []
        generation_parameters = dict(job["generation_parameters"])
        for request in job["requests"]:
            output_path = Path(request["output_audio_path"])
            temporary_path = output_path.with_suffix(".tmp.wav")
            try:
                output_path.parent.mkdir(parents=True, exist_ok=True)
                temporary_path.unlink(missing_ok=True)
                seed = int(request["seed"])
                torch.manual_seed(seed)
                torch.cuda.manual_seed_all(seed)
                torch.cuda.synchronize(device)
                started = time.perf_counter()
                wavs, sample_rate = model.generate_custom_voice(
                    text=request["synthesis_text"],
                    language=job["language"],
                    speaker=job["speaker"],
                    instruct=None,
                    non_streaming_mode=True,
                    **generation_parameters,
                )
                torch.cuda.synchronize(device)
                generation_time = time.perf_counter() - started
                if len(wavs) != 1:
                    raise RuntimeError(f"expected one waveform, received {len(wavs)}")
                sf.write(temporary_path, wavs[0], sample_rate, subtype="PCM_16")
                os.replace(temporary_path, output_path)
                results.append(
                    {
                        "run_id": request["run_id"],
                        "status": "success",
                        "generation_time_sec": generation_time,
                        "error_code": None,
                        "error_message": None,
                    }
                )
            except Exception as exc:
                temporary_path.unlink(missing_ok=True)
                if _is_fatal_runtime_error(exc) or not torch.cuda.is_available():
                    raise RuntimeError(
                        f"fatal runtime failure while processing {request['run_id']}: {exc}"
                    ) from exc
                results.append(
                    {
                        "run_id": request["run_id"],
                        "status": "failed",
                        "generation_time_sec": None,
                        "error_code": type(exc).__name__,
                        "error_message": str(exc),
                    }
                )

        result = {
            "schema_version": "1.0",
            "backend_version": importlib.metadata.version("qwen-tts"),
            "model_load_time_sec": load_time,
            "gpu_name": torch.cuda.get_device_name(device),
            "gpu_total_vram_bytes": properties.total_memory,
            "peak_allocated_vram_bytes": torch.cuda.max_memory_allocated(device),
            "peak_reserved_vram_bytes": torch.cuda.max_memory_reserved(device),
            "supported_languages": languages,
            "supported_speakers": speakers,
            "results": results,
        }
        result_path.parent.mkdir(parents=True, exist_ok=True)
        result_path.write_text(
            json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
        )
        return 0
    except Exception:
        traceback.print_exc(file=sys.stderr)
        return 2


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    return run(args.job, args.result)


if __name__ == "__main__":
    raise SystemExit(main())
