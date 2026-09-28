"""Isolated JR tail/style experiment; never imported by the production pipeline."""

from __future__ import annotations

import argparse
import importlib.metadata
import json
import math
import time
from pathlib import Path

import numpy as np


TEXT = "ジェイアール新宿駅までご案内いたします。"
STYLE_INSTRUCTION = "ホテルスタッフのような自然で明るい丁寧な口調で話してください。"
SEED = 1492265953
GENERATION_PARAMETERS = {
    "do_sample": True,
    "top_k": 50,
    "top_p": 1.0,
    "temperature": 0.9,
    "repetition_penalty": 1.05,
    "subtalker_dosample": True,
    "subtalker_top_k": 50,
    "subtalker_top_p": 1.0,
    "subtalker_temperature": 0.9,
    "max_new_tokens": 8192,
}


def _tail_metrics(waveform: np.ndarray, sample_rate: int) -> dict:
    mono = np.max(np.abs(np.asarray(waveform, dtype=np.float64).reshape(-1, 1)), axis=1)
    threshold = 10.0 ** (-50.0 / 20.0)
    audible = np.flatnonzero(mono > threshold)
    last_audible = int(audible[-1]) if len(audible) else None
    trailing = len(mono) if last_audible is None else len(mono) - last_audible - 1
    tail = mono[-min(len(mono), sample_rate // 20) :]
    rms = float(np.sqrt(np.mean(np.square(tail, dtype=np.float64))))
    return {
        "frames": len(mono),
        "duration_sec": len(mono) / sample_rate,
        "last_non_silent_sample_index_at_minus_50_dbfs": last_audible,
        "trailing_silence_sec_at_minus_50_dbfs": trailing / sample_rate,
        "final_50ms_rms": rms,
        "final_50ms_rms_dbfs": 20.0 * math.log10(rms) if rms > 0 else None,
        "final_sample_amplitude": float(mono[-1]),
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run isolated JR default/style diagnostics")
    parser.add_argument("--model-path", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    import soundfile as sf
    import torch
    from qwen_tts import Qwen3TTSModel

    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is unavailable; diagnostic CPU fallback is forbidden")
    if not torch.cuda.is_bf16_supported():
        raise RuntimeError("BF16 is unavailable; diagnostic fallback is forbidden")
    device = torch.device("cuda:0")
    properties = torch.cuda.get_device_properties(device)
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats(device)
    load_started = time.perf_counter()
    model = Qwen3TTSModel.from_pretrained(
        str(args.model_path.resolve()),
        device_map="cuda:0",
        dtype=torch.bfloat16,
        attn_implementation="sdpa",
    )
    torch.cuda.synchronize(device)
    report = {
        "diagnostic_only": True,
        "qwen_tts_version": importlib.metadata.version("qwen-tts"),
        "torch_version": torch.__version__,
        "gpu": torch.cuda.get_device_name(device),
        "gpu_total_vram_bytes": properties.total_memory,
        "model_path": str(args.model_path.resolve()),
        "model_load_time_sec": time.perf_counter() - load_started,
        "text": TEXT,
        "speaker": "Ono_Anna",
        "language": "Japanese",
        "dtype": "bfloat16",
        "attention_implementation": "sdpa",
        "non_streaming_mode": True,
        "seed": SEED,
        "generation_parameters": GENERATION_PARAMETERS,
        "variants": [],
    }
    variants = (
        ("current_default", None),
        ("service_instruction", STYLE_INSTRUCTION),
    )
    for name, instruction in variants:
        torch.manual_seed(SEED)
        torch.cuda.manual_seed_all(SEED)
        torch.cuda.synchronize(device)
        started = time.perf_counter()
        wavs, sample_rate = model.generate_custom_voice(
            text=TEXT,
            language="Japanese",
            speaker="Ono_Anna",
            instruct=instruction,
            non_streaming_mode=True,
            **GENERATION_PARAMETERS,
        )
        torch.cuda.synchronize(device)
        generation_time = time.perf_counter() - started
        if len(wavs) != 1:
            raise RuntimeError(f"expected one waveform, received {len(wavs)}")
        raw = np.asarray(wavs[0]).reshape(-1)
        raw_path = output_dir / f"jr_{name}.raw.npy"
        wav_path = output_dir / f"jr_{name}.wav"
        np.save(raw_path, raw)
        sf.write(wav_path, raw, sample_rate, subtype="PCM_16")
        saved, saved_rate = sf.read(wav_path, dtype="float64", always_2d=False)
        saved = np.asarray(saved).reshape(-1)
        report["variants"].append(
            {
                "name": name,
                "instruction": instruction,
                "generation_time_sec": generation_time,
                "raw_waveform_path": str(raw_path),
                "saved_wav_path": str(wav_path),
                "raw_dtype": str(raw.dtype),
                "raw": _tail_metrics(raw, sample_rate),
                "saved": _tail_metrics(saved, saved_rate),
                "saved_sample_rate_matches": saved_rate == sample_rate,
                "saved_frame_count_matches_raw": len(saved) == len(raw),
                "frames_lost_during_save": len(raw) - len(saved),
            }
        )
    report["peak_allocated_vram_bytes"] = torch.cuda.max_memory_allocated(device)
    report["peak_reserved_vram_bytes"] = torch.cuda.max_memory_reserved(device)
    report_path = output_dir / "jr_experiment_report.json"
    report_path.write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
