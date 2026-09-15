"""Measure float64 CPU/CUDA render time and sample error on a real profile library.

Includes loading, device transfer, chunking, and copying output back to NumPy.
Does not read or write the embedding index, and never downloads profiles.
"""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import sys
import time

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from tonehound.audioio import read
from tonehound.nam_render import DEFAULT_RENDER_CHUNK, load

ROOT = Path(__file__).resolve().parents[2]


def normalise(x: np.ndarray) -> np.ndarray:
    peak = float(np.max(np.abs(x)))
    return (x / peak * 0.95 if peak > 0 else x).astype(np.float32)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--profiles", type=Path, default=ROOT / ".cache/tone3000/profiles")
    ap.add_argument("--di", type=Path, default=ROOT / "assets/user_di/Djent DI.wav")
    ap.add_argument("--seconds", type=float, default=15.0)
    ap.add_argument("--chunk-samples", type=int, default=DEFAULT_RENDER_CHUNK)
    ap.add_argument("--limit", type=int)
    ap.add_argument("--out", type=Path, default=ROOT / "docs/nam_gpu_benchmark.json")
    args = ap.parse_args()
    if not torch.cuda.is_available():
        ap.error("CUDA is required for this comparison")
    if args.seconds <= 0 or args.chunk_samples <= 0 or (args.limit is not None and args.limit <= 0):
        ap.error("seconds, chunk-samples and limit must be positive")
    paths = sorted(args.profiles.glob("*.nam"))
    if args.limit:
        paths = paths[:args.limit]
    if not paths:
        ap.error(f"no .nam profiles in {args.profiles}")
    di = read(args.di, seconds=args.seconds)

    # Pay CUDA context/cuDNN initialisation once before measuring the library.
    warm = load(paths[0])
    warm.render(di[:8192], device="cuda", chunk_samples=args.chunk_samples)
    del warm
    torch.cuda.reset_peak_memory_stats()
    rows = []
    for i, path in enumerate(paths, 1):
        started = time.perf_counter()
        cpu = load(path).render(di)
        cpu_s = time.perf_counter() - started
        started = time.perf_counter()
        model = load(path)
        gpu = model.render(di, device="cuda", chunk_samples=args.chunk_samples)
        gpu_s = time.perf_counter() - started
        error = gpu - cpu
        row = {"profile": path.name, "architecture": model.architecture,
               "device": model.last_render_device, "note": model.last_render_note,
               "cpu_s": cpu_s, "cuda_s": gpu_s,
               "max_abs_error": float(np.max(np.abs(error))),
               "rms_error": float(np.sqrt(np.mean(error ** 2))),
               "normalised_float32_max_error": float(np.max(np.abs(
                   normalise(gpu) - normalise(cpu))))}
        rows.append(row)
        print(f"{i:3}/{len(paths)} CPU {cpu_s:.3f}s CUDA {gpu_s:.3f}s "
              f"error {row['max_abs_error']:.3e} {path.stem[:65]}", flush=True)

    cpu_s = sum(r["cpu_s"] for r in rows)
    cuda_s = sum(r["cuda_s"] for r in rows)
    report = {
        "measured_at": datetime.now(timezone.utc).isoformat(),
        "method": "load + render + transfers; warmed CUDA context; no embeddings",
        "torch": torch.__version__, "gpu": torch.cuda.get_device_name(0),
        "cpu_threads": torch.get_num_threads(), "dtype": "float64",
        "di": str(args.di), "samples": len(di), "chunk_samples": args.chunk_samples,
        "profiles": len(rows), "cuda_profiles": sum(r["device"] == "cuda" for r in rows),
        "cpu_s": cpu_s, "cuda_s": cuda_s, "speedup": cpu_s / cuda_s,
        "max_abs_error": max(r["max_abs_error"] for r in rows),
        "normalised_float32_max_error": max(r["normalised_float32_max_error"] for r in rows),
        "peak_cuda_allocated_mib": torch.cuda.max_memory_allocated() / 1024**2,
        "rows": rows}
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps({k: v for k, v in report.items() if k != "rows"}, indent=2))
    print(f"Wrote {args.out}")
    return 0 if report["max_abs_error"] < 1e-7 else 1


if __name__ == "__main__":
    raise SystemExit(main())
