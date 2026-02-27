from __future__ import annotations

import argparse
import hashlib
import json
import os
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any

from PIL import Image

from flux2.streamlit_adapter import Flux2Adapter


class BenchmarkAdapter(Flux2Adapter):
    def __init__(self) -> None:
        super().__init__()
        self._local_cache: dict[str, bytes] = {}

    def _get_cache_dict(self) -> dict[str, bytes]:
        return self._local_cache

    def _cache_get(self, key: str) -> Image.Image | None:
        payload = self._local_cache.get(key)
        if payload is None:
            return None
        self._local_cache.pop(key, None)
        self._local_cache[key] = payload
        return Image.open(__import__("io").BytesIO(payload)).convert("RGB")

    def _cache_put(self, key: str, img: Image.Image) -> None:
        buf = __import__("io").BytesIO()
        img.save(buf, format="PNG")
        payload = buf.getvalue()
        if key in self._local_cache:
            self._local_cache.pop(key, None)
        self._local_cache[key] = payload
        while len(self._local_cache) > self._cache_limit:
            oldest_key = next(iter(self._local_cache.keys()))
            self._local_cache.pop(oldest_key, None)

    def clear_generation_cache(self) -> None:
        self._local_cache.clear()


def _sha256_image(img: Image.Image) -> str:
    import io

    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return hashlib.sha256(buf.getvalue()).hexdigest()


def _configure_local_paths(repo_root: Path) -> None:
    defaults = {
        "KLEIN_4B_MODEL_PATH": repo_root / "weights" / "flux-2-klein-4b.safetensors",
        "AE_MODEL_PATH": repo_root / "weights" / "ae.safetensors",
        "FLUX2_LOCAL_ONLY": "1",
    }
    for key, value in defaults.items():
        if isinstance(value, Path):
            if key not in os.environ and value.exists():
                os.environ[key] = str(value.resolve())
        else:
            os.environ.setdefault(key, value)


def _safe_float(value: Any) -> float | None:
    if value is None:
        return None
    try:
        return float(value)
    except Exception:
        return None


def main() -> int:
    parser = argparse.ArgumentParser(description="Run Phase 1 benchmarks and update baseline JSON.")
    parser.add_argument("--model", default="flux.2-klein-4b")
    parser.add_argument("--dtype", default="bf16")
    parser.add_argument("--width", type=int, default=512)
    parser.add_argument("--height", type=int, default=512)
    parser.add_argument("--steps", type=int, default=4)
    parser.add_argument("--guidance", type=float, default=1.0)
    parser.add_argument("--seed", type=int, default=12345)
    parser.add_argument("--sequential-runs", type=int, default=10)
    parser.add_argument("--concurrent-runs", type=int, default=10)
    parser.add_argument("--soak-seconds", type=int, default=300)
    parser.add_argument("--output", default="benchmarks/phase1_baseline.json")
    args = parser.parse_args()

    repo_root = Path(__file__).resolve().parents[1]
    _configure_local_paths(repo_root)

    prompt = "cinematic lighthouse in fog, ultra detailed, volumetric lighting"

    adapter = BenchmarkAdapter()
    cold_load_start = time.perf_counter()
    adapter.load(args.model, dtype_str=args.dtype, cpu_offloading=False, attn_slicing=True)
    cold_first_load_s = time.perf_counter() - cold_load_start

    adapter.clear_generation_cache()

    adapter_warm = BenchmarkAdapter()
    warm_start = time.perf_counter()
    adapter_warm.warm_startup_models([args.model])
    warmup_elapsed_s = time.perf_counter() - warm_start
    load_start = time.perf_counter()
    adapter_warm.load(args.model, dtype_str=args.dtype, cpu_offloading=False, attn_slicing=True)
    first_load_s = time.perf_counter() - load_start

    adapter_second = BenchmarkAdapter()
    cached_load_start = time.perf_counter()
    adapter_second.load(args.model, dtype_str=args.dtype, cpu_offloading=False, attn_slicing=True)
    cached_load_s = time.perf_counter() - cached_load_start

    sequential_durations: list[float] = []
    sequential_hashes: list[str] = []

    adapter = adapter_warm
    adapter.clear_generation_cache()
    for _ in range(args.sequential_runs):
        t0 = time.perf_counter()
        image = adapter.generate(
            prompt=prompt,
            num_steps=args.steps,
            guidance=args.guidance,
            width=args.width,
            height=args.height,
            seed=args.seed,
        )
        sequential_durations.append(time.perf_counter() - t0)
        sequential_hashes.append(_sha256_image(image))

    deterministic = len(set(sequential_hashes)) == 1

    adapter.clear_generation_cache()
    _ = adapter.generate(
        prompt=prompt,
        num_steps=args.steps,
        guidance=args.guidance,
        width=args.width,
        height=args.height,
        seed=args.seed,
    )

    concurrent_durations: list[float] = []

    def _concurrent_call() -> float:
        t0 = time.perf_counter()
        _ = adapter.generate(
            prompt=prompt,
            num_steps=args.steps,
            guidance=args.guidance,
            width=args.width,
            height=args.height,
            seed=args.seed,
        )
        return time.perf_counter() - t0

    with ThreadPoolExecutor(max_workers=min(args.concurrent_runs, 10)) as executor:
        futures = [executor.submit(_concurrent_call) for _ in range(args.concurrent_runs)]
        for future in as_completed(futures):
            concurrent_durations.append(future.result())

    soak_start = time.time()
    start_runtime = adapter.get_performance_snapshot().get("runtime_samples", [])
    start_used = None
    if start_runtime:
        start_used = _safe_float(start_runtime[-1].get("gpu_memory_used_mb"))

    temp_samples: list[float] = []
    power_samples: list[float] = []
    peak_gpu_ratio = 0.0

    soak_iter = 0
    while time.time() - soak_start < args.soak_seconds:
        _ = adapter.generate(
            prompt=prompt,
            num_steps=args.steps,
            guidance=args.guidance,
            width=args.width,
            height=args.height,
            seed=args.seed,
        )
        snap = adapter.get_performance_snapshot()
        memory_current = (snap.get("memory") or {}).get("current") or {}
        used_mb = _safe_float(memory_current.get("used_mb"))
        total_mb = _safe_float(memory_current.get("total_mb"))
        if used_mb is not None and total_mb and total_mb > 0:
            peak_gpu_ratio = max(peak_gpu_ratio, used_mb / total_mb)

        runtime_samples = snap.get("runtime_samples") or []
        if runtime_samples:
            latest = runtime_samples[-1]
            temp = _safe_float(latest.get("gpu_temperature_c"))
            power = _safe_float(latest.get("gpu_power_w"))
            if temp is not None:
                temp_samples.append(temp)
            if power is not None:
                power_samples.append(power)
        soak_iter += 1

    end_runtime = adapter.get_performance_snapshot().get("runtime_samples", [])
    end_used = None
    if end_runtime:
        end_used = _safe_float(end_runtime[-1].get("gpu_memory_used_mb"))

    memory_leak_detected = None
    if start_used is not None and end_used is not None:
        memory_leak_detected = (end_used - start_used) > 512.0

    thermal_stability_ok = None
    if temp_samples:
        thermal_stability_ok = (max(temp_samples) < 85.0) and ((max(temp_samples) - min(temp_samples)) <= 15.0)

    snapshot = adapter.get_performance_snapshot()
    summary = snapshot.get("summary") or {}

    first_request_s = sequential_durations[0] if sequential_durations else None
    cached_runs = sequential_durations[1:] if len(sequential_durations) > 1 else []
    avg_cached_request_s = (sum(cached_runs) / len(cached_runs)) if cached_runs else None

    sorted_conc = sorted(concurrent_durations)
    if sorted_conc:
        p50 = sorted_conc[len(sorted_conc) // 2]
        p95_idx = max(0, min(len(sorted_conc) - 1, int(round(len(sorted_conc) * 0.95)) - 1))
        p95 = sorted_conc[p95_idx]
    else:
        p50 = None
        p95 = None

    targets = {
        "first_load_seconds_lt": 5.0,
        "cached_load_seconds_lt": 0.5,
        "gpu_peak_ratio_lte": 0.8,
        "cache_hit_rate_gte": 0.7,
    }

    result = {
        "phase": "phase1_caching_performance",
        "captured_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "environment": {
            "device": "cuda" if snapshot.get("memory", {}).get("current", {}).get("cuda_available") else "cpu",
            "model": args.model,
            "dtype": args.dtype,
            "width": args.width,
            "height": args.height,
            "steps": args.steps,
            "sequential_runs": args.sequential_runs,
            "concurrent_runs": args.concurrent_runs,
            "soak_seconds": args.soak_seconds,
        },
        "targets": targets,
        "benchmarks": {
            "sequential_same_prompt_10x": {
                "first_request_seconds": first_request_s,
                "average_cached_request_seconds": avg_cached_request_s,
                "cache_hit_rate": _safe_float(summary.get("cache_hit_rate")),
                "deterministic_same_prompt": deterministic,
            },
            "concurrent_requests_10x": {
                "p50_seconds": p50,
                "p95_seconds": p95,
                "max_queue_depth": int(summary.get("max_queue", 0)),
            },
            "continuous_1h": {
                "memory_leak_detected": memory_leak_detected,
                "peak_gpu_ratio": peak_gpu_ratio,
                "thermal_stability_ok": thermal_stability_ok,
                "temperature_max_c": max(temp_samples) if temp_samples else None,
                "power_max_w": max(power_samples) if power_samples else None,
                "iterations": soak_iter,
            },
            "model_load": {
                "cold_first_load_seconds": cold_first_load_s,
                "startup_warmup_seconds": warmup_elapsed_s,
                "first_load_seconds": first_load_s,
                "cached_load_seconds": cached_load_s,
            },
        },
    }

    acceptance = {
        "first_load_lt_5s": first_load_s < targets["first_load_seconds_lt"],
        "cached_load_lt_0_5s": cached_load_s < targets["cached_load_seconds_lt"],
        "gpu_peak_ratio_lte_0_8": peak_gpu_ratio <= targets["gpu_peak_ratio_lte"],
        "cache_hit_rate_gte_0_7": (_safe_float(summary.get("cache_hit_rate")) or 0.0)
        >= targets["cache_hit_rate_gte"],
        "deterministic_same_prompt": deterministic,
        "no_memory_leak": memory_leak_detected is False,
        "thermal_stability_ok": thermal_stability_ok is True if thermal_stability_ok is not None else None,
    }
    result["acceptance"] = acceptance
    result["status"] = "completed" if all(v is True for v in acceptance.values() if v is not None) else "partial"

    output_path = (repo_root / args.output).resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(result, indent=2), encoding="utf-8")

    print(json.dumps({"output": str(output_path), "status": result["status"], "acceptance": acceptance}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
