"""
ECOsphere — Arm64 Validation Benchmark

Purpose
-------
Run the COMPLETE ECOsphere inference pipeline on a REAL Arm64 CPU.
This is intentionally separate from the x86 benchmark.

Four configurations:
1. baseline RF + pandas DataFrame
2. baseline RF + NumPy preprocessing
3. optimized RF + pandas DataFrame
4. optimized RF + NumPy preprocessing

The comparison isolates:
- model compression
- preprocessing representation
- combined deployment optimization

This script refuses to run on non-Arm64 machines so its output cannot
accidentally be presented as an Arm measurement.

It reports:
- CPU architecture
- Python / package versions
- model sizes
- latency distribution: mean / median / stdev / p95
- end-to-end speedup
- latency reduction

No dashboard code is imported. No cloud service is required.
"""

from __future__ import annotations

import json
import os
import platform
import random
import statistics
import sys
import time
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import sklearn

import importlib.util
import sys as _sys

ROOT = Path(__file__).resolve().parent
MASTER_PATH = ROOT / "08_master_pipeline.py"
BASELINE_MODEL = ROOT / "hazard_model.joblib"
OPTIMIZED_MODEL = ROOT / "hazard_model_optimized.joblib"

REPETITIONS = 10
WARMUP_CALLS = 100
TIMED_CALLS = 400
SEED = 42


def load_master():
    spec = importlib.util.spec_from_file_location("ecos_arm_master", MASTER_PATH)
    if spec is None or spec.loader is None:
        raise ImportError(f"Cannot load {MASTER_PATH}")
    module = importlib.util.module_from_spec(spec)
    _sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def reading_sequence(n):
    base = [
        (75.0, 2.0, 0.04),
        (480.0, 3.0, 0.05),
        (90.0, 20.0, 0.45),
        (700.0, 27.0, 0.70),
        (120.0, 5.0, 0.10),
        (300.0, 8.0, 0.20),
        (60.0, 1.0, 0.02),
        (650.0, 25.0, 0.65),
    ]
    return [base[i % len(base)] for i in range(n)]


def time_one(master, model_path, use_numpy, readings):
    run = master.build_pipeline(
        model_path=str(model_path),
        use_numpy=use_numpy,
    )

    for gas, tilt, vib in readings[:WARMUP_CALLS]:
        run(gas, tilt, vib)

    start = time.perf_counter_ns()
    for gas, tilt, vib in readings[WARMUP_CALLS:]:
        run(gas, tilt, vib)
    elapsed_ns = time.perf_counter_ns() - start

    return elapsed_ns / len(readings[WARMUP_CALLS:]) / 1_000_000.0


def p95(values):
    values = sorted(values)
    if len(values) == 1:
        return values[0]
    x = (len(values) - 1) * 0.95
    lo = int(x)
    hi = min(lo + 1, len(values) - 1)
    return values[lo] + (values[hi] - values[lo]) * (x - lo)


def summarize(values):
    return {
        "mean_ms": statistics.mean(values),
        "median_ms": statistics.median(values),
        "stdev_ms": statistics.stdev(values) if len(values) > 1 else 0.0,
        "p95_ms": p95(values),
        "repetitions": len(values),
        "timed_calls_per_rep": TIMED_CALLS,
    }


def main():
    arch = platform.machine().lower()
    if arch not in {"aarch64", "arm64"}:
        raise SystemExit(
            f"REFUSING TO RUN: detected architecture={platform.machine()!r}. "
            "This validator is only for a real Arm64 CPU."
        )

    for path in (MASTER_PATH, BASELINE_MODEL, OPTIMIZED_MODEL):
        if not path.exists():
            raise FileNotFoundError(f"Missing required file: {path.name}")

    # Avoid accidental reproducibility claims based on randomized ordering.
    random.seed(SEED)
    np.random.seed(SEED)

    master = load_master()
    readings = reading_sequence(WARMUP_CALLS + TIMED_CALLS)

    configs = [
        ("baseline_dataframe", BASELINE_MODEL, False),
        ("baseline_numpy", BASELINE_MODEL, True),
        ("optimized_dataframe", OPTIMIZED_MODEL, False),
        ("optimized_numpy", OPTIMIZED_MODEL, True),
    ]

    # Fixed randomized order per repetition reduces systematic thermal/order bias.
    samples = {name: [] for name, _, _ in configs}
    for rep in range(REPETITIONS):
        order = configs[:]
        random.Random(SEED + rep).shuffle(order)
        for name, model_path, use_numpy in order:
            value = time_one(master, model_path, use_numpy, readings)
            samples[name].append(value)

    summary = {name: summarize(values) for name, values in samples.items()}

    baseline = summary["baseline_numpy"]["median_ms"]
    optimized = summary["optimized_numpy"]["median_ms"]
    speedup = baseline / optimized
    reduction = (1.0 - optimized / baseline) * 100.0

    result = {
        "environment": {
            "architecture": platform.machine(),
            "platform": platform.platform(),
            "python": sys.version,
            "numpy": np.__version__,
            "pandas": pd.__version__,
            "scikit_learn": sklearn.__version__,
            "joblib": joblib.__version__,
            "cpu_count": os.cpu_count(),
        },
        "methodology": {
            "repetitions": REPETITIONS,
            "warmup_calls": WARMUP_CALLS,
            "timed_calls": TIMED_CALLS,
            "randomized_config_order": True,
            "pipeline_scope": "drift correction + fusion + classification + output",
        },
        "model_size": {
            "baseline_kb": BASELINE_MODEL.stat().st_size / 1024.0,
            "optimized_kb": OPTIMIZED_MODEL.stat().st_size / 1024.0,
            "reduction_percent": (
                1.0
                - OPTIMIZED_MODEL.stat().st_size / BASELINE_MODEL.stat().st_size
            ) * 100.0,
        },
        "latency": summary,
        "optimized_vs_baseline_numpy": {
            "median_speedup_x": speedup,
            "median_latency_reduction_percent": reduction,
        },
    }

    out = ROOT / "arm64_validation_results.json"
    out.write_text(json.dumps(result, indent=2), encoding="utf-8")

    print("\n=== ECOsphere REAL ARM64 VALIDATION ===")
    print(f"Architecture : {platform.machine()}")
    print(f"Platform     : {platform.platform()}")
    print(f"Python       : {platform.python_version()}")
    print(f"CPU count    : {os.cpu_count()}")
    print("\nLatency (ms/call):")
    for name in samples:
        s = summary[name]
        print(
            f"  {name:22s} "
            f"median={s['median_ms']:.3f} "
            f"mean={s['mean_ms']:.3f} "
            f"stdev={s['stdev_ms']:.3f} "
            f"p95={s['p95_ms']:.3f}"
        )

    print("\nOptimized NumPy vs baseline NumPy:")
    print(f"  Median speedup    : {speedup:.2f}x")
    print(f"  Median reduction  : {reduction:.1f}%")
    print("\nModel size:")
    print(f"  Baseline          : {result['model_size']['baseline_kb']:.2f} KB")
    print(f"  Optimized         : {result['model_size']['optimized_kb']:.2f} KB")
    print(f"  Reduction         : {result['model_size']['reduction_percent']:.1f}%")
    print(f"\nSaved machine-readable result: {out.name}")


if __name__ == "__main__":
    main()
