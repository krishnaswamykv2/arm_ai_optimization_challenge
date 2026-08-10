"""
STEP 4: Verify the compressed RF + NumPy preprocessing optimization
against the original baseline, on the COMPLETE pipeline (drift
correction -> fusion -> classification -> output), not just the model
in isolation.

Compares FOUR configurations so the two changes (model compression,
preprocessing) can be told apart instead of only reporting one combined
number:
  1. baseline model   + pandas DataFrame   (current production code, as-shipped)
  2. baseline model   + NumPy array        (isolates the preprocessing effect)
  3. optimized model  + pandas DataFrame   (isolates the compression effect)
  4. optimized model  + NumPy array        (the proposed deployment config)

Everything here is measured on THIS machine (x86), not the QRB2210.
That distinction is preserved in the printed output -- do not report
these latency numbers as ARM/on-device numbers.

Usage:
    python3 11_verify_benchmarks.py
"""

import os
import random
import statistics
import time
import warnings
import joblib
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score

from importlib import import_module
master = import_module("08_master_pipeline")

warnings.filterwarnings("ignore", message="X does not have valid feature names")

BASELINE_MODEL_PATH = "hazard_model.joblib"
OPTIMIZED_MODEL_PATH = "hazard_model_optimized.joblib"
FEATURE_COLUMNS = master.FEATURE_COLUMN_ORDER
N_LATENCY_CALLS = 3000
N_WARMUP_CALLS = 200
N_LATENCY_REPETITIONS = 5
RANDOM_SEED = 42


# ---------------------------------------------------------------
# 1. MODEL SIZE
# ---------------------------------------------------------------
def compare_size():
    baseline_kb = os.path.getsize(BASELINE_MODEL_PATH) / 1024
    optimized_kb = os.path.getsize(OPTIMIZED_MODEL_PATH) / 1024
    reduction_pct = (1 - optimized_kb / baseline_kb) * 100
    return baseline_kb, optimized_kb, reduction_pct


# ---------------------------------------------------------------
# 2. ACCURACY / F1 -- classification layer only, on the held-out
#    test set, using the SAME split 03_train_model.py used. This is
#    deliberately NOT run through drift correction: drift correction
#    changes feature values relative to ground truth, which would
#    make this an accuracy measurement of drift correction, not of
#    the two models being compared.
# ---------------------------------------------------------------
def compare_accuracy():
    df = pd.read_csv("hazard_dataset_featured.csv")
    X = df[FEATURE_COLUMNS]
    y = df["label"]
    _, X_test, _, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    baseline_model = joblib.load(BASELINE_MODEL_PATH)
    optimized_model = joblib.load(OPTIMIZED_MODEL_PATH)

    results = {}
    for name, model in [("baseline", baseline_model), ("optimized", optimized_model)]:
        pred = model.predict(X_test)
        results[name] = {
            "accuracy": accuracy_score(y_test, pred),
            "macro_f1": f1_score(y_test, pred, average="macro"),
            "n_test": len(y_test),
        }
    return results


# ---------------------------------------------------------------
# 3. LATENCY -- complete pipeline (drift correction + fusion +
#    classification + output), repeated calls, mean per-call time.
#    Each config gets its OWN pipeline instance (own model, own
#    drift trackers) via master.build_pipeline() so no state leaks
#    between configs.
# ---------------------------------------------------------------
def make_reading_sequence(n):
    """Deterministic, repeatable sequence of plausible sensor readings
    (not random, so runs are comparable call-to-call)."""
    base = [
        (75, 2, 0.04), (480, 3, 0.05), (90, 20, 0.45), (700, 27, 0.7),
        (120, 5, 0.10), (300, 8, 0.20), (60, 1, 0.02), (650, 25, 0.65),
    ]
    return [base[i % len(base)] for i in range(n)]


def time_pipeline(model_path, use_numpy, n_calls, n_warmup):
    """Time one independent benchmark repetition.

    Returns mean milliseconds per complete pipeline call. A fresh pipeline
    instance is created for every repetition so drift-tracker state cannot
    leak between repetitions or configurations.
    """
    run = master.build_pipeline(model_path=model_path, use_numpy=use_numpy)
    readings = make_reading_sequence(n_calls + n_warmup)

    for gas, tilt, vib in readings[:n_warmup]:
        run(gas, tilt, vib)

    timed_readings = readings[n_warmup:]
    start = time.perf_counter()
    for gas, tilt, vib in timed_readings:
        run(gas, tilt, vib)
    elapsed = time.perf_counter() - start

    return (elapsed / len(timed_readings)) * 1000


def percentile(values, p):
    """Linear-interpolated percentile for a non-empty list of numbers."""
    ordered = sorted(values)
    if len(ordered) == 1:
        return ordered[0]
    position = (len(ordered) - 1) * p
    lower = int(position)
    upper = min(lower + 1, len(ordered) - 1)
    fraction = position - lower
    return ordered[lower] + (ordered[upper] - ordered[lower]) * fraction


def summarize(values):
    """Summarize repetition-level mean latencies."""
    return {
        "mean": statistics.mean(values),
        "median": statistics.median(values),
        "stdev": statistics.stdev(values) if len(values) > 1 else 0.0,
        "p95": percentile(values, 0.95),
        "runs": values,
    }


def compare_latency():
    configs = [
        ("baseline model + DataFrame  (current production code)", BASELINE_MODEL_PATH, False),
        ("baseline model + NumPy      (preprocessing only)", BASELINE_MODEL_PATH, True),
        ("optimized model + DataFrame (compression only)", OPTIMIZED_MODEL_PATH, False),
        ("optimized model + NumPy     (proposed deployment config)", OPTIMIZED_MODEL_PATH, True),
    ]

    # Randomize configuration order independently for each repetition so one
    # configuration is not systematically penalized by being first/last.
    rng = random.Random(RANDOM_SEED)
    samples = {label: [] for label, _, _ in configs}

    for repetition in range(1, N_LATENCY_REPETITIONS + 1):
        order = configs[:]
        rng.shuffle(order)
        print(f"  latency repetition {repetition}/{N_LATENCY_REPETITIONS}...",
              flush=True)

        for label, model_path, use_numpy in order:
            ms = time_pipeline(
                model_path, use_numpy, N_LATENCY_CALLS, N_WARMUP_CALLS
            )
            samples[label].append(ms)

    return {label: summarize(values) for label, values in samples.items()}


# ---------------------------------------------------------------
# REPORT
# ---------------------------------------------------------------
def main():
    print("=" * 78)
    print("ECOsphere -- Baseline vs Optimized Verification Benchmark")
    print(f"Machine: x86 (this environment), NOT the QRB2210 target hardware.")
    print(f"Latency: mean of {N_LATENCY_CALLS} complete-pipeline calls "
          f"({N_WARMUP_CALLS} warmup calls excluded).")
    print("=" * 78)

    print("\n--- 1. Model size on disk ---")
    baseline_kb, optimized_kb, size_reduction_pct = compare_size()
    print(f"  baseline (hazard_model.joblib)          : {baseline_kb:8.2f} KB")
    print(f"  optimized (hazard_model_optimized.joblib): {optimized_kb:8.2f} KB")
    print(f"  reduction: {size_reduction_pct:.1f}%  "
          f"({baseline_kb / optimized_kb:.2f}x smaller)")

    print("\n--- 2. Accuracy / macro F1 (held-out test set, classification layer only) ---")
    acc = compare_accuracy()
    for name in ("baseline", "optimized"):
        r = acc[name]
        print(f"  {name:9s}: accuracy={r['accuracy']*100:.2f}%  "
              f"macro_F1={r['macro_f1']:.4f}  (n={r['n_test']})")
    delta = acc["optimized"]["accuracy"] - acc["baseline"]["accuracy"]
    if delta >= 0:
        print(f"  -> no measurable accuracy degradation under the current evaluation "
              f"(delta={delta*100:+.2f}pp). Dataset is synthetic; treat this as a "
              f"held-out validation result, not proof of real-world parity.")
    else:
        print(f"  -> optimized model is {abs(delta)*100:.2f}pp LESS accurate on this "
              f"held-out set.")

    print("\n--- 3. Latency: complete pipeline, x86 ---")
    print(f"  {N_LATENCY_REPETITIONS} independent repetitions; "
          f"{N_LATENCY_CALLS} timed calls/repetition; "
          f"{N_WARMUP_CALLS} warmup calls/repetition.")
    print("  Statistics below describe the distribution of repetition-level "
          "mean ms/call values.")
    lat = compare_latency()

    for label, stats in lat.items():
        print(f"\n  {label}:")
        print(f"    mean   : {stats['mean']:8.3f} ms/call")
        print(f"    median : {stats['median']:8.3f} ms/call")
        print(f"    stdev  : {stats['stdev']:8.3f} ms/call")
        print(f"    p95    : {stats['p95']:8.3f} ms/call")
        print("    runs   : " + ", ".join(f"{x:.3f}" for x in stats["runs"]))

    baseline_ms = lat["baseline model + DataFrame  (current production code)"]["median"]
    final_ms = lat["optimized model + NumPy     (proposed deployment config)"]["median"]
    speedup = baseline_ms / final_ms
    reduction_pct = (1 - final_ms / baseline_ms) * 100
    print(f"\n  End-to-end (production baseline -> proposed optimized config), "
          f"using median repetition:")
    print(f"    {baseline_ms:.3f} ms -> {final_ms:.3f} ms "
          f"({speedup:.2f}x faster, {reduction_pct:.1f}% reduction) on x86.")
    print(f"    This is NOT ARM/QRB2210 latency -- run this same script's timing "
          f"pattern on-device to get that number.")

    print("\n--- Assumptions / limitations ---")
    print("  * Accuracy/F1 measured on a synthetic dataset's held-out split; not")
    print("    validated against real-world sensor data at scale.")
    print("  * Latency measured on x86 (this environment), not the QRB2210 target.")
    print("  * Latency figures are for the full per-reading pipeline (drift")
    print("    correction + fusion + classification + output formatting), matching")
    print("    what actually runs per sensor sample in deployment.")
    print("  * Each config used its own independent drift-tracker instance, so")
    print("    baseline and optimized timings are not contaminated by shared state.")

    print("\n--- Reproducibility ---")
    print("  * Same train/test split (random_state=42, stratify=y) as 03_train_model.py")
    print("    and 10_convert_to_tflite.py's representative dataset -- reruns will")
    print("    reproduce the same split.")
    print("  * Latency statistics summarize independent repetition-level means;")
    print("    system load can still affect x86 timing.")
    print(f"  * Configuration order is randomized per repetition with seed {RANDOM_SEED}.")
    print("  * Rerun with: python3 11_verify_benchmarks.py")


if __name__ == "__main__":
    main()