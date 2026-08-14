"""
ECOsphere -- Model-Aware Decision-Region Cache Experiment
===========================================================

LAB EXPERIMENT ONLY.
Does not modify 08_master_pipeline.py or any production/demo file.

Hypothesis:
    If the current feature vector remains inside the same leaf region of
    every tree in the optimized Random Forest, the ensemble decision is
    mathematically unchanged. We can therefore reuse the previous result
    without executing the forest again.

This experiment evaluates:
    1) decision parity against full optimized inference
    2) RF inference bypass rate
    3) end-to-end pipeline latency including drift + fusion + cache
    4) sensitivity to temporal perturbation magnitude

IMPORTANT:
    The temporal streams are controlled simulations derived from the
    held-out sensor states in hazard_dataset_featured.csv. They are NOT
    real temporal sensor logs. Results must be reported as simulation
    evidence, not real-world temporal validation.
"""

from __future__ import annotations

import importlib.util
import math
import random
import statistics
import time
from collections import deque
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split


ROOT = Path(__file__).resolve().parent
DATASET = ROOT / "hazard_dataset_featured.csv"
MODEL = ROOT / "hazard_model_optimized.joblib"
MASTER = ROOT / "08_master_pipeline.py"

RANDOM_SEED = 42
TEST_SIZE = 0.20

# Keep the experiment reasonably fast and reproducible.
N_STREAMS = 5
READINGS_PER_STREAM = 1500
WARMUP_READINGS = 100
TIMING_REPETITIONS = 3

# Temporal perturbation levels expressed as fractions of the empirical
# sensor ranges. The experiment reports all levels rather than cherry-picking.
NOISE_LEVELS = [0.0025, 0.005, 0.01, 0.02]

FEATURE_COLUMNS = [
    "gas_ppm",
    "tilt_deg",
    "vibration_g",
    "gas_risk",
    "tilt_risk",
    "vibration_risk",
    "structural_risk",
    "joint_risk",
]
RAW_COLUMNS = ["gas_ppm", "tilt_deg", "vibration_g"]


def feature_frame(x):
    """Return one feature vector with the exact names used during training."""
    arr = np.asarray(x, dtype=float).reshape(1, -1)
    return pd.DataFrame(arr, columns=FEATURE_COLUMNS)


def load_master():
    if not MASTER.exists():
        raise FileNotFoundError(f"Missing {MASTER.name}")
    spec = importlib.util.spec_from_file_location("ecosystem_master_lab", MASTER)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


master = load_master()


def load_model():
    if not MODEL.exists():
        raise FileNotFoundError(f"Missing {MODEL.name}")
    return joblib.load(MODEL)


MODEL_OBJ = load_model()

if hasattr(MODEL_OBJ, "feature_names_in_"):
    model_feature_names = list(MODEL_OBJ.feature_names_in_)
    if model_feature_names != FEATURE_COLUMNS:
        raise ValueError(
            "Feature-column mismatch between experiment and trained model. "
            f"Expected {FEATURE_COLUMNS}, got {model_feature_names}"
        )


def leaf_bounds_for_tree(estimator):
    """
    Compute the axis-aligned hyperrectangle for every leaf of one decision
    tree. sklearn trees route samples through threshold comparisons; the
    resulting leaf region is therefore an intersection of feature bounds.
    """
    tree = estimator.tree_
    n_features = tree.n_features
    leaves = {}

    def visit(node, lo, hi):
        left = tree.children_left[node]
        right = tree.children_right[node]

        if left == right:
            leaves[node] = (lo.copy(), hi.copy())
            return

        feature = tree.feature[node]
        threshold = tree.threshold[node]

        left_hi = hi.copy()
        left_hi[feature] = min(left_hi[feature], threshold)
        visit(left, lo, left_hi)

        right_lo = lo.copy()
        right_lo[feature] = max(right_lo[feature], threshold)
        visit(right, right_lo, hi)

    visit(
        0,
        np.full(n_features, -np.inf, dtype=float),
        np.full(n_features, np.inf, dtype=float),
    )
    return leaves


LEAF_BOUNDS = [
    leaf_bounds_for_tree(estimator)
    for estimator in MODEL_OBJ.estimators_
]


def inside_previous_region(x, previous_leaf_ids):
    """Return True only when x remains in every previous tree leaf region."""
    if previous_leaf_ids is None:
        return False

    for tree_index, leaf_id in enumerate(previous_leaf_ids):
        lo, hi = LEAF_BOUNDS[tree_index][leaf_id]
        if np.any(x < lo) or np.any(x > hi):
            return False
    return True


def current_leaf_ids(x):
    """Only called on a cache miss; stores the new exact decision region."""
    row = np.asarray(x, dtype=float).reshape(1, -1)
    return tuple(
        int(estimator.apply(row)[0])
        for estimator in MODEL_OBJ.estimators_
    )


def held_out_raw_states():
    df = pd.read_csv(DATASET)

    X = df[FEATURE_COLUMNS]
    y = df["label"]

    _, test_df, _, _ = train_test_split(
        df,
        y,
        test_size=TEST_SIZE,
        random_state=RANDOM_SEED,
        stratify=y,
    )

    return test_df[RAW_COLUMNS].to_numpy(dtype=float)


def make_temporal_stream(test_states, seed, noise_fraction):
    """
    Create a controlled temporal stream from held-out sensor states.

    Each anchor represents a physical state that persists for a short dwell
    period. Small perturbations emulate sensor noise / gradual movement.
    Anchors are selected across all three classes but labels are NOT used
    during inference or cache decisions.
    """
    rng = np.random.default_rng(seed)

    per_class = {}
    test_df = pd.read_csv(DATASET)
    _, test_part, _, _ = train_test_split(
        test_df,
        test_df["label"],
        test_size=TEST_SIZE,
        random_state=RANDOM_SEED,
        stratify=test_df["label"],
    )

    for label in sorted(test_part["label"].unique()):
        per_class[label] = test_part[
            test_part["label"] == label
        ][RAW_COLUMNS].to_numpy(dtype=float)

    labels = sorted(per_class)
    anchors = []

    # Balanced class sequence so the simulation does not become "mostly SAFE".
    n_anchors = math.ceil(READINGS_PER_STREAM / 40)
    for i in range(n_anchors):
        label = labels[i % len(labels)]
        pool = per_class[label]
        anchors.append(pool[rng.integers(0, len(pool))])

    anchors = np.asarray(anchors)

    raw_test = test_part[RAW_COLUMNS].to_numpy(dtype=float)
    empirical_ranges = np.ptp(raw_test, axis=0)
    empirical_ranges = np.maximum(
        np.asarray(empirical_ranges, dtype=float).reshape(3),
        1e-9,
    )
    lower_bounds = np.asarray(raw_test.min(axis=0), dtype=float).reshape(3)
    upper_bounds = np.asarray(raw_test.max(axis=0), dtype=float).reshape(3)

    stream = []
    dwell = 40

    for anchor in anchors:
        for _ in range(dwell):
            anchor = np.asarray(anchor, dtype=float).reshape(3)
            reading = anchor + rng.normal(
                0.0,
                empirical_ranges * noise_fraction,
                size=3,
            )
            # Keep within the empirical sensor envelope.
            reading = np.minimum(
                np.maximum(
                    np.asarray(reading, dtype=float).reshape(3),
                    lower_bounds,
                ),
                upper_bounds,
            )
            stream.append(reading)

    return np.asarray(stream[:READINGS_PER_STREAM])


class IndependentTrackers:
    """Same rolling-median drift correction used by the production pipeline."""

    def __init__(self):
        self.trackers = [
            master.BaselineDriftTracker(),
            master.BaselineDriftTracker(),
            master.BaselineDriftTracker(),
        ]

    def correct(self, gas, tilt, vibration):
        return tuple(
            tracker.update_and_correct(value)
            for tracker, value in zip(
                self.trackers,
                (gas, tilt, vibration),
            )
        )


def make_features(raw, trackers):
    corrected = trackers.correct(*raw)
    # Reuse the REAL production fusion function.
    features = master.apply_fusion(*corrected)
    return np.asarray(
        [features[column] for column in FEATURE_COLUMNS],
        dtype=float,
    )


def full_reference_predictions(stream):
    """
    Full optimized pipeline reference. Every reading executes the RF.
    Returns predictions, exact feature vectors and leaf IDs.
    """
    trackers = IndependentTrackers()
    features = []
    for raw in stream:
        features.append(make_features(raw, trackers))

    X = np.vstack(features)
    frame = pd.DataFrame(X, columns=FEATURE_COLUMNS)

    predictions = MODEL_OBJ.predict(frame)
    X_array = frame.to_numpy(dtype=float)
    leaf_ids = np.column_stack(
        [estimator.apply(X_array) for estimator in MODEL_OBJ.estimators_]
    ).astype(int)

    return X, predictions, leaf_ids


def validate_cache(stream):
    """
    Validation pass: full reference prediction is computed for every sample,
    while the cache is evaluated independently against that reference.
    """
    X, reference_predictions, all_leaf_ids = full_reference_predictions(stream)

    previous_leaf_ids = None
    previous_prediction = None

    hits = 0
    misses = 0
    mismatches = 0
    false_safe = 0

    for i, x in enumerate(X):
        if inside_previous_region(x, previous_leaf_ids):
            hits += 1
            cached_prediction = previous_prediction
        else:
            misses += 1
            cached_prediction = reference_predictions[i]
            previous_leaf_ids = tuple(all_leaf_ids[i])
            previous_prediction = cached_prediction

        if cached_prediction != reference_predictions[i]:
            mismatches += 1

        if cached_prediction == "safe" and reference_predictions[i] != "safe":
            false_safe += 1

    return {
        "bypass_rate": hits / len(X),
        "miss_rate": misses / len(X),
        "mismatches": mismatches,
        "false_safe": false_safe,
        "n": len(X),
    }


def time_full(stream):
    """End-to-end optimized pipeline: drift + fusion + RF + confidence."""
    samples = []

    for _ in range(TIMING_REPETITIONS):
        trackers = IndependentTrackers()

        for raw in stream[:WARMUP_READINGS]:
            x = make_features(raw, trackers)
            row = feature_frame(x)
            MODEL_OBJ.predict(row)
            MODEL_OBJ.predict_proba(row)

        start = time.perf_counter()

        for raw in stream[WARMUP_READINGS:]:
            x = make_features(raw, trackers)
            row = feature_frame(x)
            MODEL_OBJ.predict(row)
            MODEL_OBJ.predict_proba(row)

        elapsed = time.perf_counter() - start
        samples.append(
            elapsed / (len(stream) - WARMUP_READINGS) * 1000
        )

    return samples


def time_cached(stream):
    """
    End-to-end adaptive path. The cache check happens AFTER the real drift
    correction and real feature fusion, exactly where classification begins.
    """
    samples = []
    bypass_rates = []

    for _ in range(TIMING_REPETITIONS):
        trackers = IndependentTrackers()
        previous_leaf_ids = None
        previous_prediction = None
        previous_confidence = None
        hits = 0
        timed_count = 0

        for raw in stream[:WARMUP_READINGS]:
            x = make_features(raw, trackers)
            row = feature_frame(x)
            prediction = MODEL_OBJ.predict(row)[0]
            confidence = MODEL_OBJ.predict_proba(row)[0]
            previous_leaf_ids = current_leaf_ids(x)
            previous_prediction = prediction
            previous_confidence = confidence

        start = time.perf_counter()

        for raw in stream[WARMUP_READINGS:]:
            x = make_features(raw, trackers)

            if inside_previous_region(x, previous_leaf_ids):
                prediction = previous_prediction
                confidence = previous_confidence
                hits += 1
            else:
                row = feature_frame(x)
                prediction = MODEL_OBJ.predict(row)[0]
                confidence = MODEL_OBJ.predict_proba(row)[0]
                previous_leaf_ids = current_leaf_ids(x)
                previous_prediction = prediction
                previous_confidence = confidence

            timed_count += 1

        elapsed = time.perf_counter() - start
        samples.append(elapsed / timed_count * 1000)
        bypass_rates.append(hits / timed_count)

    return samples, bypass_rates


def percentile(values, p):
    values = sorted(values)
    if len(values) == 1:
        return values[0]
    position = (len(values) - 1) * p
    lo = int(position)
    hi = min(lo + 1, len(values) - 1)
    frac = position - lo
    return values[lo] + frac * (values[hi] - values[lo])


def summarize(values):
    return {
        "mean": statistics.mean(values),
        "median": statistics.median(values),
        "stdev": statistics.stdev(values) if len(values) > 1 else 0.0,
        "p95": percentile(values, 0.95),
        "runs": values,
    }


def main():
    print("=" * 78)
    print("ECOsphere -- Model-Aware Decision-Region Cache Experiment")
    print("=" * 78)
    print(f"Dataset: {DATASET.name}")
    print(f"Streams: {N_STREAMS} x {READINGS_PER_STREAM} readings")
    print(f"Timing repetitions: {TIMING_REPETITIONS}")
    print(f"Warmup readings/repetition: {WARMUP_READINGS}")
    print("Timing target: x86 host; NOT Arm/QRB2210.")
    print()

    test_states = held_out_raw_states()

    # Confirm the optimized model's architecture.
    print(
        f"Optimized model: {MODEL_OBJ.n_estimators} trees, "
        f"max_depth={MODEL_OBJ.max_depth}"
    )
    print(
        f"Total leaf regions indexed: "
        f"{sum(len(bounds) for bounds in LEAF_BOUNDS)}"
    )
    print()

    for noise_fraction in NOISE_LEVELS:
        validation = []
        all_streams = []

        print(f"--- Temporal perturbation: {noise_fraction:.2%} of sensor range ---")

        for stream_id in range(N_STREAMS):
            stream = make_temporal_stream(
                test_states,
                seed=RANDOM_SEED + stream_id,
                noise_fraction=noise_fraction,
            )
            all_streams.append(stream)

            result = validate_cache(stream)
            validation.append(result)

        bypass = [r["bypass_rate"] for r in validation]
        mismatches = sum(r["mismatches"] for r in validation)
        false_safe = sum(r["false_safe"] for r in validation)

        print(
            f"  bypass rate: mean={statistics.mean(bypass):.2%}, "
            f"min={min(bypass):.2%}, max={max(bypass):.2%}"
        )
        print(f"  decision mismatches: {mismatches}")
        print(f"  false-safe decisions: {false_safe}")

        # Timing one representative stream at each perturbation level.
        full_runs = time_full(all_streams[0])
        cache_runs, cache_bypass_runs = time_cached(all_streams[0])

        full_stats = summarize(full_runs)
        cache_stats = summarize(cache_runs)

        speedup = full_stats["median"] / cache_stats["median"]
        latency_reduction = (
            1.0 - cache_stats["median"] / full_stats["median"]
        ) * 100.0

        print(
            f"  full median:  {full_stats['median']:.4f} ms/call"
        )
        print(
            f"  cached median: {cache_stats['median']:.4f} ms/call"
        )
        print(
            f"  end-to-end speedup: {speedup:.2f}x "
            f"({latency_reduction:.1f}% reduction)"
        )
        print()

    print("--- Interpretation ---")
    print(
        "Forest-level model calls use the trained feature names; individual "
        "tree-level apply() calls use the nameless NumPy representation they "
        "were fitted with, avoiding sklearn feature-name mismatch warnings."
    )
    print(
        "A cache hit is mathematically safe with respect to the optimized "
        "Random Forest because the feature vector remains inside the same "
        "leaf region of every tree; therefore every tree reaches the same "
        "leaf and the ensemble prediction is unchanged."
    )
    print(
        "Temporal results are controlled simulations derived from held-out "
        "sensor states. They are feasibility evidence, not real-world "
        "temporal validation."
    )
    print(
        "This script is experimental only. It does not modify the production "
        "pipeline, model, benchmark script, or dashboard."
    )


if __name__ == "__main__":
    main()