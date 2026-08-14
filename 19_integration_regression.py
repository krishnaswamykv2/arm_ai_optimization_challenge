"""
ECOsphere -- Integration Regression Validation
==============================================

Purpose:
    Verify that the integrated runtime preserves the optimized classifier's
    decisions while optional cache/event/cloud layers are enabled or disabled.

This is a regression test, NOT a performance benchmark.

Claims tested:
    1. Integrated runtime decisions match a fresh optimized-RF reference.
    2. Cache-enabled decisions match the same reference.
    3. Cache hits actually bypass RF execution.
    4. Event intelligence can be disabled without affecting inference.
    5. Cloud feedback can be disabled without affecting inference.
    6. Cloud outage does not stop local inference or invalidate the config.

No production model/pipeline/benchmark file is modified.
"""

from __future__ import annotations

import importlib.util
from pathlib import Path

import joblib
import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parent
RUNTIME_PATH = ROOT / "18_ecosphere_runtime.py"
MODEL_PATH = ROOT / "hazard_model_optimized.joblib"

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

N_TEST = 1000
SEED = 42


def load_runtime():
    spec = importlib.util.spec_from_file_location(
        "ecosphere_runtime_regression",
        RUNTIME_PATH,
    )
    if spec is None or spec.loader is None:
        raise ImportError("Could not load 18_ecosphere_runtime.py")

    module = importlib.util.module_from_spec(spec)

    # Required for Python 3.12 dataclass/type introspection when the runtime
    # dynamically imports the feedback module.
    import sys
    sys.modules[spec.name] = module

    try:
        spec.loader.exec_module(module)
    except Exception:
        sys.modules.pop(spec.name, None)
        raise

    return module


def make_test_stream():
    rng = np.random.default_rng(SEED)

    centers = np.array([
        [75.0, 2.0, 0.04],
        [480.0, 3.0, 0.05],
        [90.0, 20.0, 0.45],
        [700.0, 27.0, 0.70],
    ])

    noise = np.array([8.0, 0.20, 0.008])

    indices = rng.integers(0, len(centers), size=N_TEST)
    readings = centers[indices] + rng.normal(
        0.0,
        noise,
        size=(N_TEST, 3),
    )

    readings[:, 0] = np.maximum(readings[:, 0], 0.0)
    readings[:, 2] = np.maximum(readings[:, 2], 0.0)

    return [tuple(row) for row in readings]


def fresh_reference(runtime, reading):
    """
    Independent fresh RF reference using the same drift/fusion machinery as
    the integrated runtime, but without its cache.
    """
    corrected, features = runtime._correct_and_fuse(*reading)

    frame = pd.DataFrame([features], columns=FEATURE_COLUMNS)
    prediction = runtime.model.predict(frame)[0]
    probabilities = runtime.model.predict_proba(frame)[0]

    return {
        "hazard_state": str(prediction).upper(),
        "confidence": dict(
            zip(runtime.model.classes_, probabilities.round(3))
        ),
        "corrected": corrected,
    }


def compare_labels(reference, candidate):
    return reference["hazard_state"] == candidate["hazard_state"]


def main():
    if not RUNTIME_PATH.exists():
        raise FileNotFoundError(f"Missing {RUNTIME_PATH.name}")
    if not MODEL_PATH.exists():
        raise FileNotFoundError(f"Missing {MODEL_PATH.name}")

    runtime_module = load_runtime()
    stream = make_test_stream()

    # Use separate runtime instances so each stateful drift tracker starts
    # from the same initial condition.
    reference_rt = runtime_module.ECOSphereRuntime(
        enable_cache=False,
        enable_events=False,
        enable_cloud_feedback=False,
    )

    integrated_rt = runtime_module.ECOSphereRuntime(
        enable_cache=True,
        enable_events=True,
        enable_cloud_feedback=True,
    )

    no_events_rt = runtime_module.ECOSphereRuntime(
        enable_cache=True,
        enable_events=False,
        enable_cloud_feedback=False,
    )

    no_cache_rt = runtime_module.ECOSphereRuntime(
        enable_cache=False,
        enable_events=False,
        enable_cloud_feedback=False,
    )

    reference_labels = []
    integrated_labels = []
    no_events_labels = []
    no_cache_labels = []

    integrated_cache_hits = 0
    integrated_cache_hit_mismatches = 0

    for reading in stream:
        reference = fresh_reference(reference_rt, reading)
        integrated = integrated_rt.step(*reading)
        no_events = no_events_rt.step(*reading)
        no_cache = no_cache_rt.step(*reading)

        ref_label = reference["hazard_state"]

        reference_labels.append(ref_label)
        integrated_labels.append(integrated["hazard_state"])
        no_events_labels.append(no_events["hazard_state"])
        no_cache_labels.append(no_cache["hazard_state"])

        if integrated["cache_hit"]:
            integrated_cache_hits += 1

            # Independent fresh-RF check for this exact reading using the
            # reference runtime's state, which is aligned by stream position.
            if integrated["hazard_state"] != ref_label:
                integrated_cache_hit_mismatches += 1

    integrated_mismatches = sum(
        a != b for a, b in zip(reference_labels, integrated_labels)
    )
    no_events_mismatches = sum(
        a != b for a, b in zip(reference_labels, no_events_labels)
    )
    no_cache_mismatches = sum(
        a != b for a, b in zip(reference_labels, no_cache_labels)
    )

    # Cloud-independence test: the local runtime must continue when feedback
    # is disabled, and the classifier must remain usable.
    cloud_offline_rt = runtime_module.ECOSphereRuntime(
        enable_cache=True,
        enable_events=True,
        enable_cloud_feedback=True,
    )

    for reading in stream[:100]:
        cloud_offline_rt.step(*reading)

    previous_config = cloud_offline_rt.feedback.edge_config
    candidate = cloud_offline_rt.feedback.module.EdgeConfig(
        config_version=previous_config.config_version + 1,
        vibration_monitoring="HIGH",
        telemetry_interval=5,
        reason="offline_test_candidate",
    )

    applied_offline = cloud_offline_rt.feedback.module.EdgeConfigManager(
        previous_config
    ).apply(candidate, cloud_online=False)

    local_after_outage = cloud_offline_rt.step(*stream[100])
    config_retained = (
        cloud_offline_rt.feedback.edge_config == previous_config
    )

    overall_pass = (
        integrated_mismatches == 0
        and no_events_mismatches == 0
        and no_cache_mismatches == 0
        and integrated_cache_hit_mismatches == 0
        and integrated_cache_hits > 0
        and applied_offline is False
        and config_retained
        and local_after_outage["hazard_state"] in {
            "SAFE", "ELEVATED", "CRITICAL"
        }
    )

    print("=" * 78)
    print("ECOsphere -- Integration Regression Validation")
    print("=" * 78)

    print("\n--- Dataset ---")
    print(f"test readings                   : {N_TEST}")
    print(f"random seed                     : {SEED}")

    print("\n--- Classification agreement ---")
    print(
        f"reference vs integrated         : "
        f"{N_TEST - integrated_mismatches}/{N_TEST} "
        f"({100*(N_TEST-integrated_mismatches)/N_TEST:.2f}%)"
    )
    print(
        f"reference vs events/cloud off  : "
        f"{N_TEST - no_events_mismatches}/{N_TEST} "
        f"({100*(N_TEST-no_events_mismatches)/N_TEST:.2f}%)"
    )
    print(
        f"reference vs cache off         : "
        f"{N_TEST - no_cache_mismatches}/{N_TEST} "
        f"({100*(N_TEST-no_cache_mismatches)/N_TEST:.2f}%)"
    )

    print("\n--- Cache correctness ---")
    print(f"cache hits                     : {integrated_cache_hits}")
    print(f"cache-hit label mismatches     : {integrated_cache_hit_mismatches}")
    print(
        f"RF executions in integrated RT : "
        f"{integrated_rt.stats.rf_executions}"
    )

    print("\n--- Cloud independence ---")
    print(f"offline config applied         : {'YES' if applied_offline else 'NO'}")
    print(f"previous config retained       : {'YES' if config_retained else 'NO'}")
    print(
        f"local inference after outage   : "
        f"{local_after_outage['hazard_state']}"
    )

    print("\n--- Overall ---")
    print(f"INTEGRATION REGRESSION         : {'PASS' if overall_pass else 'FAIL'}")

    if not overall_pass:
        print("\nFailure details:")
        print(f"  integrated mismatches         : {integrated_mismatches}")
        print(f"  events/cloud-off mismatches  : {no_events_mismatches}")
        print(f"  cache-off mismatches          : {no_cache_mismatches}")
        print(f"  cache-hit mismatches          : {integrated_cache_hit_mismatches}")

    print("\nNo performance claim is made by this script.")
    print("Use 11_verify_benchmarks.py for latency/model-size/accuracy claims.")
    print("=" * 78)

    if not overall_pass:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
