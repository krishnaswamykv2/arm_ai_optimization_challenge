"""
MASTER PIPELINE: chains all AI layers together.

Each layer is a separate, independently-testable function. This script
only ORCHESTRATES them -- it doesn't contain any AI logic itself, which
is intentional. If you improve/replace one layer later (e.g., swap in
a better drift correction technique, or add explainability), you only
touch that layer's code -- this file's structure doesn't change.

LAYER ORDER (matters!):
    raw sensor reading
        -> Layer 2: Drift Correction   (adjust raw values first)
        -> Layer 1: Fusion             (compute engineered/fused features)
        -> Layer 3: Classification     (predict hazard state)
        -> Layer 4: Output             (return structured result)
"""

import joblib
import pandas as pd
from collections import deque
from feature_engineering_helper import engineer_single_reading
from environmental_context import analyze_environment


# ---------------------------------------------------------------
# LAYER 2: Drift Correction
# ---------------------------------------------------------------
# Adaptive rolling-baseline correction, applied independently per sensor.
#
# WHY THIS TECHNIQUE (not the per-batch standardization validated in
# 07_drift_correction_comparison.py): that technique needs a whole batch
# of readings at once to compute z-scores. This pipeline processes ONE
# live reading at a time -- there's no "batch" to standardize against.
# Rolling-baseline correction is the standard lightweight real-time
# adaptation of the same idea: track a rolling median of recent readings
# as the "current baseline," compare it to a "reference baseline"
# established during an initial calibration period, and subtract the
# difference (the drift) from each new raw reading. This preserves real
# physical units (ppm, degrees, g) -- required here because the risk
# formulas in feature_engineering_helper.py use fixed absolute thresholds
# (GAS_MAX, TILT_MAX, etc.), not standardized scores.
#
# MEASURED LIMITATION (tested in drift_baseline_test.py against simulated
# continuous drift): this reduces drift-induced error by roughly HALF, not
# to zero. Rolling-median tracking inherently lags behind continuous,
# accelerating drift because the baseline itself is computed from a window
# that already includes some drifted readings. This is a known property of
# this class of technique, not a bug -- report it honestly as "meaningfully
# reduces drift error" rather than "eliminates drift."
class BaselineDriftTracker:
    def __init__(self, window_size=50, calibration_size=15):
        self.window_size = window_size
        self.calibration_size = calibration_size
        self.buffer = deque(maxlen=window_size)
        self.reference_baseline = None
        self._calibration_buffer = []

    def _median(self, values):
        s = sorted(values)
        n = len(s)
        mid = n // 2
        if n % 2 == 0:
            return (s[mid - 1] + s[mid]) / 2
        return s[mid]

    def update_and_correct(self, raw_value):
        if self.reference_baseline is None:
            # Still calibrating -- assumes the rover starts in a known-safe
            # state for the first `calibration_size` readings.
            self._calibration_buffer.append(raw_value)
            self.buffer.append(raw_value)
            if len(self._calibration_buffer) >= self.calibration_size:
                self.reference_baseline = self._median(self._calibration_buffer)
            return raw_value  # no correction applied yet during calibration

        self.buffer.append(raw_value)
        current_baseline = self._median(self.buffer)
        drift_offset = current_baseline - self.reference_baseline
        return raw_value - drift_offset


# One tracker per sensor stream -- module-level so state persists across
# repeated calls to run_pipeline() as the rover keeps sampling.
_gas_tracker = BaselineDriftTracker()
_tilt_tracker = BaselineDriftTracker()
_vibration_tracker = BaselineDriftTracker()


def apply_drift_correction(gas_ppm, tilt_deg, vibration_g):
    """
    Input: raw sensor values
    Output: drift-corrected sensor values (same shape/units)

    NOTE: correction only activates after each tracker's calibration
    period (first `calibration_size` calls per sensor). Until then this
    is a pass-through, on the assumption the rover starts in a known-safe
    state that's used to set the reference baseline.
    """
    corrected_gas = _gas_tracker.update_and_correct(gas_ppm)
    corrected_tilt = _tilt_tracker.update_and_correct(tilt_deg)
    corrected_vibration = _vibration_tracker.update_and_correct(vibration_g)
    return corrected_gas, corrected_tilt, corrected_vibration


# ---------------------------------------------------------------
# LAYER 1: Fusion / Feature Engineering
# ---------------------------------------------------------------
# Already built -- reuses your existing helper directly, no changes needed.
def apply_fusion(gas_ppm, tilt_deg, vibration_g):
    """
    Input: (corrected) raw sensor values
    Output: dict of engineered features including joint_risk (the fusion signal)
    """
    return engineer_single_reading(gas_ppm, tilt_deg, vibration_g)


# ---------------------------------------------------------------
# LAYER 3: Hazard Classification
# ---------------------------------------------------------------
_model = joblib.load("hazard_model.joblib")

# Column order engineer_single_reading() returns its dict in -- must match
# the column order feature_columns used at training time (03_train_model.py).
# Kept as an explicit constant (rather than relying on dict-ordering being
# obviously correct at a glance) since apply_classification's numpy path
# depends on it.
FEATURE_COLUMN_ORDER = [
    "gas_ppm", "tilt_deg", "vibration_g",
    "gas_risk", "tilt_risk", "vibration_risk",
    "structural_risk", "joint_risk",
]


def apply_classification(features_dict, model=None, use_numpy=False):
    """
    Input: dict of engineered features (from Layer 1)
    Output: (prediction, confidence_dict)

    model: which trained classifier to use. Defaults to the module-level
        baseline model (_model) so existing callers (12_laptop_live_demo.py,
        13_hardware_deployment_pipeline.py) are unaffected.
    use_numpy: if True, build a plain NumPy row instead of a 1-row pandas
        DataFrame before calling predict/predict_proba. This is the
        preprocessing optimization from 11_verify_benchmarks.py -- it's
        opt-in and defaults to False so default behavior is unchanged.
    """
    if model is None:
        model = _model

    if use_numpy:
        import numpy as np
        X = np.array([[features_dict[c] for c in FEATURE_COLUMN_ORDER]])
    else:
        X = pd.DataFrame([features_dict], columns=FEATURE_COLUMN_ORDER)

    prediction = model.predict(X)[0]
    probabilities = model.predict_proba(X)[0]
    confidence = dict(zip(model.classes_, probabilities.round(3)))
    return prediction, confidence


# ---------------------------------------------------------------
# LAYER 4: Output formatting
# ---------------------------------------------------------------
def format_output(prediction, confidence, raw_reading, corrected_reading):
    return {
        "raw_input": raw_reading,
        "corrected_input": corrected_reading,
        "hazard_state": prediction.upper(),
        "confidence": confidence,
    }


# ---------------------------------------------------------------
# PIPELINE ORCHESTRATION
# ---------------------------------------------------------------
def run_pipeline(gas_ppm, tilt_deg, vibration_g):
    raw_reading = {"gas_ppm": gas_ppm, "tilt_deg": tilt_deg, "vibration_g": vibration_g}

    # Stage 1: drift correction
    corrected_gas, corrected_tilt, corrected_vib = apply_drift_correction(
        gas_ppm, tilt_deg, vibration_g
    )
    corrected_reading = {
        "gas_ppm": corrected_gas, "tilt_deg": corrected_tilt, "vibration_g": corrected_vib
    }

    # Stage 2: fusion / feature engineering
    features = apply_fusion(corrected_gas, corrected_tilt, corrected_vib)

    # Stage 3: classification
    prediction, confidence = apply_classification(features)

    # Stage 4: output
    return format_output(prediction, confidence, raw_reading, corrected_reading)


# ---------------------------------------------------------------
# PIPELINE FACTORY (for benchmarking -- 11_verify_benchmarks.py)
# ---------------------------------------------------------------
# run_pipeline() above is left untouched for 12_laptop_live_demo.py and
# 13_hardware_deployment_pipeline.py, which both rely on the SAME module-
# level trackers persisting across repeated calls (that's the point --
# the drift tracker needs continuous state as the rover keeps sampling).
#
# For benchmarking baseline-vs-optimized, that shared global state is a
# problem: running both configs through the same trackers would let one
# config's calibration/drift history leak into the other's timing and
# predictions. build_pipeline() returns a self-contained pipeline with
# its OWN trackers and its OWN model, so two configs can be benchmarked
# side by side without contaminating each other.
def build_pipeline(model_path="hazard_model.joblib", use_numpy=False):
    """
    Returns a run(gas_ppm, tilt_deg, vibration_g) callable, with independent
    drift-tracker state, wired to the given model file and preprocessing
    style. Does not touch or reset the module-level _gas_tracker etc. used
    by run_pipeline().
    """
    model = joblib.load(model_path)
    gas_tracker = BaselineDriftTracker()
    tilt_tracker = BaselineDriftTracker()
    vibration_tracker = BaselineDriftTracker()

    def run(gas_ppm, tilt_deg, vibration_g):
        raw_reading = {"gas_ppm": gas_ppm, "tilt_deg": tilt_deg, "vibration_g": vibration_g}

        corrected_gas = gas_tracker.update_and_correct(gas_ppm)
        corrected_tilt = tilt_tracker.update_and_correct(tilt_deg)
        corrected_vib = vibration_tracker.update_and_correct(vibration_g)
        corrected_reading = {
            "gas_ppm": corrected_gas, "tilt_deg": corrected_tilt, "vibration_g": corrected_vib
        }

        features = apply_fusion(corrected_gas, corrected_tilt, corrected_vib)
        prediction, confidence = apply_classification(features, model=model, use_numpy=use_numpy)
        return format_output(prediction, confidence, raw_reading, corrected_reading)

    return run


if __name__ == "__main__":
    test_cases = [
        ("Normal", 75, 2, 0.04),
        ("Gas creeping up, structure fine", 480, 3, 0.05),
        ("Structure shaking, gas fine", 90, 20, 0.45),
        ("Both bad", 700, 27, 0.7),
    ]

    for label, gas, tilt, vib in test_cases:
        result = run_pipeline(gas, tilt, vib)
        print(f"\n{label}")
        print(f"  Raw input: {result['raw_input']}")
        print(f"  Hazard state: {result['hazard_state']}")
        print(f"  Confidence: {result['confidence']}")
