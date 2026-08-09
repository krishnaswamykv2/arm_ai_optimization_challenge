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
from feature_engineering_helper import engineer_single_reading


# ---------------------------------------------------------------
# LAYER 2: Drift Correction
# ---------------------------------------------------------------
# Placeholder for now -- your validated technique (per-batch standardization
# or reference-instrument recalibration) plugs in here later. Until you
# have live calibration data from the actual rover, this is a pass-through
# so the pipeline runs end-to-end today.
def apply_drift_correction(gas_ppm, tilt_deg, vibration_g):
    """
    Input: raw sensor values
    Output: drift-corrected sensor values (same shape/units)

    TODO (once real sensor + reference data is available):
    replace this pass-through with the real correction logic validated
    in 07_drift_correction_comparison.py
    """
    corrected_gas = gas_ppm       # no correction applied yet
    corrected_tilt = tilt_deg     # IMU drift correction not yet implemented
    corrected_vibration = vibration_g
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

def apply_classification(features_dict):
    """
    Input: dict of engineered features (from Layer 1)
    Output: (prediction, confidence_dict)
    """
    X = pd.DataFrame([features_dict])
    prediction = _model.predict(X)[0]
    probabilities = _model.predict_proba(X)[0]
    confidence = dict(zip(_model.classes_, probabilities.round(3)))
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
