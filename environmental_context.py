"""
ECOsphere Environmental Context Layer

Converts existing engineered sensor-risk features into an
interpretable environmental context.

This layer does NOT replace the trained classifier.
It explains the physical evidence supporting the classifier output.
"""


def _evidence_level(value):
    """Convert normalized risk evidence into an interpretable level."""
    if value >= 0.70:
        return "HIGH"
    if value >= 0.40:
        return "MODERATE"
    return "LOW"


def analyze_environment(features, hazard_state, confidence):
    """
    Build an interpretable environmental context from the existing
    feature-engineering output and RF classification result.

    Parameters
    ----------
    features : dict
        Output from feature_engineering_helper.engineer_single_reading()
    hazard_state : str
        RF-predicted hazard state.
    confidence : dict
        RF class probability dictionary.

    Returns
    -------
    dict
        Environmental context and sensor evidence.
    """

    gas_risk = float(features["gas_risk"])
    tilt_risk = float(features["tilt_risk"])
    vibration_risk = float(features["vibration_risk"])
    structural_risk = float(features["structural_risk"])
    joint_risk = float(features["joint_risk"])

    evidence = {
        "gas": _evidence_level(gas_risk),
        "tilt": _evidence_level(tilt_risk),
        "vibration": _evidence_level(vibration_risk),
    }

    # Count independent sensor channels showing meaningful abnormality.
    abnormal_channels = sum(
        value >= 0.40
        for value in (gas_risk, tilt_risk, vibration_risk)
    )

    if abnormal_channels == 3:
        sensor_agreement = "HIGH"
    elif abnormal_channels == 2:
        sensor_agreement = "MODERATE"
    elif abnormal_channels == 1:
        sensor_agreement = "LOW"
    else:
        sensor_agreement = "NONE"

    # Identify the strongest physical evidence.
    sensor_values = {
        "gas": gas_risk,
        "tilt": tilt_risk,
        "vibration": vibration_risk,
    }

    dominant_sensor = max(sensor_values, key=sensor_values.get)
    dominant_value = sensor_values[dominant_sensor]

    if dominant_value < 0.40:
        context = "NO SIGNIFICANT SENSOR ANOMALY"
    elif abnormal_channels == 3:
        context = "MULTI-SENSOR ENVIRONMENTAL ANOMALY"
    elif gas_risk >= 0.70 and structural_risk >= 0.40:
        context = "GAS + STRUCTURAL ANOMALY"
    elif gas_risk >= 0.70:
        context = "GAS-DOMINANT ANOMALY"
    elif tilt_risk >= 0.40 and vibration_risk >= 0.40:
        context = "STRUCTURAL TILT + VIBRATION ANOMALY"
    elif vibration_risk >= 0.40:
        context = "VIBRATION ANOMALY"
    elif tilt_risk >= 0.40:
        context = "TILT ANOMALY"
    else:
        context = "EMERGING ENVIRONMENTAL ANOMALY"

    max_confidence = float(max(confidence.values())) if confidence else 0.0
    return {
        "context": context,
        "sensor_agreement": sensor_agreement,
        "abnormal_channels": abnormal_channels,
        "dominant_sensor": dominant_sensor,
        "dominant_sensor_evidence": round(dominant_value, 3),
        "evidence": evidence,
        "risk_features": {
            "gas_risk": round(gas_risk, 3),
            "tilt_risk": round(tilt_risk, 3),
            "vibration_risk": round(vibration_risk, 3),
            "structural_risk": round(structural_risk, 3),
            "joint_risk": round(joint_risk, 3),
        },
        "model_confidence": round(max_confidence, 3),
        "hazard_state": str(hazard_state).upper(),
    }

if __name__ == "__main__":
    from feature_engineering_helper import engineer_single_reading

    test_cases = [
        ("Normal", 75, 2, 0.04),
        ("Gas dominant", 700, 3, 0.05),
        ("Structural", 90, 20, 0.45),
        ("Combined", 700, 27, 0.70),
    ]

    fake_confidence = {
        "SAFE": 0.05,
        "ELEVATED": 0.10,
        "CRITICAL": 0.85,
    }

    for label, gas, tilt, vibration in test_cases:
        features = engineer_single_reading(gas, tilt, vibration)

        result = analyze_environment(
            features,
            "CRITICAL",
            fake_confidence,
        )

        print(f"\n--- {label} ---")
        print("Context:", result["context"])
        print("Agreement:", result["sensor_agreement"])
        print("Evidence:", result["evidence"])
        print("Risk:", result["risk_features"])