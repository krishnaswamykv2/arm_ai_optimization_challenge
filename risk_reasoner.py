"""
ECOsphere Risk Reasoner

Combines current physical risk, environmental evidence,
temporal behavior, near-term projection, and model confidence.

This is an interpretable reasoning layer.
It does not replace the trained classifier.
"""


def _risk_level(risk):
    risk = float(risk)

    if risk >= 0.80:
        return "CRITICAL"
    if risk >= 0.60:
        return "HIGH"
    if risk >= 0.30:
        return "MODERATE"
    return "LOW"


def _confidence_level(confidence):
    confidence = float(confidence)

    if confidence >= 0.80:
        return "HIGH"
    if confidence >= 0.60:
        return "MODERATE"
    return "LOW"


def _alignment(model_state, physical_level, trend, projected_level):
    model_state = str(model_state).upper()
    physical_level = str(physical_level).upper()
    trend = str(trend).upper()
    projected_level = str(projected_level).upper()

    severity = {
        "SAFE": 0,
        "ELEVATED": 1,
        "CRITICAL": 2,
    }

    physical_severity = {
        "LOW": 0,
        "MODERATE": 1,
        "HIGH": 2,
        "CRITICAL": 3,
    }

    model_value = severity.get(model_state)
    physical_value = physical_severity.get(physical_level)
    projected_value = physical_severity.get(projected_level)

    if model_value is None or physical_value is None:
        return "UNKNOWN"

    if (
        trend == "RISING"
        and projected_value is not None
        and projected_value > physical_value
        and projected_value > model_value
    ):
        return "ESCALATING_BEYOND_MODEL"

    if (
        model_value >= 2
        and physical_value <= 0
        and trend == "FALLING"
    ):
        return "MODEL_PHYSICAL_DISAGREEMENT"

    if abs(model_value - physical_value) <= 1:
        return "CONSISTENT"

    return "MODEL_PHYSICAL_DISAGREEMENT"


def _dominant_cause(environment):
    evidence = environment.get("evidence", {})

    high = [
        sensor.upper()
        for sensor, state in evidence.items()
        if str(state).upper() == "HIGH"
    ]

    moderate = [
        sensor.upper()
        for sensor, state in evidence.items()
        if str(state).upper() == "MODERATE"
    ]

    if high:
        return " + ".join(high)

    if moderate:
        return " + ".join(moderate)

    return "NONE"


def assess_risk(
    hazard_state,
    confidence,
    environment,
    temporal,
    projection,
):
    """
    Produce an interpretable risk assessment.
    """

    risk_features = environment.get("risk_features", {})

    current_risk = float(
        risk_features.get("joint_risk", 0.0)
    )

    projected_risk = float(
        projection.get("projected_risk", current_risk)
    )

    trend = str(
        temporal.get("trend", "INSUFFICIENT_DATA")
    ).upper()

    trend_strength = float(
        temporal.get("trend_strength", 0.0)
    )

    agreement = str(
        environment.get("sensor_agreement", "NONE")
    ).upper()

    model_confidence = float(confidence)

    physical_level = _risk_level(current_risk)
    projected_level = _risk_level(projected_risk)

    # Determine model/physical alignment BEFORE using it for urgency.
    alignment = _alignment(
        hazard_state,
        physical_level,
        trend,
        projected_level,
    )

    # Start from the current physical state.
    risk_level = physical_level

    # A meaningful rising trajectory can elevate the assessment.
    if (
        trend == "RISING"
        and trend_strength >= 0.50
        and projected_level in ("HIGH", "CRITICAL")
    ):
        risk_level = projected_level

    # Strong current physical evidence always dominates.
    if current_risk >= 0.80:
        risk_level = "CRITICAL"
    elif current_risk >= 0.60 and risk_level == "LOW":
        risk_level = "HIGH"

    # Determine urgency separately from severity.
    if alignment == "MODEL_PHYSICAL_DISAGREEMENT":
        urgency = "PROMPT"
    elif risk_level == "CRITICAL":
        urgency = "IMMEDIATE"
    elif (
        trend == "RISING"
        and trend_strength >= 0.70
    ):
        urgency = "PROMPT"
    elif risk_level == "HIGH":
        urgency = "WATCH"
    elif risk_level == "MODERATE":
        urgency = "WATCH"
    else:
        urgency = "NORMAL"

    # Evidence quality.
    confidence_quality = _confidence_level(
        model_confidence
    )

    if (
        physical_level == "LOW"
        and agreement == "NONE"
        and confidence_quality == "HIGH"
    ):
        evidence_quality = "HIGH"
    elif (
        agreement == "HIGH"
        and confidence_quality == "HIGH"
    ):
        evidence_quality = "HIGH"
    elif agreement in ("HIGH", "MODERATE"):
        evidence_quality = "MODERATE"
    else:
        evidence_quality = "LOW"

    cause = _dominant_cause(environment)

    rationale = []

    rationale.append(
        f"Current physical risk is {current_risk:.3f}."
    )

    if trend == "RISING":
        rationale.append(
            f"Environmental risk is rising "
            f"with trend strength {trend_strength:.3f}."
        )
    elif trend == "FALLING":
        rationale.append(
            f"Environmental risk is falling "
            f"with trend strength {trend_strength:.3f}."
        )
    elif trend == "STABLE":
        rationale.append(
            "Environmental risk is currently stable."
        )
    else:
        rationale.append(
            "Temporal history is insufficient for a reliable trend."
        )

    if projected_level != physical_level:
        rationale.append(
            f"Near-term projection indicates {projected_level} "
            f"risk."
        )

    if agreement != "NONE":
        rationale.append(
            f"Sensor agreement is {agreement}."
        )

    if alignment == "ESCALATING_BEYOND_MODEL":
        rationale.append(
            "Physical trajectory is becoming more severe "
            "than the current model hazard state."
        )

    if alignment == "MODEL_PHYSICAL_DISAGREEMENT":
        rationale.append(
        "Model severity and current physical evidence "
        "are not well aligned."
    )
        rationale.append(
        "Additional verification is recommended before "
        "taking an aggressive autonomous action."
    )
    return {
        "risk_level": risk_level,
        "urgency": urgency,
        "evidence_quality": evidence_quality,
        "model_alignment": alignment,
        "dominant_cause": cause,
        "rationale": rationale,
    }


if __name__ == "__main__":

    scenarios = [
        {
            "name": "Normal",
            "hazard_state": "SAFE",
            "confidence": 0.95,
            "environment": {
                "sensor_agreement": "NONE",
                "evidence": {
                    "gas": "LOW",
                    "tilt": "LOW",
                    "vibration": "LOW",
                },
                "risk_features": {
                    "joint_risk": 0.074,
                },
            },
            "temporal": {
                "trend": "STABLE",
                "trend_strength": 0.015,
            },
            "projection": {
                "projected_risk": 0.074,
                "projected_level": "LOW",
            },
        },
        {
            "name": "Rising hazard",
            "hazard_state": "ELEVATED",
            "confidence": 0.82,
            "environment": {
                "sensor_agreement": "HIGH",
                "evidence": {
                    "gas": "HIGH",
                    "tilt": "LOW",
                    "vibration": "HIGH",
                },
                "risk_features": {
                    "joint_risk": 0.50,
                },
            },
            "temporal": {
                "trend": "RISING",
                "trend_strength": 0.70,
            },
            "projection": {
                "projected_risk": 0.68,
                "projected_level": "HIGH",
            },
        },
        {
            "name": "Rapid escalation",
            "hazard_state": "ELEVATED",
            "confidence": 0.91,
            "environment": {
                "sensor_agreement": "HIGH",
                "evidence": {
                    "gas": "HIGH",
                    "tilt": "HIGH",
                    "vibration": "HIGH",
                },
                "risk_features": {
                    "joint_risk": 0.75,
                },
            },
            "temporal": {
                "trend": "RISING",
                "trend_strength": 0.90,
            },
            "projection": {
                "projected_risk": 0.93,
                "projected_level": "CRITICAL",
            },
        },
        {
            "name": "Recovering",
            "hazard_state": "CRITICAL",
            "confidence": 0.88,
            "environment": {
                "sensor_agreement": "HIGH",
                "evidence": {
                    "gas": "MODERATE",
                    "tilt": "LOW",
                    "vibration": "LOW",
                },
                "risk_features": {
                    "joint_risk": 0.72,
                },
            },
            "temporal": {
                "trend": "FALLING",
                "trend_strength": 0.70,
            },
            "projection": {
                "projected_risk": 0.60,
                "projected_level": "HIGH",
            },
        },
        {
            "name": "Model disagreement",
            "hazard_state": "CRITICAL",
            "confidence": 0.96,
            "environment": {
                "sensor_agreement": "LOW",
                "evidence": {
                    "gas": "LOW",
                    "tilt": "LOW",
                    "vibration": "LOW",
                },
                "risk_features": {
                    "joint_risk": 0.20,
                },
            },
            "temporal": {
                "trend": "FALLING",
                "trend_strength": 0.70,
            },
            "projection": {
                "projected_risk": 0.10,
                "projected_level": "LOW",
            },
        },
    ]

    for scenario in scenarios:
        result = assess_risk(
            scenario["hazard_state"],
            scenario["confidence"],
            scenario["environment"],
            scenario["temporal"],
            scenario["projection"],
        )

        print(f"\n--- {scenario['name']} ---")
        print(result)