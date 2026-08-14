"""
ECOsphere Near-Term Risk Projection

Projects the next-step environmental risk using the current
joint-risk value and the temporal trend already calculated
by TemporalIntelligence.

This is an interpretable trajectory projection, not a trained
machine-learning prediction model.
"""


def project_risk(current_risk, slope, trend, horizon_steps=1):
    """
    Project near-term environmental risk.

    Parameters
    ----------
    current_risk : float
        Current normalized joint risk [0, 1].

    slope : float
        Temporal risk slope.

    trend : str
        Current temporal trend.

    horizon_steps : int
        Number of future observation steps to project.

    Returns
    -------
    dict
        Bounded projected risk and trajectory interpretation.
    """

    current_risk = float(current_risk)
    slope = float(slope)
    horizon_steps = int(horizon_steps)

    if horizon_steps < 1:
        raise ValueError("horizon_steps must be at least 1")

    raw_projected_risk = current_risk + (slope * horizon_steps)

    projection_clipped = not 0.0 <= raw_projected_risk <= 1.0

    projected_risk = max(
        0.0,
        min(1.0, raw_projected_risk),
    )

    if trend == "RISING" and projected_risk > current_risk:
        trajectory = "ESCALATING"
    elif trend == "FALLING" and projected_risk < current_risk:
        trajectory = "IMPROVING"
    elif trend == "STABLE":
        trajectory = "STABLE"
    else:
        trajectory = "UNCERTAIN"

    if projected_risk >= 0.80:
        projected_level = "CRITICAL"
    elif projected_risk >= 0.60:
        projected_level = "HIGH"
    elif projected_risk >= 0.30:
        projected_level = "MODERATE"
    else:
        projected_level = "LOW"

    return {
        "current_risk": round(current_risk, 3),
        "projected_risk": round(projected_risk, 3),
        "horizon_steps": horizon_steps,
        "trajectory": trajectory,
        "projected_level": projected_level,
        "slope": round(slope, 4),
        "projection_clipped": projection_clipped,
    }


if __name__ == "__main__":
    scenarios = [
        ("Rising", 0.60, 0.10, "RISING"),
        ("Rapid escalation", 0.75, 0.18, "RISING"),
        ("Falling", 0.70, -0.10, "FALLING"),
        ("Stable", 0.50, 0.002, "STABLE"),
    ]

    for name, risk, slope, trend in scenarios:
        result = project_risk(
            risk,
            slope,
            trend,
            horizon_steps=1,
        )

        print(f"\n--- {name} ---")
        print(result)