"""
ECOsphere Action Verification

Determines whether an autonomous action produced an improving,
stable, or worsening environmental outcome.

This module evaluates environmental consequence, not merely
whether an action command was executed.
"""


def verify_action(
    action,
    before_risk,
    after_risk,
    before_trend,
    after_trend,
):
    """
    Verify the environmental consequence of an action.

    Risk values are expected to be normalized to [0, 1].
    """

    action = str(action).upper()
    before_trend = str(before_trend).upper()
    after_trend = str(after_trend).upper()

    risk_change = after_risk - before_risk

    # ---------------------------------------------------------
    # 1. Clear recovery
    # ---------------------------------------------------------

    if (
        after_risk < before_risk
        and after_trend == "FALLING"
    ):
        return {
            "verification": "RECOVERY_CONFIRMED",
            "effectiveness": "ACTION_EFFECTIVE",
            "risk_change": round(risk_change, 3),
            "reason": (
                "Environmental risk decreased and the "
                "trajectory is improving."
            ),
            "next_action": "RESUME_WHEN_SAFE",
        }

    # ---------------------------------------------------------
    # 2. Clear deterioration
    # ---------------------------------------------------------

    if (
        after_risk > before_risk
        and after_trend == "RISING"
    ):
        return {
            "verification": "HAZARD_ESCALATED",
            "effectiveness": "ACTION_INEFFECTIVE",
            "risk_change": round(risk_change, 3),
            "reason": (
                "Environmental risk increased and the "
                "trajectory continues to worsen."
            ),
            "next_action": "ESCALATE",
        }

    # ---------------------------------------------------------
    # 3. Risk improved but trajectory is not yet confirmed
    # ---------------------------------------------------------

    if after_risk < before_risk:
        return {
            "verification": "PARTIAL_IMPROVEMENT",
            "effectiveness": "IMPROVING",
            "risk_change": round(risk_change, 3),
            "reason": (
                "Environmental risk decreased, but the "
                "trajectory does not yet confirm recovery."
            ),
            "next_action": "CONTINUE_MONITORING",
        }

    # ---------------------------------------------------------
    # 4. Risk increased but trend is not yet confirmed
    # ---------------------------------------------------------

    if after_risk > before_risk:
        return {
            "verification": "WARNING",
            "effectiveness": "POTENTIALLY_INEFFECTIVE",
            "risk_change": round(risk_change, 3),
            "reason": (
                "Environmental risk increased, but the "
                "trajectory is not yet sufficient to confirm "
                "continued deterioration."
            ),
            "next_action": "CONTINUE_MONITORING",
        }

    # ---------------------------------------------------------
    # 5. No meaningful change
    # ---------------------------------------------------------

    return {
        "verification": "NO_SIGNIFICANT_CHANGE",
        "effectiveness": "INCONCLUSIVE",
        "risk_change": round(risk_change, 3),
        "reason": (
            "Environmental risk did not change enough "
            "to establish action effectiveness."
        ),
        "next_action": "CONTINUE_MONITORING",
    }


if __name__ == "__main__":

    scenarios = [
        {
            "name": "Successful recovery",
            "action": "SLOW_AND_MONITOR",
            "before_risk": 0.72,
            "after_risk": 0.51,
            "before_trend": "RISING",
            "after_trend": "FALLING",
        },
        {
            "name": "Action ineffective",
            "action": "SLOW_AND_MONITOR",
            "before_risk": 0.72,
            "after_risk": 0.78,
            "before_trend": "RISING",
            "after_trend": "RISING",
        },
        {
            "name": "Partial improvement",
            "action": "SLOW_AND_MONITOR",
            "before_risk": 0.72,
            "after_risk": 0.60,
            "before_trend": "RISING",
            "after_trend": "STABLE",
        },
        {
            "name": "No significant change",
            "action": "MONITOR",
            "before_risk": 0.50,
            "after_risk": 0.50,
            "before_trend": "STABLE",
            "after_trend": "STABLE",
        },
        {
            "name": "Warning",
            "action": "MONITOR",
            "before_risk": 0.50,
            "after_risk": 0.55,
            "before_trend": "STABLE",
            "after_trend": "STABLE",
        },
    ]

    for scenario in scenarios:

        result = verify_action(
            scenario["action"],
            scenario["before_risk"],
            scenario["after_risk"],
            scenario["before_trend"],
            scenario["after_trend"],
        )

        print(f"\n--- {scenario['name']} ---")
        print(result)