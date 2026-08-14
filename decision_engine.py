"""
ECOsphere Decision Engine

Converts the interpretable risk assessment into a recommended
autonomous action.

This layer does not directly control hardware.
It produces a structured decision for the action controller.
"""


def decide(risk_assessment):
    """
    Convert risk assessment into an autonomous decision.
    """

    risk_level = str(
        risk_assessment.get("risk_level", "LOW")
    ).upper()

    urgency = str(
        risk_assessment.get("urgency", "NORMAL")
    ).upper()

    alignment = str(
        risk_assessment.get(
            "model_alignment",
            "UNKNOWN",
        )
    ).upper()

    # ---------------------------------------------------------
    # 1. Evidence conflict has highest precedence.
    # ---------------------------------------------------------

    if alignment == "MODEL_PHYSICAL_DISAGREEMENT":
        return {
            "action": "PAUSE_AND_VERIFY",
            "priority": "PROMPT",
            "trigger": "MODEL_PHYSICAL_DISAGREEMENT",
            "reason": (
                "Model severity conflicts with current "
                "physical evidence; verification is required."
            ),
            "verification_required": True,
        }

    # ---------------------------------------------------------
    # 2. Critical immediate hazard.
    # ---------------------------------------------------------

    if (
        risk_level == "CRITICAL"
        and urgency == "IMMEDIATE"
    ):
        return {
            "action": "STOP_AND_REROUTE",
            "priority": "IMMEDIATE",
            "trigger": "CRITICAL_RISK",
            "reason": (
                "Critical environmental risk requires "
                "immediate hazard avoidance."
            ),
            "verification_required": False,
        }

    # ---------------------------------------------------------
    # 3. High and actively escalating.
    # ---------------------------------------------------------

    if (
        risk_level == "HIGH"
        and urgency == "PROMPT"
    ):
        return {
            "action": "SLOW_AND_MONITOR",
            "priority": "PROMPT",
            "trigger": "HIGH_ESCALATING_RISK",
            "reason": (
                "Risk is high and requires reduced-speed "
                "movement with continued monitoring."
            ),
            "verification_required": True,
        }

    # ---------------------------------------------------------
    # 4. Moderate/watch condition.
    # ---------------------------------------------------------

    if (
        risk_level == "MODERATE"
        and urgency == "WATCH"
    ):
        return {
            "action": "MONITOR",
            "priority": "WATCH",
            "trigger": "MODERATE_RISK",
            "reason": (
                "Moderate environmental risk detected; "
                "continue cautiously while monitoring."
            ),
            "verification_required": True,
        }

    # ---------------------------------------------------------
    # 5. High but not immediately escalating.
    # ---------------------------------------------------------

    if risk_level == "HIGH":
        if alignment == "ESCALATING_BEYOND_MODEL":
            trigger = "ESCALATING_RISK"
        else:
            trigger = "HIGH_RISK"

        return {
            "action": "SLOW_AND_MONITOR",
            "priority": "WATCH",
            "trigger": trigger,
            "reason": (
                "Risk is high or projected to become high; "
                "reduce movement speed and continue monitoring."
            ),
            "verification_required": True,
        }

    # ---------------------------------------------------------
    # 6. Critical but without immediate urgency.
    # ---------------------------------------------------------

    if risk_level == "CRITICAL":
        return {
            "action": "PAUSE_AND_VERIFY",
            "priority": "PROMPT",
            "trigger": "CRITICAL_RISK_REQUIRES_VERIFICATION",
            "reason": (
                "Critical risk is present but the assessment "
                "does not indicate immediate urgency; "
                "pause and verify conditions."
            ),
            "verification_required": True,
        }

    # ---------------------------------------------------------
    # 7. Low risk / normal operation.
    # ---------------------------------------------------------

    return {
        "action": "CONTINUE",
        "priority": "NORMAL",
        "trigger": "NORMAL_OPERATION",
        "reason": (
            "Environmental conditions support "
            "continued operation."
        ),
        "verification_required": False,
    }


if __name__ == "__main__":

    scenarios = [
        {
            "name": "Normal",
            "risk_level": "LOW",
            "urgency": "NORMAL",
            "model_alignment": "CONSISTENT",
        },
        {
            "name": "Moderate",
            "risk_level": "MODERATE",
            "urgency": "WATCH",
            "model_alignment": "CONSISTENT",
        },
        {
            "name": "High escalating",
            "risk_level": "HIGH",
            "urgency": "PROMPT",
            "model_alignment": "ESCALATING_BEYOND_MODEL",
        },
        {
            "name": "Critical",
            "risk_level": "CRITICAL",
            "urgency": "IMMEDIATE",
            "model_alignment": "CONSISTENT",
        },
        {
            "name": "Model disagreement",
            "risk_level": "LOW",
            "urgency": "PROMPT",
            "model_alignment": "MODEL_PHYSICAL_DISAGREEMENT",
        },
        {
            "name": "High recovering",
            "risk_level": "HIGH",
            "urgency": "WATCH",
            "model_alignment": "CONSISTENT",
        },
    ]

    for scenario in scenarios:
        result = decide(scenario)

        print(f"\n--- {scenario['name']} ---")
        print(result)