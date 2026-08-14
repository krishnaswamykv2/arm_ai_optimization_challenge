from decision_engine import decide
from rover_action_controller import RoverActionController
from action_verifier import verify_action


def run_scenario(
    name,
    risk_assessment,
    before_risk,
    after_risk,
    before_trend,
    after_trend,
):
    print(f"\n--- {name} ---")

    # E4: decide
    decision = decide(risk_assessment)

    print("DECISION:")
    print(decision)

    # E5-A: act
    controller = RoverActionController()
    rover_state = controller.execute(decision)

    print("ROVER STATE:")
    print(rover_state)

    # E5-B: verify consequence
    verification = verify_action(
        decision["action"],
        before_risk,
        after_risk,
        before_trend,
        after_trend,
    )

    print("VERIFICATION:")
    print(verification)

    return decision, rover_state, verification


if __name__ == "__main__":

    # ---------------------------------------------------------
    # 1. Successful recovery
    # ---------------------------------------------------------

    run_scenario(
        "Successful recovery",
        {
            "risk_level": "HIGH",
            "urgency": "WATCH",
            "model_alignment": "ESCALATING_BEYOND_MODEL",
        },
        before_risk=0.72,
        after_risk=0.51,
        before_trend="RISING",
        after_trend="FALLING",
    )

    # ---------------------------------------------------------
    # 2. Failed intervention
    # ---------------------------------------------------------

    run_scenario(
        "Failed intervention",
        {
            "risk_level": "HIGH",
            "urgency": "PROMPT",
            "model_alignment": "ESCALATING_BEYOND_MODEL",
        },
        before_risk=0.72,
        after_risk=0.78,
        before_trend="RISING",
        after_trend="RISING",
    )

    # ---------------------------------------------------------
    # 3. Critical hazard avoidance
    # ---------------------------------------------------------

    run_scenario(
        "Critical hazard avoidance",
        {
            "risk_level": "CRITICAL",
            "urgency": "IMMEDIATE",
            "model_alignment": "CONSISTENT",
        },
        before_risk=0.90,
        after_risk=0.45,
        before_trend="RISING",
        after_trend="FALLING",
    )

    # ---------------------------------------------------------
    # 4. Model / physical disagreement
    # ---------------------------------------------------------

    run_scenario(
        "Model disagreement",
        {
            "risk_level": "LOW",
            "urgency": "PROMPT",
            "model_alignment": "MODEL_PHYSICAL_DISAGREEMENT",
        },
        before_risk=0.20,
        after_risk=0.21,
        before_trend="FALLING",
        after_trend="STABLE",
    )