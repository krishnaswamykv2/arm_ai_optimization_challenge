"""
ECOsphere UI Adapter
====================

Thin presentation adapter between the validated ECOsphere runtime and the
judge-facing UI.

Rules:
- Does NOT calculate a new risk.
- Does NOT override runtime decisions.
- Does NOT simulate rover behavior.
- Does NOT manufacture cloud connectivity.
- Only normalizes runtime output and maps it to human-facing presentation
  states/messages.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Optional


@dataclass
class UIState:
    """Canonical state consumed by the ECOsphere UI."""

    presentation_state: str

    risk: float
    risk_level: str
    trend: str

    gas: float
    tilt: float
    vibration: float

    hazard_state: str
    sensor_agreement: str

    decision: str
    priority: str
    trigger: str
    decision_reason: str
    verification_required: bool

    rover_motion: str
    rover_speed: str
    route: str
    rover_action: str

    verification: Optional[Dict[str, Any]]
    recovery: Optional[Dict[str, Any]]

    explanation: str
    next_step: str

    cache_hit: bool
    rf_executed: bool
    config_version: int

    # Intentionally unknown unless a real connectivity source supplies it.
    cloud_connected: Optional[bool]
    decision_path: str


def _risk(result: Dict[str, Any]) -> float:
    return float(
        result.get("environment", {})
        .get("risk_features", {})
        .get("joint_risk", 0.0)
    )


def _presentation_state(result: Dict[str, Any]) -> str:
    """
    Presentation-only mapping.

    The runtime remains the source of truth. These labels describe the
    runtime output in language useful to a human observer.
    """

    decision = result.get("decision", {})
    action = str(decision.get("action", "")).upper()

    recovery = result.get("recovery") or {}
    if recovery.get("recovery_confirmed") is True:
        return "RECOVERY_CONFIRMED"

    if action == "STOP_AND_REROUTE":
        return "CRITICAL"

    verification = result.get("verification")
    if verification is not None:
        return "CHECKING"

    if action == "SLOW_AND_MONITOR":
        return "RISK_RISING"

    temporal = result.get("temporal", {})
    trend = str(temporal.get("trend", "")).upper()

    risk_level = str(
        result.get("risk_assessment", {})
        .get("risk_level", "LOW")
    ).upper()

    hazard = str(result.get("hazard_state", "SAFE")).upper()

    if trend == "RISING" and risk_level in {"MODERATE", "HIGH", "CRITICAL"}:
        return "RISK_RISING"

    if hazard == "ELEVATED":
        return "ENVIRONMENT_CHANGED"

    return "SURVEYING"


def _explanation(result: Dict[str, Any]) -> str:
    """Use runtime-provided rationale/context; never invent new evidence."""

    risk_assessment = result.get("risk_assessment", {})
    rationale = risk_assessment.get("rationale") or []

    if rationale:
        return " ".join(str(item) for item in rationale[:3])

    environment = result.get("environment", {})
    return str(
        environment.get(
            "context",
            "Environmental conditions are being evaluated locally.",
        )
    )


def _next_step(result: Dict[str, Any]) -> str:
    """Human-facing description derived only from runtime outputs."""

    verification = result.get("verification") or {}
    recovery = result.get("recovery") or {}
    decision = result.get("decision", {})

    if recovery.get("recovery_confirmed") is True:
        return "Recovery is confirmed; safe movement can resume."

    if verification.get("next_action") == "ESCALATE":
        return "Escalate the response because the previous intervention did not reduce the hazard."

    action = str(decision.get("action", "CONTINUE")).upper()

    if action == "STOP_AND_REROUTE":
        return "Remain stopped and verify recovery before resuming."

    if decision.get("verification_required"):
        return "Observe the next reading and verify whether the intervention worked."

    if action == "SLOW_AND_MONITOR":
        return "Continue at reduced speed while monitoring the environment."

    return "Continue sensing the environment."


def to_ui_state(
    runtime_output: Dict[str, Any],
    *,
    cloud_connected: Optional[bool] = None,
) -> UIState:
    """
    Convert one real ECOSphereRuntime.step() result into the UI contract.

    cloud_connected is intentionally an explicit external input because the
    current runtime result does not expose physical network connectivity.
    """

    result = runtime_output

    corrected = result.get("corrected_input", {})
    environment = result.get("environment", {})
    risk_assessment = result.get("risk_assessment", {})
    decision = result.get("decision", {})
    rover = result.get("rover_state", {})

    return UIState(
        presentation_state=_presentation_state(result),

        risk=_risk(result),
        risk_level=str(risk_assessment.get("risk_level", "LOW")).upper(),
        trend=str(
            result.get("temporal", {}).get(
                "trend", "INSUFFICIENT_DATA"
            )
        ).upper(),

        gas=float(corrected.get("gas_ppm", 0.0)),
        tilt=float(corrected.get("tilt_deg", 0.0)),
        vibration=float(corrected.get("vibration_g", 0.0)),

        hazard_state=str(
            result.get("hazard_state", "SAFE")
        ).upper(),
        sensor_agreement=str(
            environment.get("sensor_agreement", "NONE")
        ).upper(),

        decision=str(
            decision.get("action", "CONTINUE")
        ).upper(),
        priority=str(
            decision.get("priority", "NORMAL")
        ).upper(),
        trigger=str(
            decision.get("trigger", "NORMAL_OPERATION")
        ).upper(),
        decision_reason=str(
            decision.get(
                "reason",
                "Environmental conditions support continued operation.",
            )
        ),
        verification_required=bool(
            decision.get("verification_required", False)
        ),

        rover_motion=str(
            rover.get("motion_state", "UNKNOWN")
        ).upper(),
        rover_speed=str(
            rover.get("speed_mode", "UNKNOWN")
        ).upper(),
        route=str(
            rover.get("route_state", "UNKNOWN")
        ).upper(),
        rover_action=str(
            rover.get("last_action", decision.get("action", "CONTINUE"))
        ).upper(),

        verification=result.get("verification"),
        recovery=result.get("recovery"),

        explanation=_explanation(result),
        next_step=_next_step(result),

        cache_hit=bool(result.get("cache_hit", False)),
        rf_executed=bool(result.get("rf_executed", False)),
        config_version=int(result.get("config_version", 1)),

        cloud_connected=cloud_connected,
        decision_path="LOCAL",
    )


def ui_state_dict(ui_state: UIState) -> Dict[str, Any]:
    """Convenience helper for Streamlit/debug rendering."""
    return {
        key: value
        for key, value in ui_state.__dict__.items()
    }


if __name__ == "__main__":
    print("ECOsphere UI adapter loaded.")
    print("Use to_ui_state(runtime.step(...)) to create the UI contract.")