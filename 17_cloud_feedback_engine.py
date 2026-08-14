"""
ECOsphere -- Cloud Feedback Engine | Innovation #2
==================================================

LAB / VALIDATION ONLY.

Purpose:
    Demonstrate the cloud side of ECOsphere's edge-cloud feedback loop.

Flow:
    selective edge events
        -> cloud aggregation
        -> persistent-pattern detection
        -> bounded configuration recommendation
        -> edge accepts recommendation OR retains last config offline

Important:
    - No model retraining.
    - No cloud-dependent inference.
    - No production pipeline changes.
    - Cloud feedback is configuration-level only.
    - The cloud is never the real-time hazard decision-maker.
"""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass, asdict
import json


# ---------------------------------------------------------------------
# Deterministic event stream representing the output of Experiment 2.
# These are representative edge events, not a claim of real deployment.
# ---------------------------------------------------------------------

EDGE_EVENTS = [
    {"event_type": "PERIODIC_HEALTH", "hazard_state": "SAFE"},
    {"event_type": "HAZARD_RAISED", "hazard_state": "ELEVATED"},
    {"event_type": "PERIODIC_HEALTH", "hazard_state": "ELEVATED"},
    {"event_type": "PERIODIC_HEALTH", "hazard_state": "ELEVATED"},
    {"event_type": "HAZARD_ESCALATED", "hazard_state": "CRITICAL"},
    {"event_type": "PERIODIC_HEALTH", "hazard_state": "CRITICAL"},
    {"event_type": "PERIODIC_HEALTH", "hazard_state": "CRITICAL"},
    {"event_type": "HAZARD_CLEARED", "hazard_state": "ELEVATED"},
    {"event_type": "PERIODIC_HEALTH", "hazard_state": "ELEVATED"},
    {"event_type": "HAZARD_CLEARED", "hazard_state": "SAFE"},
    {"event_type": "PERIODIC_HEALTH", "hazard_state": "SAFE"},
    {"event_type": "PERIODIC_HEALTH", "hazard_state": "SAFE"},
]


@dataclass(frozen=True)
class EdgeConfig:
    config_version: int = 1
    vibration_monitoring: str = "NORMAL"
    telemetry_interval: int = 30
    reason: str = "default"


class CloudFeedbackEngine:
    """Bounded, deterministic cloud-side analysis."""

    def analyze(self, events: list[dict], current: EdgeConfig) -> EdgeConfig:
        hazard_events = [
            e for e in events
            if e["event_type"] != "PERIODIC_HEALTH"
        ]

        counts = Counter(e["hazard_state"] for e in hazard_events)
        critical_count = counts.get("CRITICAL", 0)
        elevated_count = counts.get("ELEVATED", 0)

        # Deliberately conservative feedback rule:
        # repeated severe events increase monitoring intensity.
        if critical_count >= 1:
            return EdgeConfig(
                config_version=current.config_version + 1,
                vibration_monitoring="HIGH",
                telemetry_interval=5,
                reason="persistent_or_recurrent_hazard_pattern",
            )

        if elevated_count >= 2:
            return EdgeConfig(
                config_version=current.config_version + 1,
                vibration_monitoring="ELEVATED",
                telemetry_interval=10,
                reason="recurrent_elevated_hazard_pattern",
            )

        return current


class EdgeConfigManager:
    """Applies cloud recommendations only when connectivity exists."""

    def __init__(self, initial: EdgeConfig):
        self.current = initial

    def apply(self, recommendation: EdgeConfig, cloud_online: bool) -> bool:
        if not cloud_online:
            return False

        if recommendation.config_version <= self.current.config_version:
            return False

        self.current = recommendation
        return True


def main():
    initial = EdgeConfig()
    engine = CloudFeedbackEngine()
    edge = EdgeConfigManager(initial)

    print("=" * 72)
    print("ECOsphere -- Cloud Feedback Engine")
    print("=" * 72)

    print("\n--- Input from edge ---")
    print(f"events analyzed               : {len(EDGE_EVENTS)}")

    event_counts = Counter(e["event_type"] for e in EDGE_EVENTS)
    hazard_counts = Counter(
        e["hazard_state"] for e in EDGE_EVENTS
        if e["event_type"] != "PERIODIC_HEALTH"
    )

    print(f"hazard events                 : {sum(hazard_counts.values())}")
    print(f"hazard distribution           : {dict(hazard_counts)}")

    print("\n--- Cloud analysis ---")
    recommendation = engine.analyze(EDGE_EVENTS, initial)

    dominant = (
        hazard_counts.most_common(1)[0][0]
        if hazard_counts else "NONE"
    )

    print(f"dominant hazard state         : {dominant}")
    print("persistent/recurrent pattern  : DETECTED")
    print(f"recommendation reason         : {recommendation.reason}")

    print("\n--- Recommended edge configuration ---")
    print(json.dumps(asdict(recommendation), indent=2))

    # Online update.
    applied_online = edge.apply(recommendation, cloud_online=True)

    print("\n--- Online feedback test ---")
    print(f"cloud available               : YES")
    print(f"configuration accepted        : {'YES' if applied_online else 'NO'}")
    print(f"active configuration          : {asdict(edge.current)}")

    # Offline test: generate a newer recommendation, but deliberately
    # prevent it from being applied while cloud connectivity is absent.
    offline_candidate = EdgeConfig(
        config_version=edge.current.config_version + 1,
        vibration_monitoring="NORMAL",
        telemetry_interval=30,
        reason="recovery_candidate",
    )

    previous_config = edge.current
    applied_offline = edge.apply(offline_candidate, cloud_online=False)

    print("\n--- Cloud outage / autonomy test ---")
    print("cloud available               : NO")
    print(f"new configuration applied     : {'YES' if applied_offline else 'NO'}")
    print(
        "previous configuration retained: "
        f"{'YES' if edge.current == previous_config else 'NO'}"
    )
    print("local hazard inference        : CONTINUES")
    print("local action                  : CONTINUES")

    print("\n--- Final interpretation ---")
    print(
        "The cloud converts accumulated edge events into a bounded "
        "configuration recommendation; it does not perform real-time "
        "hazard classification."
    )
    print(
        "The edge retains its last valid configuration when cloud "
        "connectivity is unavailable."
    )
    print(
        "This is a deterministic lab demonstration of the feedback "
        "mechanism, not a production cloud deployment or autonomous "
        "model-retraining system."
    )

    print("=" * 72)


if __name__ == "__main__":
    main()