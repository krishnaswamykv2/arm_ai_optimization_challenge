"""
ECOsphere -- Edge Event Intelligence, Experiment 2
==================================================

LAB / VALIDATION ONLY.

Experiment 2 uses persistent temporal regimes rather than rapidly alternating
scenarios. The goal is to test whether selective telemetry remains useful when
the physical state changes slowly and hazards persist.

This does NOT claim real-world temporal validation. The stream is controlled
and synthetic, built from the project's existing sensor operating points.

No production file, model, cache experiment, benchmark, or dashboard is
modified.
"""

from __future__ import annotations

import importlib.util
from collections import deque
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parent
MASTER = ROOT / "08_master_pipeline.py"
MODEL = ROOT / "hazard_model_optimized.joblib"

RANDOM_SEED = 42
HEALTH_INTERVAL = 100
MAX_BUFFER = 256

# Long persistent regimes: the physical state remains stable for hundreds of
# readings before moving to another regime.
REGIMES = [
    ("SAFE",     np.array([75.0,  2.0, 0.04]), 450),
    ("ELEVATED", np.array([480.0, 3.0, 0.05]), 350),
    ("CRITICAL", np.array([700.0, 27.0, 0.70]), 300),
    ("ELEVATED", np.array([480.0, 3.0, 0.05]), 350),
    ("SAFE",     np.array([75.0,  2.0, 0.04]), 450),
]

# Small within-regime sensor noise. It is deliberately much smaller than the
# distance between the operating points.
NOISE_STD = np.array([8.0, 0.20, 0.008])

OFFLINE_START = 900
OFFLINE_END = 1200


SEVERITY = {
    "SAFE": 0,
    "ELEVATED": 1,
    "CRITICAL": 2,
}


def load_master():
    if not MASTER.exists():
        raise FileNotFoundError(f"Missing {MASTER.name}")

    spec = importlib.util.spec_from_file_location(
        "ecosphere_master_experiment2",
        MASTER,
    )
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


class CloudSink:
    def __init__(self):
        self.online = True
        self.received = []

    def send(self, event):
        if not self.online:
            return False
        self.received.append(event)
        return True


class EdgeEventIntelligence:
    def __init__(self, health_interval=100, max_buffer=256):
        self.health_interval = health_interval
        self.buffer = deque(maxlen=max_buffer)
        self.previous = None

        self.readings = 0
        self.events_generated = 0
        self.events_delivered = 0
        self.events_buffered = 0

    def observe(self, result):
        self.readings += 1
        current = result["hazard_state"].upper()
        events = []

        if self.previous is not None and current != self.previous:
            old_level = SEVERITY.get(self.previous, 0)
            new_level = SEVERITY.get(current, 0)

            if self.previous == "SAFE" and new_level > old_level:
                event_type = "HAZARD_RAISED"
            elif current == "SAFE" and old_level > new_level:
                event_type = "HAZARD_CLEARED"
            elif new_level > old_level:
                event_type = "HAZARD_ESCALATED"
            else:
                event_type = "HAZARD_TRANSITION"

            events.append({
                "event_type": event_type,
                "reading_index": self.readings,
                "hazard_state": current,
                "confidence": result["confidence"],
            })

        if self.readings % self.health_interval == 0:
            events.append({
                "event_type": "PERIODIC_HEALTH",
                "reading_index": self.readings,
                "hazard_state": current,
                "confidence": result["confidence"],
            })

        self.previous = current
        self.events_generated += len(events)
        return events

    def dispatch(self, events, sink):
        for event in events:
            if sink.send(event):
                self.events_delivered += 1
            else:
                self.buffer.append(event)
                self.events_buffered += 1

    def flush(self, sink):
        if not sink.online:
            return

        while self.buffer:
            if not sink.send(self.buffer[0]):
                return
            self.buffer.popleft()
            self.events_delivered += 1


def make_persistent_stream():
    rng = np.random.default_rng(RANDOM_SEED)
    readings = []
    regime_truth = []

    for regime_name, center, count in REGIMES:
        for _ in range(count):
            reading = center + rng.normal(0.0, NOISE_STD)
            reading[0] = max(0.0, reading[0])
            reading[2] = max(0.0, reading[2])

            readings.append(tuple(reading))
            regime_truth.append(regime_name)

    return readings, regime_truth


def main():
    if not MODEL.exists():
        raise FileNotFoundError(f"Missing {MODEL.name}")

    master = load_master()

    # DataFrame path is intentional here: the optimized RF was fitted with
    # feature names, so this experiment should run without sklearn warnings.
    run = master.build_pipeline(
        model_path=str(MODEL),
        use_numpy=False,
    )

    stream, regime_truth = make_persistent_stream()

    selector = EdgeEventIntelligence(
        health_interval=HEALTH_INTERVAL,
        max_buffer=MAX_BUFFER,
    )
    cloud = CloudSink()

    observed_transitions = 0
    previous_state = None

    for index, reading in enumerate(stream):
        result = run(*reading)
        current_state = result["hazard_state"].upper()

        if previous_state is not None and current_state != previous_state:
            observed_transitions += 1

        previous_state = current_state

        events = selector.observe(result)

        # Simulated connectivity outage.
        cloud.online = not (OFFLINE_START <= index < OFFLINE_END)

        selector.dispatch(events, cloud)

        if cloud.online:
            selector.flush(cloud)

    # Recovery: reconnect and flush anything buffered during the outage.
    cloud.online = True
    selector.flush(cloud)

    transition_types = {
        "HAZARD_RAISED",
        "HAZARD_CLEARED",
        "HAZARD_ESCALATED",
        "HAZARD_TRANSITION",
    }

    captured_transitions = sum(
        event["event_type"] in transition_types
        for event in cloud.received
    )

    missed = observed_transitions - captured_transitions

    telemetry_reduction = (
        1.0 - selector.events_generated / selector.readings
    ) * 100.0

    print("=" * 78)
    print("ECOsphere -- Edge Event Intelligence | Experiment 2")
    print("=" * 78)

    print("--- Persistent temporal stream ---")
    print(f"readings processed              : {selector.readings}")
    print(f"persistent regimes              : {len(REGIMES)}")
    print(f"regime sequence                 : "
          f"{' -> '.join(name for name, _, _ in REGIMES)}")
    print(f"within-regime noise             : controlled, seed={RANDOM_SEED}")
    print()

    print("--- Selective telemetry ---")
    print(f"events generated at edge        : {selector.events_generated}")
    print(f"events delivered to cloud       : {selector.events_delivered}")
    print(f"telemetry reduction             : {telemetry_reduction:.2f}%")
    print()

    print("--- Meaningful transition capture ---")
    print(f"full-pipeline transitions       : {observed_transitions}")
    print(f"transition events captured      : {captured_transitions}")
    print(f"missed meaningful transitions   : {missed}")
    print()

    print("--- Cloud-offline resilience ---")
    print(f"offline window                  : readings "
          f"{OFFLINE_START + 1}-{OFFLINE_END}")
    print(f"events buffered during outage   : {selector.events_buffered}")
    print("edge processing during outage   : CONTINUED")
    print("buffer flushed after recovery   : YES")
    print()

    print("--- Interpretation ---")
    print(
        "The stream uses long-lived sensor regimes instead of rapid "
        "scenario alternation, making telemetry behavior more representative "
        "of slowly changing environmental conditions."
    )
    print(
        "Transition capture is checked against the same full local inference "
        "stream before telemetry filtering; this isolates event-selection "
        "loss from model-classification behavior."
    )
    print(
        "Cloud connectivity is simulated with an in-memory sink. This is "
        "feasibility evidence, not real-world temporal or network validation."
    )
    print(
        "This module does not modify the production pipeline, optimized model, "
        "decision-region cache, benchmark, or dashboard."
    )
    print("=" * 78)


if __name__ == "__main__":
    main()