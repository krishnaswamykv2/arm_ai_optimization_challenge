"""
LAPTOP LIVE DEMO -- runs entirely on a laptop, no rover/hardware needed.

This is the demo vehicle for anyone reviewing the project before the
physical rover is available. It feeds sensor readings (a mix of
realistic test scenarios and a simulated drift sequence) through the
EXACT SAME pipeline that will run on the real hardware
(08_master_pipeline.run_pipeline), so what you see here is genuinely
representative of the model's behavior -- nothing here is faked or
mocked, only the sensor SOURCE is (real datasets / simulated drift
instead of live sensors).

For the actual hardware deployment path (real sensors -> STM32 ->
Bridge -> QRB2210 -> classification -> actuation), see
13_hardware_deployment_pipeline.py and uno_q_sketch/sketch.ino instead.
Keep these two paths separate -- this file must never import or depend
on arduino.app_utils / Bridge, so it always runs on a plain laptop.

Usage:
    python3 12_laptop_live_demo.py             # runs the full demo
    python3 12_laptop_live_demo.py --fast      # no pauses between readings
"""

import sys
import time
import random
from importlib import import_module

master = import_module("08_master_pipeline")

FAST = "--fast" in sys.argv
PAUSE = 0.0 if FAST else 0.4


def show_reading(step_num, label, gas, tilt, vib):
    result = master.run_pipeline(gas, tilt, vib)
    raw = result["raw_input"]
    corrected = result["corrected_input"]
    hazard = result["hazard_state"]
    conf = result["confidence"]

    hazard_marker = {"SAFE": "  [SAFE]  ", "ELEVATED": " [ELEVATED]", "CRITICAL": " [CRITICAL]"}
    marker = hazard_marker.get(hazard, hazard)

    print(f"[{step_num:>3}] {label:<28} "
          f"raw(gas={raw['gas_ppm']:.0f}, tilt={raw['tilt_deg']:.0f}, vib={raw['vibration_g']:.2f}) "
          f"-> corrected(gas={corrected['gas_ppm']:.0f}) "
          f"-> {marker}  {conf}")
    time.sleep(PAUSE)
    return hazard


def section(title):
    print(f"\n{'=' * 70}\n{title}\n{'=' * 70}")


def main():
    print("EcoSphere -- Laptop Live Demo")
    print("(Runs the real classification pipeline. No hardware required.)")

    # --- Section 1: fixed scenarios, no drift, to show baseline behavior ---
    section("SECTION 1 -- Baseline scenarios (no drift)")
    scenarios = [
        ("Normal conditions", 75, 2, 0.04),
        ("Gas creeping up, structure fine", 480, 3, 0.05),
        ("Structure shaking, gas fine", 90, 20, 0.45),
        ("Both bad", 700, 27, 0.7),
    ]
    step = 0
    for label, gas, tilt, vib in scenarios:
        step += 1
        show_reading(step, label, gas, tilt, vib)

    # --- Section 2: calibration period for the drift tracker ---
    section("SECTION 2 -- Calibration (rover assumed to start in a safe state)")
    print("Feeding steady safe-range readings so the drift tracker can")
    print("establish its reference baseline (needs 15 readings)...")
    for i in range(15):
        step += 1
        gas = 75 + random.uniform(-3, 3)
        show_reading(step, f"Calibration reading {i+1}/15", gas, 2, 0.04)

    # --- Section 3: simulated drift, uncorrected effect visible ---
    section("SECTION 3 -- Simulated sensor drift (watch corrected vs raw diverge)")
    print("Gas sensor baseline is drifting upward over time (simulated).")
    print("Watch 'raw' climb while 'corrected' stays closer to the true value.\n")
    for i in range(40):
        step += 1
        drift_amount = i * 5  # accelerating drift
        noise = random.uniform(-3, 3)
        gas = 75 + drift_amount + noise
        show_reading(step, f"Drift step {i+1}/40", gas, 2, 0.04)

    section("Demo complete")
    print("Summary: the drift-correction layer measurably reduces the gap")
    print("between raw and corrected readings under sustained drift, though")
    print("(as documented) it reduces error by roughly half, not to zero --")
    print("see REAL_DATA_CLAIMS.md for the honest, full characterization.")


if __name__ == "__main__":
    main()
