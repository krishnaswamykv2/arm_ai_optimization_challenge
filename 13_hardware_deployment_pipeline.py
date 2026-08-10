"""
LIVE ROVER MODE -- runs continuously on the QRB2210 (Linux side of the
UNO Q), reading real sensors from the STM32U585 and sending hazard
decisions back for actuation.

THIS FILE CANNOT BE TESTED OFF-DEVICE. `arduino.app_utils` (the Bridge
RPC library) only exists on the actual UNO Q runtime. Everything in
08_master_pipeline.py has been run and verified; this file has only
been reasoned through, not executed, until it runs on real hardware.
Treat first real run as a test, not a demo.

HOW THIS CONNECTS TO THE STM32 SKETCH (uno_q_sketch/sketch.ino):
  This script calls three RPC methods that the sketch must register
  with Bridge.provide() / Bridge.provide_safe():
    - "get_gas_reading"        -> returns float (gas_ppm)
    - "get_tilt_reading"       -> returns float (tilt_deg)
    - "get_vibration_reading"  -> returns float (vibration_g)
  And calls one RPC method the sketch must register to receive:
    - "set_hazard_response"    <- sends string ("SAFE"/"ELEVATED"/"CRITICAL")

  The exact sensor->value conversion (analog vs I2C, calibration
  constants, ADC scaling) lives entirely in the sketch -- this script
  doesn't need to know it, which is why it doesn't need pin numbers.
  See HARDWARE_REQUIREMENTS.md for exactly what the sketch needs to
  provide.

SAFETY NOTE: if a Bridge call fails or times out (sensor disconnected,
MCU not responding), this script does NOT guess a value -- it skips
that reading cycle and logs the failure. Silently substituting a
default reading on a hazard-classification system would be worse than
no reading at all.
"""

import time
from arduino.app_utils import Bridge, App
from importlib import import_module

master = import_module("08_master_pipeline")

READ_INTERVAL_SECONDS = 1.0  # how often to sample and classify
BRIDGE_TIMEOUT_SECONDS = 2.0


def read_sensors():
    """
    Pull one reading from each sensor via the STM32 Bridge RPC.
    Returns None if any read fails -- caller must handle that, never
    substitute a fabricated value for a real sensor failure.
    """
    try:
        gas_ppm = Bridge.call("get_gas_reading", timeout=BRIDGE_TIMEOUT_SECONDS)
        tilt_deg = Bridge.call("get_tilt_reading", timeout=BRIDGE_TIMEOUT_SECONDS)
        vibration_g = Bridge.call("get_vibration_reading", timeout=BRIDGE_TIMEOUT_SECONDS)
        return float(gas_ppm), float(tilt_deg), float(vibration_g)
    except Exception as e:
        print(f"[SENSOR READ FAILED] {e} -- skipping this cycle")
        return None


def send_hazard_response(hazard_state):
    """
    Push the classified hazard state to the STM32 for actuation
    (LED/buzzer/motor-stop -- whatever the sketch implements).
    """
    try:
        Bridge.call("set_hazard_response", hazard_state, timeout=BRIDGE_TIMEOUT_SECONDS)
    except Exception as e:
        print(f"[ACTUATION SEND FAILED] {e}")


def loop():
    reading = read_sensors()
    if reading is None:
        time.sleep(READ_INTERVAL_SECONDS)
        return

    gas_ppm, tilt_deg, vibration_g = reading
    result = master.run_pipeline(gas_ppm, tilt_deg, vibration_g)

    print(
        f"raw={result['raw_input']}  "
        f"corrected={result['corrected_input']}  "
        f"hazard={result['hazard_state']}  "
        f"confidence={result['confidence']}"
    )

    send_hazard_response(result["hazard_state"])
    time.sleep(READ_INTERVAL_SECONDS)


if __name__ == "__main__":
    print("EcoSphere live rover mode starting...")
    print(f"Sampling every {READ_INTERVAL_SECONDS}s. Ctrl+C to stop.")
    App.run(user_loop=loop)
