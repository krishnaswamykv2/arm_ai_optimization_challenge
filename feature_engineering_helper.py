"""
Shared helper so the SAME feature engineering logic used during training
(Step 2) is used during live prediction (Step 4). This consistency matters:
if training and live prediction compute features differently, the model's
learned patterns won't apply correctly.
"""

GAS_MAX = 800
TILT_MAX = 30
VIB_MAX = 1.0

def engineer_single_reading(gas_ppm, tilt_deg, vibration_g):
    gas_risk = min(max(gas_ppm / GAS_MAX, 0), 1)
    tilt_risk = min(max(tilt_deg / TILT_MAX, 0), 1)
    vibration_risk = min(max(vibration_g / VIB_MAX, 0), 1)
    structural_risk = (tilt_risk + vibration_risk) / 2
    joint_risk = 0.5 * gas_risk + 0.5 * structural_risk

    return {
        "gas_ppm": gas_ppm,
        "tilt_deg": tilt_deg,
        "vibration_g": vibration_g,
        "gas_risk": gas_risk,
        "tilt_risk": tilt_risk,
        "vibration_risk": vibration_risk,
        "structural_risk": structural_risk,
        "joint_risk": joint_risk,
    }
