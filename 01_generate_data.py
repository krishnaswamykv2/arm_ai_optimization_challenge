"""
STEP 1: Generate synthetic training data.

Why we do this: a machine learning model learns from EXAMPLES, not rules.
Since real sensor data isn't collected yet, we simulate realistic examples
for each hazard class (safe / elevated / critical) based on approximate,
sensible sensor ranges. When real data is collected later, it replaces
this CSV — nothing else in the pipeline changes.

Sensors simulated:
  gas_ppm     -> proxy for MQ-series gas sensor reading (parts-per-million-like scale)
  tilt_deg    -> proxy for IMU-derived tilt angle in degrees
  vibration_g -> proxy for IMU-derived vibration magnitude in g-force units
"""

import numpy as np
import pandas as pd

np.random.seed(42)  # makes results reproducible every time you run this

SAMPLES_PER_CLASS = 300

def make_class_data(label, gas_center, gas_spread, tilt_center, tilt_spread,
                     vib_center, vib_spread, n=SAMPLES_PER_CLASS):
    """Generate n noisy samples around a 'typical' center point for one class."""
    gas = np.random.normal(gas_center, gas_spread, n).clip(min=0)
    tilt = np.random.normal(tilt_center, tilt_spread, n).clip(min=0)
    vib = np.random.normal(vib_center, vib_spread, n).clip(min=0)
    df = pd.DataFrame({
        "gas_ppm": gas.round(1),
        "tilt_deg": tilt.round(2),
        "vibration_g": vib.round(3),
        "label": label
    })
    return df

# --- Define what "typical" looks like for each class ---
# These numbers are reasonable placeholders. Once you have real sensor data,
# you'll replace these centers/spreads with values measured from your rover.

safe = make_class_data(
    "safe",
    gas_center=80,  gas_spread=20,     # low, stable gas reading
    tilt_center=2,  tilt_spread=1,     # rover roughly level
    vib_center=0.05, vib_spread=0.02   # minimal vibration
)

elevated = make_class_data(
    "elevated",
    gas_center=300, gas_spread=60,     # gas trending up but not extreme
    tilt_center=10, tilt_spread=3,     # noticeable tilt
    vib_center=0.25, vib_spread=0.08   # moderate shaking
)

critical = make_class_data(
    "critical",
    gas_center=650, gas_spread=100,    # high gas concentration
    tilt_center=25, tilt_spread=6,     # steep, dangerous tilt
    vib_center=0.6, vib_spread=0.15    # strong vibration/impact
)

# --- Add a few "mixed evidence" rows on purpose ---
# Real-world data is messy: sometimes gas is high but structure is fine, or
# vice-versa. Including some of these teaches the model the two signals
# aren't always in sync, which is exactly why we need FUSION (Step 3) rather
# than judging each sensor alone.
mixed_gas_only = make_class_data(
    "elevated",
    gas_center=500, gas_spread=50,     # gas alone looks bad
    tilt_center=3,  tilt_spread=1,     # structure looks fine
    vib_center=0.06, vib_spread=0.02,
    n=40
)
mixed_structural_only = make_class_data(
    "elevated",
    gas_center=90,  gas_spread=15,     # gas looks fine
    tilt_center=18, tilt_spread=3,     # structure alone looks bad
    vib_center=0.4, vib_spread=0.08,
    n=40
)

data = pd.concat(
    [safe, elevated, critical, mixed_gas_only, mixed_structural_only],
    ignore_index=True
)

# Shuffle rows so classes aren't grouped in order
data = data.sample(frac=1, random_state=42).reset_index(drop=True)

data.to_csv("hazard_dataset.csv", index=False)
print(f"Generated {len(data)} rows.")
print(data["label"].value_counts())
print("\nFirst 5 rows:")
print(data.head())
