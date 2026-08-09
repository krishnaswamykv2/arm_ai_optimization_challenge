"""
STEP 2: Feature engineering + fusion.

Raw sensor values (gas_ppm, tilt_deg, vibration_g) are on very different
scales (0-700 vs 0-30 vs 0-1). We:
  1. Normalize each to a comparable 0-1 "risk" scale
  2. Combine (fuse) the normalized structural signals (tilt + vibration)
     into one structural_risk score
  3. Create a joint_risk score = a weighted combination of gas_risk and
     structural_risk. This is the actual FUSION step: it captures cases
     where the two signal types together imply more danger than either
     alone.

The model will be trained on BOTH the raw values AND these engineered
fusion features, so it can use whichever signals turn out to matter most.
"""

import pandas as pd

def engineer_features(df):
    df = df.copy()

    # Normalize each raw sensor to roughly 0-1 using sensible max reference points.
    # These reference maxima are placeholders -- replace with real sensor
    # datasheet max/danger values once you have them.
    GAS_MAX = 800
    TILT_MAX = 30
    VIB_MAX = 1.0

    df["gas_risk"] = (df["gas_ppm"] / GAS_MAX).clip(0, 1)
    df["tilt_risk"] = (df["tilt_deg"] / TILT_MAX).clip(0, 1)
    df["vibration_risk"] = (df["vibration_g"] / VIB_MAX).clip(0, 1)

    # Fuse the two structural signals into one structural_risk score
    df["structural_risk"] = (df["tilt_risk"] + df["vibration_risk"]) / 2

    # FUSION: combine gas evidence + structural evidence into one joint score.
    # Weighted 50/50 here as a starting point -- can be tuned later once
    # real data shows which signal is a stronger/earlier predictor.
    df["joint_risk"] = 0.5 * df["gas_risk"] + 0.5 * df["structural_risk"]

    return df


if __name__ == "__main__":
    raw = pd.read_csv("hazard_dataset.csv")
    featured = engineer_features(raw)
    featured.to_csv("hazard_dataset_featured.csv", index=False)
    print("Engineered features added. Preview:")
    print(featured.head())
