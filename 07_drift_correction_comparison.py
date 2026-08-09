"""
Drift Correction Comparison.

Technique: per-batch standardization. Each batch's features are
normalized using THAT BATCH's own mean/std before being fed to the
model. This corrects for drift that manifests as a shift/scale change
in sensor output level, while preserving the relative pattern that
actually distinguishes one gas from another.

We compare:
  - UNCORRECTED accuracy: model trained on raw batch 1, tested on raw
    data from every other batch (same as 06_drift_curve.py)
  - CORRECTED accuracy: model trained on STANDARDIZED batch 1, tested
    on each other batch after standardizing IT using its own stats

If drift correction works, the corrected curve should sit meaningfully
higher than the uncorrected curve, especially in the batches that
dropped sharply (2 through 5).
"""

import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

df = pd.read_csv("real_gas_sensor_data.csv")
feature_cols = [c for c in df.columns if c.startswith("feature_")]

def standardize_batch(batch_df):
    """Normalize a single batch's features using its own mean/std."""
    X = batch_df[feature_cols]
    mean = X.mean()
    std = X.std().replace(0, 1)  # avoid divide-by-zero on constant columns
    return (X - mean) / std

TRAIN_BATCH = 1
all_batches = sorted(df["batch"].unique())

# --- UNCORRECTED model (raw values, same as before) ---
train_raw = df[df["batch"] == TRAIN_BATCH]
X_train_raw, y_train = train_raw[feature_cols], train_raw["gas_class"]
model_raw = RandomForestClassifier(n_estimators=100, random_state=42)
model_raw.fit(X_train_raw, y_train)

# --- CORRECTED model (standardized values) ---
train_std = standardize_batch(train_raw)
model_std = RandomForestClassifier(n_estimators=100, random_state=42)
model_std.fit(train_std, y_train)

print(f"{'Batch':<8}{'Uncorrected':<14}{'Corrected':<12}{'Improvement':<12}")
print("-" * 46)

results = []
for batch_num in all_batches:
    test_df = df[df["batch"] == batch_num]
    if len(test_df) < 5:
        continue

    y_test = test_df["gas_class"]

    # Uncorrected: raw values straight into model_raw
    X_test_raw = test_df[feature_cols]
    acc_raw = accuracy_score(y_test, model_raw.predict(X_test_raw))

    # Corrected: standardize THIS batch using its own stats, then predict
    X_test_std = standardize_batch(test_df)
    acc_std = accuracy_score(y_test, model_std.predict(X_test_std))

    improvement = acc_std - acc_raw
    marker = "  <- training batch" if batch_num == TRAIN_BATCH else ""
    print(f"{batch_num:<8}{acc_raw:<14.4f}{acc_std:<12.4f}{improvement:+.4f}{marker}")

    results.append({
        "batch": batch_num,
        "uncorrected_accuracy": round(acc_raw, 4),
        "corrected_accuracy": round(acc_std, 4),
        "improvement": round(improvement, 4)
    })

results_df = pd.DataFrame(results)
results_df.to_csv("drift_correction_comparison.csv", index=False)

print(f"\nSaved comparison to drift_correction_comparison.csv")

print("\n=== Summary ===")
non_train = results_df[results_df["batch"] != TRAIN_BATCH]
avg_raw = non_train["uncorrected_accuracy"].mean()
avg_std = non_train["corrected_accuracy"].mean()
print(f"Average uncorrected accuracy (excl. training batch): {avg_raw:.3f}")
print(f"Average corrected accuracy (excl. training batch):   {avg_std:.3f}")
print(f"Average improvement from drift correction:           {(avg_std - avg_raw):+.3f}")
