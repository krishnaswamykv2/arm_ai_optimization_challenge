"""
Drift demonstration using the REAL Gas Sensor Array Drift Dataset.

This shows sensor drift is a real, measurable phenomenon using genuine
chemical sensor data (not synthetic), and that a model's accuracy changes
across batches (time periods) if drift isn't accounted for -- exactly the
problem your project's drift-correction component addresses.

Run 00_load_real_gas_data.py FIRST to produce real_gas_sensor_data.csv.
"""

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

df = pd.read_csv("real_gas_sensor_data.csv")
feature_cols = [c for c in df.columns if c.startswith("feature_")]

print("=== Demonstrating drift: train on early batches, test on later ones ===\n")

# Train on batch 1 (earliest data), test on the LAST available batch.
# If drift is real, accuracy should be noticeably lower than a normal
# same-batch train/test split -- this is the actual evidence of drift.
train_batch = df["batch"].min()
test_batch = df["batch"].max()

train_df = df[df["batch"] == train_batch]
test_df = df[df["batch"] == test_batch]

if len(train_df) < 5 or len(test_df) < 5 or train_batch == test_batch:
    print("Not enough distinct batches in this data to demonstrate drift "
          "(this mock/small dataset only has 1 batch -- with the real "
          "10-batch dataset this comparison will be meaningful).")
else:
    X_train, y_train = train_df[feature_cols], train_df["gas_class"]
    X_test, y_test = test_df[feature_cols], test_df["gas_class"]

    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)
    preds = model.predict(X_test)
    acc = accuracy_score(y_test, preds)

    print(f"Trained on batch {train_batch}, tested on batch {test_batch}")
    print(f"Cross-batch (drifted) accuracy: {acc:.3f}")
    print("(Compare this to same-batch accuracy below)")

    # Same-batch split for comparison
    X_same, y_same = train_df[feature_cols], train_df["gas_class"]
    Xtr, Xte, ytr, yte = train_test_split(X_same, y_same, test_size=0.3, random_state=42)
    model2 = RandomForestClassifier(n_estimators=100, random_state=42)
    model2.fit(Xtr, ytr)
    same_batch_acc = accuracy_score(yte, model2.predict(Xte))
    print(f"Same-batch accuracy (no drift): {same_batch_acc:.3f}")
    print(f"\nAccuracy drop due to drift: {(same_batch_acc - acc):.3f}")
