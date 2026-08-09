"""
Drift Curve: train on batch 1, test against EVERY other batch (2-10),
to see how accuracy degrades progressively over time -- not just a
single before/after snapshot.

Run 00_load_real_gas_data.py FIRST to produce real_gas_sensor_data.csv.
"""

import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

df = pd.read_csv("real_gas_sensor_data.csv")
feature_cols = [c for c in df.columns if c.startswith("feature_")]

TRAIN_BATCH = 1

train_df = df[df["batch"] == TRAIN_BATCH]
if len(train_df) < 5:
    raise ValueError(f"Not enough data in batch {TRAIN_BATCH} to train. Check your CSV.")

X_train, y_train = train_df[feature_cols], train_df["gas_class"]

print(f"Training once on batch {TRAIN_BATCH} ({len(train_df)} samples)...\n")
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

all_batches = sorted(df["batch"].unique())
results = []

print(f"{'Batch':<8}{'Samples':<10}{'Accuracy':<10}")
print("-" * 28)

for batch_num in all_batches:
    test_df = df[df["batch"] == batch_num]
    if len(test_df) < 5:
        continue

    X_test, y_test = test_df[feature_cols], test_df["gas_class"]
    preds = model.predict(X_test)
    acc = accuracy_score(y_test, preds)

    results.append({"batch": batch_num, "samples": len(test_df), "accuracy": round(acc, 4)})
    marker = "  <- training batch" if batch_num == TRAIN_BATCH else ""
    print(f"{batch_num:<8}{len(test_df):<10}{acc:<10.4f}{marker}")

results_df = pd.DataFrame(results)
results_df.to_csv("drift_curve_results.csv", index=False)

print(f"\nSaved full results to drift_curve_results.csv")

# Simple text-based summary of the trend
print("\n=== Summary ===")
start_acc = results_df.iloc[0]["accuracy"]
end_acc = results_df.iloc[-1]["accuracy"]
print(f"Accuracy at batch {int(results_df.iloc[0]['batch'])}: {start_acc:.3f}")
print(f"Accuracy at batch {int(results_df.iloc[-1]['batch'])}: {end_acc:.3f}")
print(f"Total drop across full range: {(start_acc - end_acc):.3f}")

# Find first batch where accuracy drops below common usability thresholds
for threshold in [0.9, 0.8, 0.7, 0.5]:
    below = results_df[results_df["accuracy"] < threshold]
    if not below.empty:
        first_batch_below = int(below.iloc[0]["batch"])
        print(f"First batch below {int(threshold*100)}% accuracy: batch {first_batch_below}")
    else:
        print(f"Accuracy never drops below {int(threshold*100)}% in this range.")
