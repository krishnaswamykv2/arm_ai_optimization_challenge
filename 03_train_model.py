"""
STEP 3: Train and evaluate the hazard classification model.

We use a Random Forest: it builds many simple decision trees, each one
asking questions like "is gas_risk > 0.4?" and "is joint_risk > 0.5?",
then votes on the final answer. It's a good first model because:
  - It doesn't need much data to work reasonably well
  - It's fast to train (seconds, not hours)
  - It tells you which features mattered most (explainability, for free)
  - It runs efficiently on small edge devices like the UNO Q
"""

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix
import joblib

df = pd.read_csv("hazard_dataset_featured.csv")

# Features the model is allowed to look at (X) and the answer it must predict (y)
feature_columns = [
    "gas_ppm", "tilt_deg", "vibration_g",       # raw sensor values
    "gas_risk", "tilt_risk", "vibration_risk",   # normalized individual risks
    "structural_risk", "joint_risk"              # fused features
]
X = df[feature_columns]
y = df["label"]

# Split: 80% train, 20% test. stratify=y keeps class proportions equal in both.
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

print(f"Training on {len(X_train)} samples, testing on {len(X_test)} samples.\n")

# Train the model
model = RandomForestClassifier(
    n_estimators=100,   # number of decision trees in the "forest"
    max_depth=6,        # limits complexity to avoid overfitting on small data
    random_state=42
)
model.fit(X_train, y_train)

# Evaluate: how well does it do on data it has NEVER seen?
y_pred = model.predict(X_test)

print("=== Classification Report ===")
print(classification_report(y_test, y_pred))

print("=== Confusion Matrix ===")
print("(rows = actual class, columns = predicted class)")
labels = sorted(y.unique())
cm = confusion_matrix(y_test, y_pred, labels=labels)
print("Labels order:", labels)
print(cm)

print("\n=== Feature Importance ===")
print("(which signals the model relied on most -- higher = more influential)")
importances = pd.Series(model.feature_importances_, index=feature_columns)
print(importances.sort_values(ascending=False))

# Save the trained model so it can be loaded later for live predictions
joblib.dump(model, "hazard_model.joblib")
print("\nModel saved to hazard_model.joblib")
