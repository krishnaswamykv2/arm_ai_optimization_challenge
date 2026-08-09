"""
STEP 1: Train a small neural network (TensorFlow/Keras) to replace the
Random Forest for TFLite conversion.

WHY a new model: Random Forests (scikit-learn) cannot be converted to
TFLite -- TFLite conversion only works on TensorFlow/Keras models. This
trains a small neural network on the SAME data and SAME engineered
features (gas_risk, tilt_risk, vibration_risk, structural_risk,
joint_risk, plus raw values) so it does the same job.

The network is deliberately small (a few dense layers, ~1000 total
parameters) because it needs to run on a resource-constrained edge
device (UNO Q), not a full-size model.
"""

import pandas as pd
import numpy as np
import tensorflow as tf
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

# Load the same featured dataset used for the Random Forest
df = pd.read_csv("hazard_dataset_featured.csv")

feature_columns = [
    "gas_ppm", "tilt_deg", "vibration_g",
    "gas_risk", "tilt_risk", "vibration_risk",
    "structural_risk", "joint_risk"
]
X = df[feature_columns].values.astype("float32")

# Neural networks need numeric labels, not text -- encode safe/elevated/critical as 0/1/2
label_encoder = LabelEncoder()
y = label_encoder.fit_transform(df["label"])  # e.g. safe=2, elevated=0, critical=1 (alphabetical)
num_classes = len(label_encoder.classes_)
print(f"Classes: {list(label_encoder.classes_)} -> encoded as {list(range(num_classes))}")

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# Neural networks train better when inputs are on a similar scale.
# We normalize using TRAINING data statistics only (never test data --
# that would leak information from test set into training).
mean = X_train.mean(axis=0)
std = X_train.std(axis=0)
std[std == 0] = 1  # avoid divide-by-zero on constant columns

X_train_norm = (X_train - mean) / std
X_test_norm = (X_test - mean) / std

# Save normalization stats -- the live inference script MUST use these
# exact same numbers to normalize new readings, or predictions will be wrong.
np.save("norm_mean.npy", mean)
np.save("norm_std.npy", std)

# --- Build a small neural network ---
model = tf.keras.Sequential([
    tf.keras.layers.Input(shape=(len(feature_columns),)),
    tf.keras.layers.Dense(16, activation="relu"),
    tf.keras.layers.Dense(8, activation="relu"),
    tf.keras.layers.Dense(num_classes, activation="softmax"),
])

model.compile(
    optimizer="adam",
    loss="sparse_categorical_crossentropy",
    metrics=["accuracy"]
)

print("\n=== Model Architecture ===")
model.summary()

print("\n=== Training ===")
history = model.fit(
    X_train_norm, y_train,
    validation_data=(X_test_norm, y_test),
    epochs=30,
    batch_size=16,
    verbose=1
)

test_loss, test_acc = model.evaluate(X_test_norm, y_test, verbose=0)
print(f"\nFinal test accuracy: {test_acc:.4f}")

# Save the Keras model (before conversion) and the label encoder classes
model.save("hazard_model_keras.keras")
np.save("label_classes.npy", label_encoder.classes_)
print("\nSaved hazard_model_keras.keras and normalization/label files.")
