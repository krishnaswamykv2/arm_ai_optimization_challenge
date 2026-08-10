"""
STEP 2: Convert the trained Keras model to TensorFlow Lite (.tflite).

Two versions are produced:
  1. hazard_model.tflite          -- standard float32 conversion
  2. hazard_model_quantized.tflite -- int8 quantized (smaller, faster,
     the version that actually matters for the Arm Optimization
     Challenge, since quantization is one of the standard "prove you
     optimized for Arm" techniques)

Quantization needs a small "representative dataset" -- a sample of
real-ish input data -- so the converter knows the actual range of
values each feature takes, and can map float32 -> int8 accurately.
"""

import tensorflow as tf
import numpy as np
import pandas as pd
import os

model = tf.keras.models.load_model("hazard_model_keras.keras")
mean = np.load("norm_mean.npy")
std = np.load("norm_std.npy")

feature_columns = [
    "gas_ppm", "tilt_deg", "vibration_g",
    "gas_risk", "tilt_risk", "vibration_risk",
    "structural_risk", "joint_risk"
]

# --- Conversion 1: standard float32 TFLite model ---
converter = tf.lite.TFLiteConverter.from_keras_model(model)
tflite_model = converter.convert()

with open("hazard_model.tflite", "wb") as f:
    f.write(tflite_model)

float_size_kb = os.path.getsize("hazard_model.tflite") / 1024
print(f"Standard TFLite model saved: hazard_model.tflite ({float_size_kb:.1f} KB)")


# --- Conversion 2: quantized (int8) TFLite model ---
df = pd.read_csv("hazard_dataset_featured.csv")
X = df[feature_columns].values.astype("float32")
X_norm = (X - mean) / std

def representative_dataset():
    # Feed a sample of real (normalized) data so the converter learns
    # the actual value ranges for accurate int8 quantization
    for i in range(min(200, len(X_norm))):
        yield [X_norm[i:i+1].astype(np.float32)]

converter_quant = tf.lite.TFLiteConverter.from_keras_model(model)
converter_quant.optimizations = [tf.lite.Optimize.DEFAULT]
converter_quant.representative_dataset = representative_dataset
converter_quant.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
converter_quant.inference_input_type = tf.int8
converter_quant.inference_output_type = tf.int8

tflite_quant_model = converter_quant.convert()

with open("hazard_model_quantized.tflite", "wb") as f:
    f.write(tflite_quant_model)

quant_size_kb = os.path.getsize("hazard_model_quantized.tflite") / 1024
print(f"Quantized TFLite model saved: hazard_model_quantized.tflite ({quant_size_kb:.1f} KB)")

size_change_pct = (quant_size_kb / float_size_kb - 1) * 100
print(f"\nFloat32 baseline: {float_size_kb:.2f} KB  |  INT8 quantized: {quant_size_kb:.2f} KB")
if size_change_pct > 0:
    print(f"Quantization INCREASED size by {size_change_pct:.1f}%.")
    print("This model has ~307 parameters -- too small for INT8 quantization to pay off.")
    print("Per-tensor scale/zero-point metadata overhead exceeds the weight-storage")
    print("savings at this scale. Dynamic-range (weights-only) quantization was also")
    print("tested and produced 0% size change for the same reason.")
    print("\nAt this model size, the honest optimization claim is NOT file size --")
    print("it's inference LATENCY, which must be measured on real Arm hardware")
    print("(QRB2210), not estimated from file size. INT8 ops can be faster on ARM")
    print("even when the file itself isn't smaller. Run 11_tflite_inference.py")
    print("on-device and time it to get that number.")
else:
    print(f"Quantization reduced size by {abs(size_change_pct):.1f}%.")
