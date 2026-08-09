"""
STEP 3: Run inference using the TFLite model.

This is the ACTUAL code pattern that will run on the UNO Q. On a real
device you'd install 'tflite-runtime' (a lightweight package, much
smaller than full TensorFlow) instead of full TensorFlow -- but the
Interpreter API is identical, so this code doesn't change when you
move to the board.
"""

import numpy as np

try:
    import tflite_runtime.interpreter as tflite
    print("(Using tflite_runtime -- this is what runs on the actual board)")
except ImportError:
    import tensorflow as tf
    tflite = tf.lite
    print("(Using full TensorFlow's tflite module -- fine for development,")
    print(" but install 'tflite_runtime' on the actual UNO Q for a lighter footprint)")

# Load normalization stats and label classes (saved during training)
mean = np.load("norm_mean.npy")
std = np.load("norm_std.npy")
label_classes = np.load("label_classes.npy", allow_pickle=True)

# Load the TFLite model (use the standard one; swap to the quantized
# path if/when quantization actually helps for your model size)
interpreter = tflite.Interpreter(model_path="hazard_model.tflite")
interpreter.allocate_tensors()

input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

def predict_tflite(gas_ppm, tilt_deg, vibration_g):
    # Recompute the same engineered features used in training
    GAS_MAX, TILT_MAX, VIB_MAX = 800, 30, 1.0
    gas_risk = min(max(gas_ppm / GAS_MAX, 0), 1)
    tilt_risk = min(max(tilt_deg / TILT_MAX, 0), 1)
    vibration_risk = min(max(vibration_g / VIB_MAX, 0), 1)
    structural_risk = (tilt_risk + vibration_risk) / 2
    joint_risk = 0.5 * gas_risk + 0.5 * structural_risk

    raw_features = np.array([[
        gas_ppm, tilt_deg, vibration_g,
        gas_risk, tilt_risk, vibration_risk,
        structural_risk, joint_risk
    ]], dtype=np.float32)

    # Normalize using the SAME mean/std computed during training
    normalized = (raw_features - mean) / std
    normalized = normalized.astype(np.float32)

    interpreter.set_tensor(input_details[0]["index"], normalized)
    interpreter.invoke()
    output = interpreter.get_tensor(output_details[0]["index"])[0]

    predicted_index = int(np.argmax(output))
    predicted_label = label_classes[predicted_index]
    confidence = {label_classes[i]: round(float(output[i]), 3) for i in range(len(label_classes))}

    return predicted_label, confidence


if __name__ == "__main__":
    test_cases = [
        ("Normal", 75, 2, 0.04),
        ("Gas creeping up, structure fine", 480, 3, 0.05),
        ("Structure shaking, gas fine", 90, 20, 0.45),
        ("Both bad", 700, 27, 0.7),
    ]

    print("\n=== TFLite Inference Results ===")
    for description, gas, tilt, vib in test_cases:
        label, confidence = predict_tflite(gas, tilt, vib)
        print(f"\n{description}")
        print(f"  Input -> gas={gas}, tilt={tilt}, vibration={vib}")
        print(f"  Prediction: {label.upper()}")
        print(f"  Confidence: {confidence}")
