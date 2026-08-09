"""
STEP 4: Simulate a live prediction, the way it will run on the UNO Q.

On the real rover: the STM32 side reads gas + IMU sensors and sends the
raw values to the QRB2210 (Linux) side. This script mimics that hand-off:
you get raw sensor values in, and get a hazard classification out.
"""

import pandas as pd
import joblib
from feature_engineering_helper import engineer_single_reading  # see below

model = joblib.load("hazard_model.joblib")

def predict(gas_ppm, tilt_deg, vibration_g):
    row = engineer_single_reading(gas_ppm, tilt_deg, vibration_g)
    X = pd.DataFrame([row])
    prediction = model.predict(X)[0]
    probabilities = model.predict_proba(X)[0]
    class_names = model.classes_
    prob_dict = dict(zip(class_names, probabilities.round(3)))
    return prediction, prob_dict


if __name__ == "__main__":
    test_cases = [
        ("Looks totally normal", 75, 2, 0.04),
        ("Gas creeping up, structure fine", 480, 3, 0.05),
        ("Structure shaking, gas fine", 90, 20, 0.45),
        ("Both signals bad -- real danger", 700, 27, 0.7),
    ]

    for description, gas, tilt, vib in test_cases:
        pred, probs = predict(gas, tilt, vib)
        print(f"{description}")
        print(f"  Input -> gas={gas}ppm, tilt={tilt}deg, vibration={vib}g")
        print(f"  Prediction: {pred.upper()}")
        print(f"  Confidence: {probs}\n")
