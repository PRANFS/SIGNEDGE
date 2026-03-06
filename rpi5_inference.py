"""
ASL Hand Sign Classifier Raspberry Pi 5 Inference Script
============================================================
Requires (install in order):
    pip install "numpy<2"
    pip install tflite-runtime

  IMPORTANT: tflite-runtime was compiled against NumPy 1.x.
  NumPy 2.x breaks the C-extension ABI and causes an
  'AttributeError: _ARRAY_API not found' crash at runtime.
  Always pin numpy<2 in the same venv as tflite-runtime.

Compatibility: Debian GNU/Linux 12 Bookworm, Python 3.11
Model:   INT8 quantized MLP (~few KB)
Latency: <1 ms per inference on Cortex-A76
RAM:     ~5-10 MB total (interpreter + numpy)

Place this script alongside:
  - asl_mlp_int8.tflite
  - label_map.json
  - scaler_mean.npy
  - scaler_scale.npy
"""

import numpy as np
import json
import time

# --- Use tflite_runtime for minimal footprint on RPi 5 ---
from tflite_runtime.interpreter import Interpreter

# ============================================================
# Load model and preprocessing parameters
# ============================================================
MODEL_PATH = "asl_mlp_int8.tflite"

interpreter = Interpreter(model_path=MODEL_PATH)
interpreter.allocate_tensors()
input_details  = interpreter.get_input_details()
output_details = interpreter.get_output_details()

# Load scaler params (saved as numpy arrays no sklearn needed on RPi 5)
scaler_mean  = np.load("scaler_mean.npy")
scaler_scale = np.load("scaler_scale.npy")

# Load label mapping
with open("label_map.json") as f:
    label_map = json.load(f)


def predict_sign(landmarks_63):
    """
    Predict ASL sign from 63 MediaPipe hand landmark values.

    Args:
        landmarks_63: numpy array of shape (63,)
                      flattened [x0, y0, z0, ..., x20, y20, z20]

    Returns:
        (predicted_label, confidence, top3_predictions)
    """
    # 1. Normalize using saved scaler parameters
    x = ((landmarks_63 - scaler_mean) / scaler_scale).astype(np.float32)
    x = x.reshape(1, 63)

    # 2. Run inference
    interpreter.set_tensor(input_details[0]["index"], x)
    interpreter.invoke()
    probs = interpreter.get_tensor(output_details[0]["index"])[0]

    # 3. Decode prediction
    top_idx    = int(np.argmax(probs))
    confidence = float(probs[top_idx])
    label      = label_map[str(top_idx)]

    # 4. Top-3 predictions (for "did you mean...?" UX)
    top3_idxs = np.argsort(probs)[-3:][::-1]
    top3 = [(label_map[str(int(i))], float(probs[i])) for i in top3_idxs]

    return label, confidence, top3


# ============================================================
# Quick benchmark run on RPi 5 to check latency
# ============================================================
if __name__ == "__main__":
    dummy = np.random.randn(63).astype(np.float32)
    label, conf, top3 = predict_sign(dummy)
    print(f"Test prediction: {label} ({conf:.1%})")
    print(f"Top 3: {top3}")

    # Latency benchmark (1000 iterations)
    times = []
    for _ in range(1000):
        t0 = time.perf_counter()
        predict_sign(dummy)
        times.append(time.perf_counter() - t0)

    print(f"\\nLatency over 1000 calls:")
    print(f"  Mean: {np.mean(times)*1000:.2f} ms")
    print(f"  P50:  {np.percentile(times, 50)*1000:.2f} ms")
    print(f"  P99:  {np.percentile(times, 99)*1000:.2f} ms")
