"""
ASL Hand Sign Real-Time Demo  Raspberry Pi 5
==============================================
Live camera feed + MediaPipe landmarks + TFLite MLP inference.

Install dependencies (same venv as rpi5_inference.py):
    pip install "numpy<2"
    pip install tflite-runtime
    pip install mediapipe opencv-python

Run:
    python demo.py

Controls:
    Q   quit
    S   save current frame as screenshot
"""

# Force matplotlib to use a non-interactive backend BEFORE mediapipe (or
# anything else) can import it.  On a headless Raspberry Pi, matplotlib
# otherwise tries to connect to a display server (TkAgg / Qt5Agg), which
# either hangs for tens of seconds or raises an error.
import os
os.environ.setdefault("MPLBACKEND", "Agg")

print("Loading libraries – please wait...", flush=True)

import cv2
import mediapipe as mp
import numpy as np
import time
from collections import deque

# Import predict_sign from rpi5_inference.py (loads model + scaler at import time)
# Both files must be in the same directory.
from rpi5_inference import predict_sign

# ============================================================
# MediaPipe Hands setup
# Must match the settings used during training data collection
# ============================================================
mp_hands       = mp.solutions.hands
mp_draw        = mp.solutions.drawing_utils
mp_draw_styles = mp.solutions.drawing_styles

hands = mp_hands.Hands(
    static_image_mode=False,      # Tracking mode (faster than per-frame detection)
    max_num_hands=1,              # Only need one hand for ASL signs
    min_detection_confidence=0.6, # Matches landmark extraction settings
    min_tracking_confidence=0.6,  # Matches landmark extraction settings
    model_complexity=0            # Lite model  fastest on Cortex-A76
)

# ============================================================
# Temporal smoothing
# Raw predictions flicker (same sign, slightly different landmark
# position each frame). Majority vote over last N frames gives a
# stable display without adding latency to the actual inference.
# ============================================================
SMOOTH_WINDOW = 7
pred_history  = deque(maxlen=SMOOTH_WINDOW)

# ============================================================
# Camera setup
# ============================================================
cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FRAME_WIDTH,  640)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
cap.set(cv2.CAP_PROP_FPS,          30)

if not cap.isOpened():
    raise RuntimeError(
        "Could not open camera. Check that /dev/video0 exists and run:\n"
        "  sudo usermod -aG video $USER  (then log out and back in)"
    )

# ============================================================
# FPS tracking
# ============================================================
fps_start   = time.perf_counter()
fps_counter = 0
fps_display = 0.0

print("Demo running  press Q to quit, S to save screenshot.")

screenshot_idx = 0

while True:
    ret, frame = cap.read()
    if not ret:
        print("Frame capture failed  check camera connection.")
        break

    # Mirror horizontally so hand movement matches visual feedback
    frame = cv2.flip(frame, 1)
    h, w  = frame.shape[:2]

    # --------------------------------------------------------
    # Hand landmark detection
    # --------------------------------------------------------
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    rgb.flags.writeable = False        # Avoid unnecessary copy in MediaPipe
    results = hands.process(rgb)
    rgb.flags.writeable = True

    label_display = "--"
    confidence    = 0.0
    top3          = []
    hand_detected = False
    
    if results.multi_hand_landmarks:
        hand_detected = True
        lm_set = results.multi_hand_landmarks[0]

        # Draw skeleton (wrist + finger joints + connections)
        mp_draw.draw_landmarks(
            frame, lm_set,
            mp_hands.HAND_CONNECTIONS,
            mp_draw_styles.get_default_hand_landmarks_style(),
            mp_draw_styles.get_default_hand_connections_style()
        )

        # Build 63-feature vector: [x0,y0,z0, x1,y1,z1, ..., x20,y20,z20]
        # Order matches the training CSV column layout exactly.
        landmarks_63 = np.array(
            [[pt.x, pt.y, pt.z] for pt in lm_set.landmark],
            dtype=np.float32
        ).flatten()  # shape: (63,)

        # Run TFLite inference (normalisation happens inside predict_sign)
        pred_label, confidence, top3 = predict_sign(landmarks_63)

        # Temporal smoothing: majority vote across recent frames
        pred_history.append(pred_label)
        label_display = max(set(pred_history), key=pred_history.count)

    # --------------------------------------------------------
    # FPS calculation (updated every 20 frames)
    # --------------------------------------------------------
    fps_counter += 1
    if fps_counter >= 20:
        elapsed     = time.perf_counter() - fps_start
        fps_display = fps_counter / elapsed
        fps_start   = time.perf_counter()
        fps_counter = 0

    # ============================================================
    # HUD Overlay
    # ============================================================
    
    # --- Top panel: semi-transparent black background ---
    panel = frame.copy()
    cv2.rectangle(panel, (0, 0), (w, 95), (0, 0, 0), -1)
    frame = cv2.addWeighted(panel, 0.55, frame, 0.45, 0)

    # --- Big predicted letter ---
    cv2.putText(frame, label_display, (18, 78),
                cv2.FONT_HERSHEY_SIMPLEX, 2.8,
                (0, 255, 110) if hand_detected else (120, 120, 120), 3,
                cv2.LINE_AA)

    if hand_detected:
        # --- Confidence percentage ---
        cv2.putText(frame, f"{confidence:.0%}", (185, 72),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.6, (255, 255, 255), 2,
                    cv2.LINE_AA)

        # --- Confidence bar ---
        bx, by, bw, bh = 18, 82, 340, 9
        cv2.rectangle(frame, (bx, by), (bx + bw, by + bh), (50, 50, 50), -1)
        bar_color = (0, 210, 0) if confidence >= 0.70 else (0, 160, 210)
        cv2.rectangle(frame, (bx, by),
                      (bx + int(bw * confidence), by + bh),
                      bar_color, -1)

        # --- Top-3 panel at bottom ---
        bottom_panel = frame.copy()
        cv2.rectangle(bottom_panel, (0, h - 55), (w, h), (0, 0, 0), -1)
        frame = cv2.addWeighted(bottom_panel, 0.6, frame, 0.4, 0)

        cv2.putText(frame, "Top 3:", (14, h - 33),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, (160, 160, 160), 1,
                    cv2.LINE_AA)

        x_offset = 80
        for rank, (lbl, prob) in enumerate(top3):
            color = (0, 255, 110) if rank == 0 else (180, 180, 180)
            txt   = f"  {lbl}  {prob:.0%}"
            cv2.putText(frame, txt, (x_offset + rank * 185, h - 33),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.65, color, 2, cv2.LINE_AA)

        # --- Top-3 mini bars ---
        for rank, (_, prob) in enumerate(top3):
            mini_x = x_offset + rank * 185
            bar_color_mini = (0, 200, 80) if rank == 0 else (100, 100, 200)
            cv2.rectangle(frame, (mini_x, h - 22),
                          (mini_x + int(160 * prob), h - 14),
                          bar_color_mini, -1)

    # --- Hand detection status badge ---
    badge_color = (0, 200, 0) if hand_detected else (0, 50, 200)
    badge_text  = "HAND DETECTED" if hand_detected else "NO HAND"
    cv2.rectangle(frame, (w - 230, 8), (w - 8, 38), badge_color, -1)
    cv2.putText(frame, badge_text, (w - 222, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.62, (255, 255, 255), 2,
                cv2.LINE_AA)

    # --- FPS ---
    cv2.putText(frame, f"FPS {fps_display:.0f}", (w - 100, 60),
                cv2.FONT_HERSHEY_SIMPLEX, 0.55, (180, 180, 180), 1,
                cv2.LINE_AA)

    # ============================================================
    # Show frame
    # ============================================================
    cv2.imshow("ASL Sign Recognition  |  RPi 5  |  press Q to quit", frame)

    key = cv2.waitKey(1) & 0xFF
    if key == ord('q'):
        break
    elif key == ord('s'):
        fname = f"screenshot_{screenshot_idx:03d}.jpg"
        cv2.imwrite(fname, frame)
        print(f"Saved {fname}")
        screenshot_idx += 1

# ============================================================
# Cleanup
# ============================================================
cap.release()
hands.close()
cv2.destroyAllWindows()
print("Demo closed.")
