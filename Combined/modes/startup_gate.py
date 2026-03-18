import time

import cv2
import mediapipe as mp
import numpy as np

from core.config import BOOT_GATE_SECONDS, CAMERA_FPS, CAMERA_HEIGHT, CAMERA_INDEX, CAMERA_WIDTH

class StartupGate:
    HAND_HOLD_SECONDS = 1.5

    def __init__(self):
        self.start_time = time.perf_counter()
        self.capture = cv2.VideoCapture(CAMERA_INDEX)
        self.capture.set(cv2.CAP_PROP_FRAME_WIDTH, CAMERA_WIDTH)
        self.capture.set(cv2.CAP_PROP_FRAME_HEIGHT, CAMERA_HEIGHT)
        self.capture.set(cv2.CAP_PROP_FPS, CAMERA_FPS)

        self.mp_hands = mp.solutions.hands
        self.hands = self.mp_hands.Hands(
            static_image_mode=False,
            max_num_hands=1,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5,
            model_complexity=0,
        )

        self.hand_start_time = None

    def tick(self):
        ok, frame = self.capture.read()
        if not ok:
            frame = np.zeros((CAMERA_HEIGHT, CAMERA_WIDTH, 3), dtype=np.uint8)

        frame = cv2.flip(frame, 1)
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = self.hands.process(rgb)

        elapsed = time.perf_counter() - self.start_time
        remaining = max(0.0, BOOT_GATE_SECONDS - elapsed)

        hand_detected = bool(results.multi_hand_landmarks)

        # Track how long the hand is held up
        now = time.perf_counter()
        if hand_detected:
            if self.hand_start_time is None:
                self.hand_start_time = now
            hand_held_duration = now - self.hand_start_time
        else:
            self.hand_start_time = None
            hand_held_duration = 0.0

        # Only allow ASL mode if hand held for required duration
        select_asl = hand_held_duration >= self.HAND_HOLD_SECONDS

        if select_asl:
            status = f"Hand held up for {hand_held_duration:.1f}s. Switching to ASL mode"
        elif hand_detected:
            status = f"Hand detected. Hold for {self.HAND_HOLD_SECONDS - hand_held_duration:.1f}s more to switch to ASL mode"
        elif remaining <= 0:
            status = "No hand detected. Starting eye tracking mode"
        else:
            status = "Raise your hand now to switch to ASL mode"

        cv2.rectangle(frame, (20, 20), (frame.shape[1] - 20, 150), (30, 30, 30), -1)
        cv2.putText(frame, "Startup Mode Selection", (40, 65), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 255), 2, cv2.LINE_AA)
        cv2.putText(frame, f"Countdown: {remaining:0.1f}s", (40, 100), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (100, 220, 255), 2, cv2.LINE_AA)
        cv2.putText(frame, status, (40, 132), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (180, 240, 180), 2, cv2.LINE_AA)

        timeout = remaining <= 0.0
        return frame, select_asl, timeout

    def close(self):
        if self.capture is not None:
            self.capture.release()
            self.capture = None
        if self.hands is not None:
            self.hands.close()
            self.hands = None
