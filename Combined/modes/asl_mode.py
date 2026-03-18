import json
import time
from collections import Counter, deque
from dataclasses import dataclass

import cv2
import mediapipe as mp
import numpy as np

from core.config import (
    ASSETS_DIR,
    ASL_CONFIDENCE_THRESHOLD,
    ASL_HOLD_SECONDS,
    ASL_RELEASE_SECONDS,
    ASL_SEND_DWELL_SECONDS,
    ASL_SEND_ZONE_RATIO,
    ASL_SMOOTH_WINDOW,
    CAMERA_FPS,
    CAMERA_HEIGHT,
    CAMERA_INDEX,
    CAMERA_WIDTH,
)


def normalize_prediction_label(label):
    if not label:
        return None
    value = str(label).strip().upper()
    if value in {"SPACE", "SPC"}:
        return "SPACE"
    if value in {"DEL", "DELETE", "BACKSPACE"}:
        return "DEL"
    if len(value) == 1 and value.isalpha():
        return value
    return value


def top_prediction_mode(predictions):
    if not predictions:
        return None
    counts = Counter(predictions)
    return counts.most_common(1)[0][0]


@dataclass
class FrameAnalysis:
    frame: np.ndarray
    hand_detected: bool
    smoothed_label: str | None
    confidence: float
    top3: list
    hand_center_x: float | None


class TFLiteSignPredictor:
    def __init__(self):
        self.model_path = ASSETS_DIR / "asl_mlp_int8.tflite"
        self.mean_path = ASSETS_DIR / "scaler_mean.npy"
        self.scale_path = ASSETS_DIR / "scaler_scale.npy"
        self.label_map_path = ASSETS_DIR / "label_map.json"

        required = [self.model_path, self.mean_path, self.scale_path, self.label_map_path]
        missing = [path.name for path in required if not path.exists()]
        if missing:
            raise FileNotFoundError("Missing ASL assets: " + ", ".join(missing))

        from tflite_runtime.interpreter import Interpreter

        self.interpreter = Interpreter(model_path=str(self.model_path))
        self.interpreter.allocate_tensors()
        self.input_details = self.interpreter.get_input_details()
        self.output_details = self.interpreter.get_output_details()
        self.scaler_mean = np.load(self.mean_path)
        self.scaler_scale = np.load(self.scale_path)
        with self.label_map_path.open("r", encoding="utf-8") as handle:
            self.label_map = json.load(handle)

    def predict(self, landmarks_63):
        x = ((landmarks_63 - self.scaler_mean) / self.scaler_scale).astype(np.float32)
        x = x.reshape(1, 63)
        self.interpreter.set_tensor(self.input_details[0]["index"], x)
        self.interpreter.invoke()
        probs = self.interpreter.get_tensor(self.output_details[0]["index"])[0]
        top_idx = int(np.argmax(probs))
        confidence = float(probs[top_idx])
        label = self.label_map[str(top_idx)]
        top3_idxs = np.argsort(probs)[-3:][::-1]
        top3 = [(self.label_map[str(int(idx))], float(probs[idx])) for idx in top3_idxs]
        return label, confidence, top3


class StableTextComposer:
    def __init__(self, confidence_threshold, hold_seconds, release_seconds):
        self.confidence_threshold = confidence_threshold
        self.hold_seconds = hold_seconds
        self.release_seconds = release_seconds
        self.buffer = ""
        self.last_sent = ""
        self.last_committed = None
        self.candidate_label = None
        self.candidate_since = None
        self.release_since = None
        self.locked_until_release = False

    def update(self, label, confidence, hand_detected, now):
        if not hand_detected or not label or confidence < self.confidence_threshold:
            self._handle_release(now)
            return {"event": None, "progress": 0.0, "candidate": None}

        self.release_since = None

        if self.locked_until_release:
            if label != self.last_committed:
                self.locked_until_release = False
            else:
                return {"event": None, "progress": 0.0, "candidate": label}

        if label != self.candidate_label:
            self.candidate_label = label
            self.candidate_since = now
            return {"event": None, "progress": 0.0, "candidate": label}

        hold_elapsed = now - self.candidate_since
        progress = min(hold_elapsed / self.hold_seconds, 1.0)
        if hold_elapsed < self.hold_seconds:
            return {"event": None, "progress": progress, "candidate": label}

        event = self._commit(label)
        self.last_committed = label
        self.locked_until_release = True
        self.candidate_label = None
        self.candidate_since = None
        return {"event": event, "progress": 1.0, "candidate": label}

    def send(self):
        sent_text = self.buffer.strip()
        self.last_sent = sent_text
        self.buffer = ""
        self.last_committed = None
        self.candidate_label = None
        self.candidate_since = None
        self.release_since = None
        self.locked_until_release = False
        return sent_text

    def _handle_release(self, now):
        self.candidate_label = None
        self.candidate_since = None
        if not self.locked_until_release:
            return
        if self.release_since is None:
            self.release_since = now
            return
        if now - self.release_since >= self.release_seconds:
            self.locked_until_release = False
            self.release_since = None

    def _commit(self, label):
        if label == "SPACE":
            if self.buffer and not self.buffer.endswith(" "):
                self.buffer += " "
            return "Space added"
        if label == "DEL":
            if self.buffer:
                self.buffer = self.buffer[:-1]
            return "Deleted last character"
        if len(label) == 1 and label.isalpha():
            self.buffer += label
            return f"Committed {label}"
        return f"Ignored unsupported label: {label}"


class ASLMode:
    def __init__(self):
        self.capture = cv2.VideoCapture(CAMERA_INDEX)
        self.capture.set(cv2.CAP_PROP_FRAME_WIDTH, CAMERA_WIDTH)
        self.capture.set(cv2.CAP_PROP_FRAME_HEIGHT, CAMERA_HEIGHT)
        self.capture.set(cv2.CAP_PROP_FPS, CAMERA_FPS)
        if not self.capture.isOpened():
            raise RuntimeError("Could not open camera for ASL mode")

        self.predictor = TFLiteSignPredictor()
        self.composer = StableTextComposer(
            confidence_threshold=ASL_CONFIDENCE_THRESHOLD,
            hold_seconds=ASL_HOLD_SECONDS,
            release_seconds=ASL_RELEASE_SECONDS,
        )
        self.pred_history = deque(maxlen=ASL_SMOOTH_WINDOW)
        self.send_dwell_started = None
        self.send_progress = 0.0
        self.message = "ASL mode active"

        self.mp_hands = mp.solutions.hands
        self.mp_draw = mp.solutions.drawing_utils
        self.mp_styles = mp.solutions.drawing_styles
        self.hands = self.mp_hands.Hands(
            static_image_mode=False,
            max_num_hands=1,
            min_detection_confidence=0.6,
            min_tracking_confidence=0.6,
            model_complexity=0,
        )

    def tick(self):
        ok, frame = self.capture.read()
        if not ok:
            raise RuntimeError("Camera frame capture failed in ASL mode")

        frame = cv2.flip(frame, 1)
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        rgb.flags.writeable = False
        results = self.hands.process(rgb)

        hand_detected = False
        confidence = 0.0
        top3 = []
        hand_center_x = None
        smoothed_label = None

        if results.multi_hand_landmarks:
            hand_detected = True
            hand_landmarks = results.multi_hand_landmarks[0]
            self.mp_draw.draw_landmarks(
                frame,
                hand_landmarks,
                self.mp_hands.HAND_CONNECTIONS,
                self.mp_styles.get_default_hand_landmarks_style(),
                self.mp_styles.get_default_hand_connections_style(),
            )

            points = np.array([[p.x, p.y, p.z] for p in hand_landmarks.landmark], dtype=np.float32)
            hand_center_x = float(points[:, 0].mean())
            pred_label, confidence, top3 = self.predictor.predict(points.flatten())
            normalized = normalize_prediction_label(pred_label)
            if normalized:
                self.pred_history.append(normalized)
                smoothed_label = top_prediction_mode(self.pred_history)
        else:
            self.pred_history.clear()

        analysis = FrameAnalysis(
            frame=frame,
            hand_detected=hand_detected,
            smoothed_label=smoothed_label,
            confidence=confidence,
            top3=top3,
            hand_center_x=hand_center_x,
        )

        now = time.perf_counter()
        in_send_zone = bool(
            analysis.hand_detected
            and analysis.hand_center_x is not None
            and analysis.hand_center_x >= 1.0 - ASL_SEND_ZONE_RATIO
            and bool(self.composer.buffer.strip())
        )

        sent_text = None
        if in_send_zone:
            self.pred_history.clear()
            self.send_progress = self._update_send_dwell(now)
            compose_result = {"event": None, "progress": 0.0, "candidate": None}
        else:
            self._reset_send_dwell()
            compose_result = self.composer.update(
                analysis.smoothed_label,
                analysis.confidence,
                analysis.hand_detected,
                now,
            )

        if compose_result["event"]:
            self.message = compose_result["event"]

        if self.send_progress >= 1.0:
            sent_text = self.composer.send()
            self._reset_send_dwell()
            self.message = f"Sent: {sent_text}" if sent_text else "Nothing to send"

        rendered = self._decorate_frame(analysis, compose_result, in_send_zone)
        return rendered, sent_text, self.message

    def _update_send_dwell(self, now):
        if self.send_dwell_started is None:
            self.send_dwell_started = now
        return min((now - self.send_dwell_started) / ASL_SEND_DWELL_SECONDS, 1.0)

    def _reset_send_dwell(self):
        self.send_dwell_started = None
        self.send_progress = 0.0

    def _decorate_frame(self, analysis, compose_result, in_send_zone):
        frame = analysis.frame.copy()
        h, w = frame.shape[:2]

        cv2.rectangle(frame, (0, 0), (w, 100), (20, 20, 20), -1)
        cv2.putText(frame, "ASL Finger Recognition Mode", (20, 35), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 255, 255), 2, cv2.LINE_AA)
        label = analysis.smoothed_label if analysis.smoothed_label else "--"
        cv2.putText(frame, f"Label: {label}", (20, 67), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (100, 230, 170), 2, cv2.LINE_AA)
        cv2.putText(frame, f"Conf: {analysis.confidence:.0%}", (230, 67), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (200, 200, 200), 2, cv2.LINE_AA)

        send_zone_start = int(w * (1.0 - ASL_SEND_ZONE_RATIO))
        zone_color = (40, 160, 80) if in_send_zone else (50, 80, 100)
        cv2.rectangle(frame, (send_zone_start, 0), (w, h), zone_color, 2)
        cv2.putText(frame, "SEND", (send_zone_start + 15, 35), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2, cv2.LINE_AA)

        bar_h = h - 140
        bar_top = 110
        bar_x1 = w - 30
        cv2.rectangle(frame, (bar_x1, bar_top), (bar_x1 + 14, bar_top + bar_h), (40, 40, 40), -1)
        if in_send_zone and self.send_progress > 0:
            fill = int(bar_h * self.send_progress)
            cv2.rectangle(frame, (bar_x1, bar_top + bar_h - fill), (bar_x1 + 14, bar_top + bar_h), (255, 180, 84), -1)

        cv2.rectangle(frame, (0, h - 110), (w, h), (20, 20, 20), -1)
        cv2.putText(frame, f"Sentence: {self.composer.buffer or '...'}", (20, h - 70), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (240, 240, 240), 2, cv2.LINE_AA)
        cv2.putText(frame, self.message, (20, h - 30), cv2.FONT_HERSHEY_SIMPLEX, 0.62, (255, 210, 120), 2, cv2.LINE_AA)

        return frame

    def close(self):
        if self.capture is not None:
            self.capture.release()
            self.capture = None
        if self.hands is not None:
            self.hands.close()
            self.hands = None
