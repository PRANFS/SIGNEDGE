import math
import os
import queue
import threading
import time
from collections import deque

import cv2
import mediapipe as mp
import numpy as np

from core.config import (
    CAMERA_FPS,
    CAMERA_HEIGHT,
    CAMERA_INDEX,
    CAMERA_WIDTH,
    EYE_CURSOR_ALPHA,
    EYE_DWELL_TIME,
    EYE_EAR_THRESHOLD,
    EYE_FREQUENCIES_FILE,
    EYE_LLM_MODEL,
    EYE_LLM_MODEL_PATH,
    EYE_SMOOTHING_FRAMES,
)


class GazeTracker:
    def __init__(self, camera_index=0, smoothing_frames=20, ear_threshold=0.2, dwell_time=0.6):
        self.cap = cv2.VideoCapture(camera_index)
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, CAMERA_WIDTH)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, CAMERA_HEIGHT)
        self.cap.set(cv2.CAP_PROP_FPS, CAMERA_FPS)

        if not self.cap.isOpened():
            raise RuntimeError("Could not open camera for eye mode")

        self.mp_face_mesh = mp.solutions.face_mesh
        self.face_mesh = self.mp_face_mesh.FaceMesh(
            max_num_faces=1,
            refine_landmarks=True,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5,
        )
        self.ear_threshold = ear_threshold
        self.dwell_time = dwell_time
        self.blink_start_time = None
        self.x_history = deque(maxlen=smoothing_frames)
        self.y_history = deque(maxlen=smoothing_frames)

    def _calculate_ear(self, landmarks, indices):
        v1 = np.linalg.norm(
            np.array([landmarks[indices[1]].x, landmarks[indices[1]].y])
            - np.array([landmarks[indices[5]].x, landmarks[indices[5]].y])
        )
        v2 = np.linalg.norm(
            np.array([landmarks[indices[2]].x, landmarks[indices[2]].y])
            - np.array([landmarks[indices[4]].x, landmarks[indices[4]].y])
        )
        h = np.linalg.norm(
            np.array([landmarks[indices[0]].x, landmarks[indices[0]].y])
            - np.array([landmarks[indices[3]].x, landmarks[indices[3]].y])
        )
        return (v1 + v2) / (2.0 * max(h, 1e-6))

    def process(self):
        success, frame = self.cap.read()
        if not success:
            return np.zeros((CAMERA_HEIGHT, CAMERA_WIDTH, 3), dtype=np.uint8), None, None, False

        frame = cv2.flip(frame, 1)
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        rgb.flags.writeable = False
        results = self.face_mesh.process(rgb)

        x_center, y_center, is_clicking = None, None, False
        if results.multi_face_landmarks:
            landmarks = results.multi_face_landmarks[0].landmark

            left_ear = self._calculate_ear(landmarks, [33, 160, 158, 133, 153, 144])
            right_ear = self._calculate_ear(landmarks, [362, 385, 387, 263, 373, 380])
            is_blinking = ((left_ear + right_ear) / 2.0) < self.ear_threshold

            left_iris = [landmarks[i] for i in range(474, 478)]
            right_iris = [landmarks[i] for i in range(469, 473)]
            x_raw = np.mean([lm.x for lm in left_iris + right_iris])
            y_raw = np.mean([lm.y for lm in left_iris + right_iris])

            if is_blinking:
                if len(self.x_history) > 0:
                    x_center = sum(self.x_history) / len(self.x_history)
                    y_center = sum(self.y_history) / len(self.y_history)
                else:
                    x_center, y_center = x_raw, y_raw

                if self.blink_start_time is None:
                    self.blink_start_time = time.time()
                elif (time.time() - self.blink_start_time) >= self.dwell_time:
                    is_clicking = True
                    self.blink_start_time = time.time() + 1.2
            else:
                self.blink_start_time = None
                self.x_history.append(x_raw)
                self.y_history.append(y_raw)
                x_center = sum(self.x_history) / len(self.x_history)
                y_center = sum(self.y_history) / len(self.y_history)

        return frame, x_center, y_center, is_clicking

    def close(self):
        if self.cap is not None:
            self.cap.release()
            self.cap = None
        if self.face_mesh is not None:
            self.face_mesh.close()
            self.face_mesh = None


class LLMPredictor:
    def __init__(self, model_path):
        self.enabled = False
        self.request_queue = queue.Queue()
        self.result_queue = queue.Queue()

        try:
            from llama_cpp import Llama

            if not os.path.exists(model_path):
                return
            self.llm = Llama(model_path=model_path, n_ctx=256, n_threads=2, verbose=False)
            self.enabled = True
            self.worker_thread = threading.Thread(target=self._worker, daemon=True)
            self.worker_thread.start()
        except Exception:
            self.enabled = False

    def _worker(self):
        while True:
            sentence = self.request_queue.get()
            if sentence is None:
                break
            prompt = (
                "<|system|>\n"
                "You are an AAC typing assistant. Provide ONLY the next 1 to 2 words to complete the sentence."
                "No punctuation. No explanation.</s>\n<|user|>\n"
                f"Complete this: {sentence}</s>\n<|assistant|>\n"
            )
            try:
                output = self.llm(prompt, max_tokens=3, stop=["\n", ".", ","], temperature=0.2)
                prediction = output["choices"][0]["text"].strip().upper().replace('"', "").replace("'", "")
                if prediction:
                    self.result_queue.put(prediction)
            except Exception:
                continue

    def request_next_words(self, current_text):
        if not self.enabled:
            return
        while not self.request_queue.empty():
            try:
                self.request_queue.get_nowait()
            except queue.Empty:
                break
        self.request_queue.put(current_text)

    def check_for_results(self):
        if not self.enabled:
            return None
        if not self.result_queue.empty():
            return self.result_queue.get()
        return None


# ---------------------------------------------------------------------------
# Trie
# ---------------------------------------------------------------------------

class TrieNode:
    __slots__ = ("children", "is_end_of_word", "frequency")

    def __init__(self):
        self.children: dict = {}
        self.is_end_of_word = False
        self.frequency = 0


class FrequencyTrie:
    def __init__(self, dictionary_file):
        self.root = TrieNode()
        self.load_dictionary(dictionary_file)

    def insert(self, word, frequency):
        node = self.root
        for char in word:
            if char not in node.children:
                node.children[char] = TrieNode()
            node = node.children[char]
        node.is_end_of_word = True
        node.frequency = frequency

    def load_dictionary(self, filepath):
        if not os.path.exists(filepath):
            defaults = [("HELLO", 100), ("HELP", 90), ("WATER", 80), ("FOOD", 70), ("PLEASE", 60)]
            for word, freq in defaults:
                self.insert(word, freq)
            return
        with open(filepath, "r", encoding="utf-8") as file:
            for line in file:
                parts = line.strip().split(",")
                if len(parts) != 2:
                    continue
                try:
                    self.insert(parts[0].strip().upper(), int(parts[1].strip()))
                except ValueError:
                    continue

    def search_prefix(self, prefix):
        node = self.root
        for char in prefix:
            if char not in node.children:
                return []
            node = node.children[char]
        return self._collect_iterative(node, prefix)

    # BUG FIX: original _collect() was recursive — deep tries (long words /
    # large dictionaries) hit Python's default ~1000 recursion limit.
    # Replaced with an explicit stack so depth is unbounded.
    def _collect_iterative(self, start_node, start_prefix):
        results = []
        stack = [(start_node, start_prefix)]
        while stack:
            node, prefix = stack.pop()
            if node.is_end_of_word:
                results.append((prefix, node.frequency))
            for char, child in node.children.items():
                stack.append((child, prefix + char))
        return results

    def get_top_k_predictions(self, prefix, k=4):
        if not prefix:
            return []
        matches = self.search_prefix(prefix)
        matches.sort(key=lambda item: item[1], reverse=True)
        return [word for word, _ in matches[:k]]


# ---------------------------------------------------------------------------
# EyeMode
# ---------------------------------------------------------------------------

class EyeMode:
    def __init__(self):
        self.gaze = GazeTracker(
            camera_index=CAMERA_INDEX,
            smoothing_frames=EYE_SMOOTHING_FRAMES,
            ear_threshold=EYE_EAR_THRESHOLD,
            dwell_time=EYE_DWELL_TIME,
        )

        llm_path = str(EYE_LLM_MODEL_PATH.resolve())
        if not os.path.exists(llm_path):
            llm_path = EYE_LLM_MODEL
        self.llm = LLMPredictor(llm_path)
        self.predictor = FrequencyTrie(str(EYE_FREQUENCIES_FILE))

        self._start_calibration()

        self.current_text = ""
        self.current_word = ""
        self.predictions = ["", "", "", ""]

        self.cursor_x = CAMERA_WIDTH / 2
        self.cursor_y = CAMERA_HEIGHT / 2
        self.blink_click_count = 0
        self.last_hovered_key = None
        self.keys = []
        self._keys_frame_size = (0, 0)   # track frame size so keyboard rebuilds if needed
        self.message = "Eye mode active"
        self.last_metrics = {
            "state": self.state,
            "raw_x": None,
            "raw_y": None,
            "cursor_x": self.cursor_x,
            "cursor_y": self.cursor_y,
            "is_clicking": False,
            "blink_click_count": self.blink_click_count,
            "hovered_key": None,
            "text_length": 0,
            "message": self.message,
        }

    # ------------------------------------------------------------------
    def _start_calibration(self):
        self.state = "CALIBRATING"
        self.calib_points = ["TOP-LEFT", "TOP-RIGHT", "BOTTOM-LEFT", "BOTTOM-RIGHT"]
        self.calib_index = 0
        self.calib_data = {"x_min": 1.0, "x_max": 0.0, "y_min": 1.0, "y_max": 0.0}

    # ------------------------------------------------------------------
    def tick(self):
        frame, raw_x, raw_y, is_clicking = self.gaze.process()
        frame = cv2.resize(frame, (CAMERA_WIDTH, CAMERA_HEIGHT))

        llm_result = self.llm.check_for_results()
        if llm_result and not self.current_word:
            self.predictions[0] = llm_result

        spoken_text = None

        if self.state == "CALIBRATING":
            if raw_x is not None and raw_y is not None and is_clicking:
                self.calib_data["x_min"] = min(self.calib_data["x_min"], raw_x)
                self.calib_data["x_max"] = max(self.calib_data["x_max"], raw_x)
                self.calib_data["y_min"] = min(self.calib_data["y_min"], raw_y)
                self.calib_data["y_max"] = max(self.calib_data["y_max"], raw_y)
                self.calib_index += 1
                if self.calib_index >= len(self.calib_points):
                    self.state = "TYPING"
                    self.message = "Eye calibration done — start typing"
            rendered = self._render_calibration(frame)
            self.last_metrics = {
                "state": self.state,
                "raw_x": raw_x,
                "raw_y": raw_y,
                "cursor_x": self.cursor_x,
                "cursor_y": self.cursor_y,
                "is_clicking": is_clicking,
                "blink_click_count": self.blink_click_count,
                "hovered_key": None,
                "text_length": len(self.current_text),
                "message": self.message,
            }
            return rendered, None, self.message

        if raw_x is not None and raw_y is not None:
            x_min = self.calib_data["x_min"]
            x_max = self.calib_data["x_max"]
            y_min = self.calib_data["y_min"]
            y_max = self.calib_data["y_max"]

            if abs(x_max - x_min) < 1e-6:
                x_max = x_min + 1e-6
            if abs(y_max - y_min) < 1e-6:
                y_max = y_min + 1e-6

            target_x = np.interp(raw_x, [x_min, x_max], [0, CAMERA_WIDTH])
            target_y = np.interp(raw_y, [y_min, y_max], [0, CAMERA_HEIGHT])
            distance = math.hypot(target_x - self.cursor_x, target_y - self.cursor_y)
            if distance > 15:
                self.cursor_x = (EYE_CURSOR_ALPHA * target_x) + ((1 - EYE_CURSOR_ALPHA) * self.cursor_x)
                self.cursor_y = (EYE_CURSOR_ALPHA * target_y) + ((1 - EYE_CURSOR_ALPHA) * self.cursor_y)

        hovered_char = self._get_hovered_key()
        self.last_hovered_key = hovered_char
        if is_clicking and hovered_char:
            self.blink_click_count += 1
            spoken_text = self._process_key_click(hovered_char)

        rendered = self._render_typing(frame, hovered_char)
        self.last_metrics = {
            "state": self.state,
            "raw_x": raw_x,
            "raw_y": raw_y,
            "cursor_x": self.cursor_x,
            "cursor_y": self.cursor_y,
            "is_clicking": is_clicking,
            "blink_click_count": self.blink_click_count,
            "hovered_key": hovered_char,
            "text_length": len(self.current_text),
            "message": self.message,
            "spoken_text": spoken_text,
        }
        return rendered, spoken_text, self.message

    # ── Calibration screen ──────────────────────────────────────────────
    # KEY FIX: keep the live camera feed visible so the person can see
    # themselves during calibration — only a semi-transparent banner on top.
    def _render_calibration(self, frame):
        canvas = frame.copy()
        h, w = canvas.shape[:2]

        banner_h = 160
        overlay = canvas.copy()
        cv2.rectangle(overlay, (0, 0), (w, banner_h), (15, 15, 15), -1)
        cv2.addWeighted(overlay, 0.70, canvas, 0.30, 0, canvas)

        step_text = f"Step {self.calib_index + 1} / {len(self.calib_points)}"
        cv2.putText(canvas, "Eye Tracking Calibration  —  " + step_text,
                    (24, 46), cv2.FONT_HERSHEY_SIMPLEX, 0.85, (255, 255, 255), 2, cv2.LINE_AA)

        target = self.calib_points[min(self.calib_index, len(self.calib_points) - 1)]
        instruction = f"Look at RED dot  ({target})  then BLINK to confirm"
        cv2.putText(canvas, instruction,
                    (24, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.72, (100, 210, 255), 2, cv2.LINE_AA)
        cv2.putText(canvas, "Keep your head still and look at the corner",
                    (24, 128), cv2.FONT_HERSHEY_SIMPLEX, 0.60, (180, 180, 180), 1, cv2.LINE_AA)

        margin = 55
        corner_map = {
            "TOP-LEFT":     (margin, margin),
            "TOP-RIGHT":    (w - margin, margin),
            "BOTTOM-LEFT":  (margin, h - margin),
            "BOTTOM-RIGHT": (w - margin, h - margin),
        }
        cx, cy = corner_map[target]
        cv2.circle(canvas, (cx, cy), 36, (255, 255, 255), 3)
        cv2.circle(canvas, (cx, cy), 26, (0, 0, 220), -1)
        cv2.circle(canvas, (cx, cy), 10, (255, 80, 80), -1)

        done_map = {0: "TOP-LEFT", 1: "TOP-RIGHT", 2: "BOTTOM-LEFT", 3: "BOTTOM-RIGHT"}
        for i in range(self.calib_index):
            dx, dy = corner_map[done_map[i]]
            cv2.circle(canvas, (dx, dy), 14, (60, 200, 100), -1)
            cv2.putText(canvas, "ok", (dx - 10, dy + 5),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 2, cv2.LINE_AA)

        return canvas

    # ── Typing keyboard ─────────────────────────────────────────────────
    def _build_keys(self, w, h):
        """Rebuild keyboard layout sized to actual frame dimensions."""
        self.keys = []
        y_start = 215
        key_h = max(30, (h - y_start - 20) // 3)

        # Prediction row: 4 word predictions + 1 purple RECAL button
        n_slots = 5
        pred_w = max(20, (w - 40) // n_slots)
        for i in range(4):
            x1 = 20 + i * pred_w
            x2 = x1 + pred_w - 8
            self.keys.append((x1, 115, x2, 190, f"PRED_{i}"))
        x1 = 20 + 4 * pred_w
        x2 = x1 + pred_w - 8
        self.keys.append((x1, 115, x2, 190, "RECAL"))

        layout = [
            ["Q", "W", "E", "R", "T", "Y", "U", "I", "O", "P"],
            ["A", "S", "D", "F", "G", "H", "J", "K", "L", "DEL"],
            ["Z", "X", "C", "V", "B", "N", "M", "SPACE", "CLR", "SPEAK"],
        ]
        for row_idx, row in enumerate(layout):
            key_w = max(20, (w - 40) // len(row))
            for col_idx, char in enumerate(row):
                x1 = 20 + col_idx * key_w
                y1 = y_start + row_idx * key_h
                x2 = x1 + key_w - 8
                y2 = y1 + key_h - 8
                self.keys.append((x1, y1, x2, y2, char))

    def _render_typing(self, frame, hovered_char):
        h, w = frame.shape[:2]

        # BUG FIX: keyboard was built once from constants; now it rebuilds
        # automatically if the rendered frame size changes.
        if self._keys_frame_size != (w, h):
            self._build_keys(w, h)
            self._keys_frame_size = (w, h)

        canvas = frame.copy()

        cv2.rectangle(canvas, (0, 0), (w, 105), (20, 20, 20), -1)
        display_text = (self.current_text + "|")[-75:]
        cv2.putText(canvas, "Eye Tracking Mode", (18, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2, cv2.LINE_AA)
        cv2.putText(canvas, display_text, (18, 80),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.95, (245, 245, 245), 2, cv2.LINE_AA)

        for x1, y1, x2, y2, key in self.keys:
            bg = self._key_color(key)
            if key == hovered_char:
                bg = (0, 122, 204)
            cv2.rectangle(canvas, (x1, y1), (x2, y2), bg, -1)
            cv2.rectangle(canvas, (x1, y1), (x2, y2), (80, 80, 80), 1)

            text = key
            if key.startswith("PRED_"):
                idx = int(key.split("_")[1])
                text = self.predictions[idx] if idx < len(self.predictions) else ""
            fs = 0.55 if key in {"SPEAK", "SPACE", "RECAL"} else 0.72
            tw = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, fs, 2)[0][0]
            tx = x1 + max(4, (x2 - x1 - tw) // 2)
            ty = y1 + (y2 - y1) // 2 + 8
            cv2.putText(canvas, text, (tx, ty),
                        cv2.FONT_HERSHEY_SIMPLEX, fs, (255, 255, 255), 2, cv2.LINE_AA)

        cv2.circle(canvas, (int(self.cursor_x), int(self.cursor_y)), 10, (0, 255, 255), -1)
        return canvas

    def _key_color(self, key):
        if key in {"DEL", "CLR"}:
            return (42, 42, 90)
        if key == "SPACE":
            return (42, 90, 42)
        if key == "SPEAK":
            return (164, 85, 0)
        if key == "RECAL":
            return (90, 42, 90)
        if key.startswith("PRED_"):
            return (70, 70, 70)
        return (42, 42, 42)

    def _get_hovered_key(self):
        for x1, y1, x2, y2, char in self.keys:
            if x1 <= self.cursor_x <= x2 and y1 <= self.cursor_y <= y2:
                return char
        return None

    def _update_predictions(self):
        if self.current_word:
            matches = self.predictor.get_top_k_predictions(self.current_word, k=4)
            self.predictions = matches + [""] * (4 - len(matches))
        else:
            self.predictions = ["", "", "", ""]

    def _process_key_click(self, char):
        spoken_text = None

        if char == "RECAL":
            # NEW: allow recalibration at any time from the typing screen
            self._start_calibration()
            self.keys = []
            self._keys_frame_size = (0, 0)
            self.message = "Recalibrating…"
            return None

        if char.startswith("PRED_"):
            idx = int(char.split("_")[1])
            if self.predictions[idx]:
                pred = self.predictions[idx]
                if " " in pred:
                    self.current_text += pred + " "
                else:
                    # BUG FIX: original slice [-0:] returned the whole string
                    # when current_word was empty, causing text duplication.
                    if self.current_word:
                        self.current_text = self.current_text[: -len(self.current_word)]
                    self.current_text += pred + " "
                self.current_word = ""
                self.llm.request_next_words(self.current_text)
        elif char == "SPACE":
            self.current_text += " "
            self.current_word = ""
            if len(self.current_text.strip()) > 2:
                self.llm.request_next_words(self.current_text)
        elif char == "DEL":
            if self.current_text:
                if self.current_text[-1] != " ":
                    self.current_word = self.current_word[:-1]
                self.current_text = self.current_text[:-1]
        elif char == "CLR":
            self.current_text = ""
            self.current_word = ""
            self.predictions = ["", "", "", ""]
        elif char == "SPEAK":
            spoken_text = self.current_text.strip()
            self.message = "Eye mode speak triggered" if spoken_text else "Nothing to speak"
        else:
            self.current_text += char
            self.current_word += char

        self._update_predictions()
        return spoken_text

    def close(self):
        if self.gaze is not None:
            self.gaze.close()
            self.gaze = None
