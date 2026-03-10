"""
SignEdge main visual application for Raspberry Pi 5.

This app uses a single OpenCV rendering loop for better responsiveness on Pi:
    - MediaPipe Hands for landmark extraction
    - TFLite MLP inference loaded directly inside this app
    - Stable-hold sign commit for A-Z, SPACE, and DEL
    - Camera-based dwell-to-send interaction
    - Offline neural speech via Piper TTS (piper1-gpl), falls back to espeak-ng

Piper setup (one-time on the Pi):
    pip install piper-tts
    python3 -m piper.download_voices en_US-lessac-medium
"""

import os
os.environ.setdefault("MPLBACKEND", "Agg")

import json
import queue
import shutil
import subprocess
import threading
import time
from collections import Counter, deque
from dataclasses import dataclass
from pathlib import Path

import cv2
import mediapipe as mp
import numpy as np


WINDOW_TITLE = "SignEdge Visual Communicator"
CAMERA_INDEX = 0
CAMERA_WIDTH = 960
CAMERA_HEIGHT = 540
CAMERA_FPS = 30

SMOOTH_WINDOW = 7
CONFIDENCE_THRESHOLD = 0.65
HOLD_SECONDS = 0.85
RELEASE_SECONDS = 0.30
SEND_DWELL_SECONDS = 1.75
SEND_ZONE_RATIO = 0.22
TTS_RATE = 165
PIPER_VOICE = "en_US-amy-medium"

TOP_BAR_H = 110
BOTTOM_BAR_H = 160
SIDE_PANEL_W = 220
PANEL_BG = (18, 24, 32)


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


def format_prediction_label(label):
    if not label:
        return "--"
    if label == "SPACE":
        return "SPACE"
    if label == "DEL":
        return "DELETE"
    return label


def top_prediction_mode(predictions):
    if not predictions:
        return None

    counts = Counter(predictions)
    return counts.most_common(1)[0][0]


def wrap_text(text, max_chars):
    words = text.split()
    if not words:
        return [""]

    lines = []
    current = words[0]
    for word in words[1:]:
        candidate = current + " " + word
        if len(candidate) <= max_chars:
            current = candidate
        else:
            lines.append(current)
            current = word
    lines.append(current)
    return lines


@dataclass
class FrameAnalysis:
    frame: np.ndarray
    hand_detected: bool
    smoothed_label: str | None
    confidence: float
    top3: list
    hand_center_x: float | None


class TFLiteSignPredictor:
    def __init__(self, model_dir):
        self.model_dir = Path(model_dir)
        self.model_path = self.model_dir / "asl_mlp_int8.tflite"
        self.mean_path = self.model_dir / "scaler_mean.npy"
        self.scale_path = self.model_dir / "scaler_scale.npy"
        self.label_map_path = self.model_dir / "label_map.json"

        required_paths = [
            self.model_path,
            self.mean_path,
            self.scale_path,
            self.label_map_path,
        ]
        missing_paths = [path.name for path in required_paths if not path.exists()]
        if missing_paths:
            raise FileNotFoundError(
                "Missing required model assets: " + ", ".join(missing_paths)
            )

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
        top3 = [(self.label_map[str(int(index))], float(probs[index])) for index in top3_idxs]
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


class OfflineSpeaker:
    def __init__(self):
        self.command_queue = queue.Queue()
        self.event_queue = queue.Queue()
        self.stop_event = threading.Event()
        self.piper_voice = None
        self.tts_command = None

        engine_name, piper_issue = self._load_piper()
        if engine_name is None:
            print(f"[TTS] Piper not loaded: {piper_issue} — falling back to espeak-ng")
            engine_name = self._load_espeak()

        self.aplay_command = shutil.which("aplay")
        self.thread = threading.Thread(target=self._worker, daemon=True)
        self.thread.start()
        self._publish("ready", f"Speech ready via {engine_name}")

    def _load_piper(self):
        """Try to load Piper TTS. Returns (engine_name, None) on success, (None, reason) on failure."""
        try:
            from piper import PiperVoice  # type: ignore[import]
        except ImportError:
            return None, "piper-tts not installed (run: pip install piper-tts)"

        model_path = Path(__file__).resolve().parent / f"{PIPER_VOICE}.onnx"
        if not model_path.exists():
            return None, f"model file not found: {model_path}"

        try:
            self.piper_voice = PiperVoice.load(str(model_path))
            return f"piper ({PIPER_VOICE})", None
        except Exception as exc:
            return None, f"PiperVoice.load failed: {exc}"

    def _load_espeak(self):
        """Fall back to espeak-ng / espeak. Raises if neither is available."""
        self.tts_command = shutil.which("espeak-ng") or shutil.which("espeak")
        if not self.tts_command:
            raise RuntimeError(
                "No TTS engine found. Install piper-tts (pip install piper-tts) "
                "or espeak-ng (sudo apt install espeak-ng)."
            )
        return Path(self.tts_command).name

    def speak(self, text):
        self.command_queue.put(text)

    def stop(self):
        self.stop_event.set()
        self.command_queue.put(None)
        self.thread.join(timeout=2.0)

    def _publish(self, event_type, message):
        self.event_queue.put((event_type, message))

    def _worker(self):
        while not self.stop_event.is_set():
            text = self.command_queue.get()
            if text is None:
                break

            try:
                self._publish("status", f"Speaking: {text}")
                if self.piper_voice is not None:
                    self._speak_piper(text)
                else:
                    self._speak_espeak(text)
                self._publish("spoken", text)
            except Exception as exc:
                self._publish("error", f"Speech failed: {exc}")

    def _speak_piper(self, text):
        chunks = list(self.piper_voice.synthesize(text))
        if not chunks:
            return

        first = chunks[0]
        sample_rate = first.sample_rate
        channels = first.sample_channels

        if self.aplay_command:
            proc = subprocess.Popen(
                [
                    self.aplay_command,
                    "-r", str(sample_rate),
                    "-f", "S16_LE",
                    "-c", str(channels),
                    "-t", "raw",
                    "-",
                ],
                stdin=subprocess.PIPE,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
            )
            for chunk in chunks:
                proc.stdin.write(chunk.audio_int16_bytes)
            proc.stdin.close()
            proc.wait()
        else:
            # aplay unavailable — write WAV to a temp file and play via ffplay
            import io
            import tempfile
            import wave

            buf = io.BytesIO()
            with wave.open(buf, "wb") as wf:
                wf.setnchannels(channels)
                wf.setsampwidth(2)
                wf.setframerate(sample_rate)
                for chunk in chunks:
                    wf.writeframes(chunk.audio_int16_bytes)

            ffplay = shutil.which("ffplay")
            if ffplay:
                with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
                    tmp.write(buf.getvalue())
                    tmp_path = tmp.name
                subprocess.run(
                    [ffplay, "-nodisp", "-autoexit", tmp_path],
                    stdout=subprocess.DEVNULL,
                    stderr=subprocess.DEVNULL,
                )
                Path(tmp_path).unlink(missing_ok=True)

    def _speak_espeak(self, text):
        subprocess.run(
            [self.tts_command, "-s", str(TTS_RATE), text],
            check=True,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.PIPE,
            text=True,
        )


class SignEdgeApp:
    def __init__(self):
        self.project_dir = Path(__file__).resolve().parent
        self.predictor = TFLiteSignPredictor(self.project_dir)
        self.capture = self._initialize_camera()
        self.hands, self.mp_hands, self.mp_draw, self.mp_styles = self._initialize_hands()
        self.composer = StableTextComposer(
            confidence_threshold=CONFIDENCE_THRESHOLD,
            hold_seconds=HOLD_SECONDS,
            release_seconds=RELEASE_SECONDS,
        )
        self.speaker = OfflineSpeaker()
        self.pred_history = deque(maxlen=SMOOTH_WINDOW)
        self.send_dwell_started = None
        self.send_progress = 0.0
        self.is_speaking = False
        self.fps_start = time.perf_counter()
        self.fps_counter = 0
        self.fps_display = 0.0
        self.message = "Initializing visual pipeline"

    def _initialize_camera(self):
        capture = cv2.VideoCapture(CAMERA_INDEX)
        capture.set(cv2.CAP_PROP_FRAME_WIDTH, CAMERA_WIDTH)
        capture.set(cv2.CAP_PROP_FRAME_HEIGHT, CAMERA_HEIGHT)
        capture.set(cv2.CAP_PROP_FPS, CAMERA_FPS)
        if not capture.isOpened():
            raise RuntimeError(f"Could not open camera index {CAMERA_INDEX} on Raspberry Pi")
        return capture

    def _initialize_hands(self):
        mp_hands = mp.solutions.hands
        mp_draw = mp.solutions.drawing_utils
        mp_styles = mp.solutions.drawing_styles
        hands = mp_hands.Hands(
            static_image_mode=False,
            max_num_hands=1,
            min_detection_confidence=0.6,
            min_tracking_confidence=0.6,
            model_complexity=0,
        )
        return hands, mp_hands, mp_draw, mp_styles

    def run(self):
        cv2.namedWindow(WINDOW_TITLE, cv2.WINDOW_NORMAL)
        cv2.resizeWindow(WINDOW_TITLE, CAMERA_WIDTH + SIDE_PANEL_W, TOP_BAR_H + CAMERA_HEIGHT + BOTTOM_BAR_H)

        try:
            while True:
                self._drain_speech_events()
                analysis = self._process_frame()
                now = time.perf_counter()
                in_send_zone = bool(
                    analysis.hand_detected
                    and analysis.hand_center_x is not None
                    and analysis.hand_center_x >= 1.0 - SEND_ZONE_RATIO
                    and bool(self.composer.buffer.strip())
                )

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

                self._update_fps()
                display_frame = self._decorate_frame(analysis, compose_result, in_send_zone)
                cv2.imshow(WINDOW_TITLE, display_frame)

                key = cv2.waitKey(1) & 0xFF
                if key in (27, ord("q")):
                    break
        finally:
            self.close()

    def _process_frame(self):
        ok, frame = self.capture.read()
        if not ok:
            raise RuntimeError("Camera frame capture failed")

        frame = cv2.flip(frame, 1)
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        rgb.flags.writeable = False
        results = self.hands.process(rgb)
        rgb.flags.writeable = True

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

            points = np.array(
                [[point.x, point.y, point.z] for point in hand_landmarks.landmark],
                dtype=np.float32,
            )
            hand_center_x = float(points[:, 0].mean())

            pred_label, confidence, top3 = self.predictor.predict(points.flatten())
            normalized = normalize_prediction_label(pred_label)
            if normalized:
                self.pred_history.append(normalized)
                smoothed_label = top_prediction_mode(self.pred_history)

        if not hand_detected:
            self.pred_history.clear()

        return FrameAnalysis(
            frame=frame,
            hand_detected=hand_detected,
            smoothed_label=smoothed_label,
            confidence=confidence,
            top3=top3,
            hand_center_x=hand_center_x,
        )

    def _decorate_frame(self, analysis, compose_result, in_send_zone):
        cam_h, cam_w = analysis.frame.shape[:2]
        canvas_w = cam_w + SIDE_PANEL_W
        canvas_h = TOP_BAR_H + cam_h + BOTTOM_BAR_H

        canvas = np.full((canvas_h, canvas_w, 3), PANEL_BG, dtype=np.uint8)

        # Camera feed (skeleton already drawn on it) — untouched camera pixels
        canvas[TOP_BAR_H:TOP_BAR_H + cam_h, 0:cam_w] = analysis.frame

        # ── TOP BAR ──────────────────────────────────────────────────────────
        current_label = format_prediction_label(analysis.smoothed_label)
        label_color = (92, 224, 160) if analysis.hand_detected else (130, 140, 150)
        cv2.putText(canvas, current_label, (26, 72), cv2.FONT_HERSHEY_SIMPLEX, 2.0, label_color, 4, cv2.LINE_AA)
        conf_text = f"Confidence {analysis.confidence:.0%}" if analysis.hand_detected else "No hand detected"
        cv2.putText(canvas, conf_text, (28, 100), cv2.FONT_HERSHEY_SIMPLEX, 0.68, (200, 210, 220), 2, cv2.LINE_AA)
        cv2.putText(canvas, self.message, (260, 52), cv2.FONT_HERSHEY_SIMPLEX, 0.58, (236, 242, 246), 2, cv2.LINE_AA)
        cv2.putText(canvas, f"FPS {self.fps_display:.0f}", (canvas_w - 120, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (200, 210, 220), 2, cv2.LINE_AA)
        cv2.line(canvas, (0, TOP_BAR_H - 1), (canvas_w, TOP_BAR_H - 1), (55, 65, 78), 1)

        # ── SEND PANEL (right strip beside camera) ───────────────────────────
        send_x0 = cam_w
        zone_color = (31, 191, 117) if in_send_zone else (30, 52, 78)
        cv2.rectangle(canvas, (send_x0, TOP_BAR_H), (canvas_w, TOP_BAR_H + cam_h), zone_color, -1)
        cv2.line(canvas, (send_x0, TOP_BAR_H), (send_x0, TOP_BAR_H + cam_h), (225, 240, 248), 2)
        send_cx = send_x0 + SIDE_PANEL_W // 2
        cv2.putText(canvas, "SEND", (send_cx - 46, TOP_BAR_H + 55), cv2.FONT_HERSHEY_SIMPLEX, 1.1, (255, 255, 255), 3, cv2.LINE_AA)
        cv2.putText(canvas, "Move hand to", (send_cx - 58, TOP_BAR_H + 90), cv2.FONT_HERSHEY_SIMPLEX, 0.54, (220, 235, 245), 2, cv2.LINE_AA)
        cv2.putText(canvas, "right edge", (send_cx - 44, TOP_BAR_H + 113), cv2.FONT_HERSHEY_SIMPLEX, 0.54, (220, 235, 245), 2, cv2.LINE_AA)
        # Vertical dwell progress bar
        vbar_x = send_cx - 12
        vbar_y0 = TOP_BAR_H + 145
        vbar_h = cam_h - 185
        cv2.rectangle(canvas, (vbar_x, vbar_y0), (vbar_x + 24, vbar_y0 + vbar_h), (20, 35, 50), -1)
        if in_send_zone and self.send_progress > 0:
            filled = int(vbar_h * self.send_progress)
            cv2.rectangle(canvas, (vbar_x, vbar_y0 + vbar_h - filled), (vbar_x + 24, vbar_y0 + vbar_h), (255, 180, 84), -1)

        # ── BOTTOM BAR ───────────────────────────────────────────────────────
        by0 = TOP_BAR_H + cam_h
        cv2.line(canvas, (0, by0), (canvas_w, by0), (55, 65, 78), 1)

        # Sentence (left)
        sentence_text = self.composer.buffer if self.composer.buffer else "Start fingerspelling to build a sentence"
        sentence_lines = wrap_text(sentence_text, 52)
        cv2.putText(canvas, "Sentence", (24, by0 + 26), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (164, 184, 198), 2, cv2.LINE_AA)
        for idx, line in enumerate(sentence_lines[:2]):
            cv2.putText(canvas, line, (24, by0 + 56 + idx * 30), cv2.FONT_HERSHEY_SIMPLEX, 0.80, (238, 244, 247), 2, cv2.LINE_AA)

        # Status text + hold progress bar
        status_text, _ = self._build_status_text(analysis, compose_result, in_send_zone)
        cv2.putText(canvas, status_text, (24, by0 + 126), cv2.FONT_HERSHEY_SIMPLEX, 0.58, (255, 180, 84), 2, cv2.LINE_AA)
        progress = compose_result["progress"] if not in_send_zone else 0.0
        hbar_y = by0 + BOTTOM_BAR_H - 16
        cv2.rectangle(canvas, (24, hbar_y - 10), (344, hbar_y), (42, 54, 66), -1)
        cv2.rectangle(canvas, (24, hbar_y - 10), (24 + int(320 * progress), hbar_y), (31, 191, 117), -1)

        # Top 3 (right of bottom bar)
        top3_x = cam_w - 240
        cv2.putText(canvas, "Top 3", (top3_x, by0 + 26), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (164, 184, 198), 2, cv2.LINE_AA)
        for rank, (raw_label, probability) in enumerate(analysis.top3[:3]):
            lbl = format_prediction_label(normalize_prediction_label(raw_label))
            color = (92, 224, 160) if rank == 0 else (200, 210, 220)
            cv2.putText(canvas, f"{lbl} {probability:.0%}", (top3_x, by0 + 56 + rank * 30), cv2.FONT_HERSHEY_SIMPLEX, 0.68, color, 2, cv2.LINE_AA)
        cv2.putText(canvas, f"Last sent: {self.composer.last_sent or '--'}", (top3_x, by0 + BOTTOM_BAR_H - 16), cv2.FONT_HERSHEY_SIMPLEX, 0.58, (182, 196, 206), 2, cv2.LINE_AA)

        return canvas

    def _build_status_text(self, analysis, compose_result, in_send_zone):
        if in_send_zone:
            return "SEND zone active", f"Send progress {self.send_progress:.0%}"

        candidate = compose_result["candidate"] if compose_result else None
        if not analysis.hand_detected:
            return "No hand detected", "Hold progress waiting"
        if candidate:
            return f"Recognizing {format_prediction_label(candidate)}", f"Hold progress {compose_result['progress']:.0%}"
        return "Hand detected", "Hold a stable sign to commit"

    def _update_fps(self):
        self.fps_counter += 1
        if self.fps_counter < 20:
            return

        elapsed = time.perf_counter() - self.fps_start
        if elapsed > 0:
            self.fps_display = self.fps_counter / elapsed
        self.fps_counter = 0
        self.fps_start = time.perf_counter()

    def _update_send_dwell(self, now):
        if self.send_dwell_started is None:
            self.send_dwell_started = now

        progress = min((now - self.send_dwell_started) / SEND_DWELL_SECONDS, 1.0)
        if progress >= 1.0 and not self.is_speaking:
            self._send_current_sentence()
            self._reset_send_dwell()
        return progress

    def _reset_send_dwell(self):
        self.send_dwell_started = None
        self.send_progress = 0.0

    def _send_current_sentence(self):
        text = self.composer.send()
        if not text:
            self.message = "Nothing to send yet"
            return

        self.is_speaking = True
        self.message = f"Sending: {text}"
        self.speaker.speak(text)

    def _drain_speech_events(self):
        while True:
            try:
                event_type, message = self.speaker.event_queue.get_nowait()
            except queue.Empty:
                break

            if event_type == "ready":
                self.message = message
            elif event_type == "status":
                self.message = message
            elif event_type == "spoken":
                self.is_speaking = False
                self.message = f"Spoken: {message}"
            elif event_type == "error":
                self.is_speaking = False
                self.message = message

    def close(self):
        try:
            if self.capture is not None:
                self.capture.release()
        finally:
            self.capture = None

        try:
            if self.hands is not None:
                self.hands.close()
        finally:
            self.hands = None

        self.speaker.stop()
        cv2.destroyAllWindows()


def main():
    app = SignEdgeApp()
    app.run()


if __name__ == "__main__":
    main()
