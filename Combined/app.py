import queue
import tkinter as tk
from tkinter import ttk
import time

import cv2
from PIL import Image, ImageTk

from core.audio_stt import OfflineSTTEngine
from core.audio_tts import OfflineSpeaker
from core.config import (
    APP_FPS,
    MQTT_BASE_TOPIC,
    MQTT_BROKER_HOST,
    MQTT_BROKER_PORT,
    MQTT_ENABLED,
    MQTT_QOS,
    WINDOW_TITLE,
)
from core.mqtt_publisher import MQTTPublisher
from core.transcript_store import TranscriptStore
from modes.asl_mode import ASLMode
from modes.eye_mode import EyeMode
from modes.startup_gate import StartupGate
from ui.chat_panel import ChatPanel


class UnifiedSignEdgeApp:
    def __init__(self):
        self.root = tk.Tk()
        self.root.title(WINDOW_TITLE)
        self.root.configure(bg="#0d0f12")
        self._maximize_window()

        self.transcript = TranscriptStore()
        self.speaker = OfflineSpeaker()
        self.stt_engine = OfflineSTTEngine(enabled=True)

        self.mode = "BOOT_GATE"
        self.mode_message = "Initializing"
        self.stt_partial = ""
        self._session_started_at = time.time()
        self._tick_count = 0
        self._telemetry_every_n_ticks = 5

        self.gate_mode = StartupGate()
        self.eye_mode = None
        self.asl_mode = None

        self.publisher = MQTTPublisher(
            host=MQTT_BROKER_HOST,
            port=MQTT_BROKER_PORT,
            base_topic=MQTT_BASE_TOPIC,
            qos=MQTT_QOS,
            enabled=MQTT_ENABLED,
        )

        self._current_image = None

        self._build_ui()
        self._bind_keys()
        self.root.protocol("WM_DELETE_WINDOW", self.close)

    def _publish(self, topic_suffix, payload):
        try:
            self.publisher.publish(topic_suffix, payload)
        except Exception:
            pass

    def _maximize_window(self):
        try:
            self.root.state("zoomed")
            return
        except tk.TclError:
            pass
        try:
            self.root.attributes("-zoomed", True)
            return
        except tk.TclError:
            pass
        width = self.root.winfo_screenwidth()
        height = self.root.winfo_screenheight()
        self.root.geometry(f"{width}x{height}+0+0")

    def _build_ui(self):
        # Side-by-side layout: left spacer | main | thin chat
        # Effective main/chat split is ~82/18 for usable content.
        # Spacer keeps the main pane visually near center.
        self.root.columnconfigure(0, weight=14)
        self.root.columnconfigure(1, weight=82)
        self.root.columnconfigure(2, weight=18)
        self.root.rowconfigure(0, weight=1)

        self.left_spacer = tk.Frame(self.root, bg="#0d0f12")
        self.left_spacer.grid(row=0, column=0, sticky="nsew")

        # ── LEFT: thin status bar + video ──
        self.left_frame = tk.Frame(self.root, bg="#0d0f12")
        self.left_frame.grid(row=0, column=1, sticky="nsew", padx=(6, 4), pady=6)
        self.left_frame.columnconfigure(0, weight=1)
        self.left_frame.rowconfigure(1, weight=1)

        self.status_var = tk.StringVar(value="Starting…")
        status_lbl = tk.Label(
            self.left_frame,
            textvariable=self.status_var,
            font=("Segoe UI", 10),
            bg="#0d0f12",
            fg="#9ca3af",
            anchor="w",
        )
        status_lbl.grid(row=0, column=0, sticky="ew", padx=4, pady=(2, 2))

        # Video fills all remaining space
        self.video_label = tk.Label(self.left_frame, bg="#0d0f12", anchor="center")
        self.video_label.grid(row=1, column=0, sticky="nsew")

        # ── RIGHT: chat panel ──
        self.right_frame = tk.Frame(self.root, bg="#0d0f12")
        self.right_frame.grid(row=0, column=2, sticky="nsew", padx=(4, 6), pady=6)
        self.right_frame.columnconfigure(0, weight=1)
        self.right_frame.rowconfigure(0, weight=1)
        self.right_frame.grid_propagate(False)

        self.chat_panel = ChatPanel(self.right_frame)

    def _bind_keys(self):
        # Press Escape at any time to restart mode selection
        self.root.bind("<Escape>", lambda _: self._restart_gate())

    def _restart_gate(self):
        """Return to the startup gate so the user can switch between eye / ASL mode."""
        old_mode = self.mode
        self._close_current_mode()
        self.mode = "BOOT_GATE"
        self.gate_mode = StartupGate()
        self.mode_message = "Mode selection restarted"
        self.transcript.append("SYS", "Mode selection restarted (Escape pressed)")
        self._publish(
            "mode/switch",
            {
                "old_mode": old_mode,
                "new_mode": "BOOT_GATE",
                "reason": "escape_pressed",
            },
        )

    def start(self):
        self.transcript.append(
            "SYS",
            "App started. Default path is eye mode unless hand is raised in 10s. "
            "Press Escape at any time to restart mode selection.",
        )
        self._tick()
        self._publish(
            "system/start",
            {
                "mode": self.mode,
                "message": "app_started",
                "mqtt_connected": self.publisher.connected,
            },
        )
        self.root.mainloop()

    def _switch_to_eye_mode(self):
        old_mode = self.mode
        self._close_current_mode()
        self.mode = "EYE"
        self.eye_mode = EyeMode()
        self.mode_message = "Eye mode active"
        self.transcript.append("SYS", "Switched to eye tracking mode")
        self._publish(
            "mode/switch",
            {"old_mode": old_mode, "new_mode": "EYE", "message": self.mode_message},
        )

    def _switch_to_asl_mode(self):
        old_mode = self.mode
        self._close_current_mode()
        self.mode = "ASL"
        self.asl_mode = ASLMode()
        self.mode_message = "ASL mode active"
        self.transcript.append("SYS", "Switched to ASL finger recognition mode")
        self._publish(
            "mode/switch",
            {"old_mode": old_mode, "new_mode": "ASL", "message": self.mode_message},
        )

    def _close_current_mode(self):
        for attr in ("gate_mode", "eye_mode", "asl_mode"):
            obj = getattr(self, attr, None)
            if obj is not None:
                try:
                    obj.close()
                except Exception:
                    pass
                setattr(self, attr, None)

    def _drain_stt_events(self):
        while True:
            try:
                event_type, message = self.stt_engine.event_queue.get_nowait()
            except queue.Empty:
                break
            if event_type == "partial":
                self.stt_partial = message
                self._publish("stt/partial", {"text": message})
            elif event_type == "final":
                self.stt_partial = ""
                self.transcript.append("STT", message)
                self._publish("stt/final", {"text": message})
            elif event_type in {"ready", "error"}:
                self.transcript.append("SYS", message)
                self._publish("stt/status", {"event": event_type, "message": message})

    def _drain_tts_events(self):
        while True:
            try:
                event_type, message = self.speaker.event_queue.get_nowait()
            except queue.Empty:
                break
            # BUG FIX: original only handled "ready" and "error".
            # "status" events (e.g. Piper fallback notice) were silently dropped.
            if event_type in {"ready", "error", "status"}:
                self.transcript.append("SYS", message)
                self._publish("tts/status", {"event": event_type, "message": message})

    def _handle_mode_tick(self):
        if self.mode == "BOOT_GATE" and self.gate_mode is not None:
            frame, select_asl, timeout = self.gate_mode.tick()
            self.mode_message = "Startup gate (raise hand for ASL)"
            if select_asl:
                self._switch_to_asl_mode()
            elif timeout:
                self._switch_to_eye_mode()
            return frame

        if self.mode == "ASL" and self.asl_mode is not None:
            frame, sent_text, msg = self.asl_mode.tick()
            self.mode_message = msg
            if sent_text:
                self.speaker.speak(sent_text)
                self.transcript.append("TTS_ASL", sent_text)
            return frame

        if self.mode == "EYE" and self.eye_mode is not None:
            frame, spoken_text, msg = self.eye_mode.tick()
            self.mode_message = msg
            if spoken_text:
                self.speaker.speak(spoken_text)
                self.transcript.append("TTS_EYE", spoken_text)
            return frame

        return None

    def _render_frame(self, frame_bgr):
        if frame_bgr is None:
            return

        frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        image = Image.fromarray(frame_rgb)

        pane_w = max(320, self.video_label.winfo_width())
        pane_h = max(240, self.video_label.winfo_height())
        image.thumbnail((pane_w, pane_h))

        self._current_image = ImageTk.PhotoImage(image=image)
        self.video_label.configure(image=self._current_image)

    def _tick(self):
        tick_started = time.perf_counter()
        self._drain_stt_events()
        self._drain_tts_events()

        # Wrap the mode tick so a crash in one frame doesn't kill the whole app
        try:
            frame = self._handle_mode_tick()
        except Exception as exc:
            self.transcript.append("SYS", f"Mode error: {exc}")
            frame = None

        self._render_frame(frame)

        partial = f"  ·  STT: {self.stt_partial}" if self.stt_partial else ""
        self.status_var.set(
            f"Mode: {self.mode}  ·  {self.mode_message}{partial}"
            "    [Esc = switch mode]"
        )

        self.chat_panel.refresh(self.transcript.snapshot())

        self._tick_count += 1
        if self._tick_count % self._telemetry_every_n_ticks == 0:
            loop_ms = (time.perf_counter() - tick_started) * 1000.0
            self._publish(
                "system/heartbeat",
                {
                    "mode": self.mode,
                    "mode_message": self.mode_message,
                    "stt_partial": self.stt_partial,
                    "uptime_s": time.time() - self._session_started_at,
                    "loop_ms": loop_ms,
                    "fps_target": APP_FPS,
                    "mqtt_connected": self.publisher.connected,
                },
            )

            if self.mode == "ASL" and self.asl_mode is not None:
                self._publish("asl/metrics", self.asl_mode.last_metrics)
            elif self.mode == "EYE" and self.eye_mode is not None:
                self._publish("eye/metrics", self.eye_mode.last_metrics)

        interval = int(1000 / APP_FPS)
        self.root.after(interval, self._tick)

    def close(self):
        self._publish(
            "system/stop",
            {
                "mode": self.mode,
                "message": "app_stopping",
                "uptime_s": time.time() - self._session_started_at,
            },
        )
        self._close_current_mode()
        self.stt_engine.stop()
        self.speaker.stop()
        self.publisher.stop()
        self.root.destroy()


def main():
    app = UnifiedSignEdgeApp()
    app.start()


if __name__ == "__main__":
    main()
