import queue
import tkinter as tk
from tkinter import ttk

import cv2
from PIL import Image, ImageTk

from core.audio_stt import OfflineSTTEngine
from core.audio_tts import OfflineSpeaker
from core.config import APP_FPS, WINDOW_TITLE
from core.transcript_store import TranscriptStore
from modes.asl_mode import ASLMode
from modes.eye_mode import EyeMode
from modes.startup_gate import StartupGate
from ui.chat_panel import ChatPanel


class UnifiedSignEdgeApp:
    def __init__(self):
        self.root = tk.Tk()
        self.root.title(WINDOW_TITLE)
        self._maximize_window()

        self.transcript = TranscriptStore()
        self.speaker = OfflineSpeaker()
        self.stt_engine = OfflineSTTEngine(enabled=True)

        self.mode = "BOOT_GATE"
        self.mode_message = "Initializing"
        self.stt_partial = ""

        self.gate_mode = StartupGate()
        self.eye_mode = None
        self.asl_mode = None

        self._current_image = None

        self._build_ui()
        self.root.protocol("WM_DELETE_WINDOW", self.close)

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
        self.root.columnconfigure(0, weight=3)
        self.root.columnconfigure(1, weight=1)
        self.root.rowconfigure(0, weight=1)

        left_frame = ttk.Frame(self.root)
        left_frame.grid(row=0, column=0, sticky="nsew")
        left_frame.columnconfigure(0, weight=1)
        left_frame.rowconfigure(1, weight=1)

        self.status_var = tk.StringVar(value="Starting...")
        status_lbl = ttk.Label(left_frame, textvariable=self.status_var, font=("Segoe UI", 11))
        status_lbl.grid(row=0, column=0, sticky="ew", padx=8, pady=8)

        self.video_label = ttk.Label(left_frame)
        self.video_label.grid(row=1, column=0, sticky="nsew", padx=8, pady=(0, 8))

        right_frame = ttk.Frame(self.root)
        right_frame.grid(row=0, column=1, sticky="nsew")
        self.chat_panel = ChatPanel(right_frame)

    def start(self):
        self.transcript.append("SYS", "App started. Default path is eye mode unless hand is raised in 10s.")
        self._tick()
        self.root.mainloop()

    def _switch_to_eye_mode(self):
        self._close_current_mode()
        self.mode = "EYE"
        self.eye_mode = EyeMode()
        self.mode_message = "Eye mode active"
        self.transcript.append("SYS", "Switched to eye tracking mode")

    def _switch_to_asl_mode(self):
        self._close_current_mode()
        self.mode = "ASL"
        self.asl_mode = ASLMode()
        self.mode_message = "ASL mode active"
        self.transcript.append("SYS", "Switched to ASL finger recognition mode")

    def _close_current_mode(self):
        if self.gate_mode is not None:
            self.gate_mode.close()
            self.gate_mode = None
        if self.eye_mode is not None:
            self.eye_mode.close()
            self.eye_mode = None
        if self.asl_mode is not None:
            self.asl_mode.close()
            self.asl_mode = None

    def _drain_stt_events(self):
        while True:
            try:
                event_type, message = self.stt_engine.event_queue.get_nowait()
            except queue.Empty:
                break

            if event_type == "partial":
                self.stt_partial = message
            elif event_type == "final":
                self.stt_partial = ""
                self.transcript.append("STT", message)
            elif event_type in {"ready", "error"}:
                self.transcript.append("SYS", message)

    def _drain_tts_events(self):
        while True:
            try:
                event_type, message = self.speaker.event_queue.get_nowait()
            except queue.Empty:
                break
            if event_type in {"ready", "error"}:
                self.transcript.append("SYS", message)

    def _handle_mode_tick(self):
        if self.mode == "BOOT_GATE":
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
        self._drain_stt_events()
        self._drain_tts_events()

        frame = self._handle_mode_tick()
        self._render_frame(frame)

        partial = f" | STT: {self.stt_partial}" if self.stt_partial else ""
        self.status_var.set(f"Mode: {self.mode} | {self.mode_message}{partial}")

        self.chat_panel.refresh(self.transcript.snapshot())

        interval = int(1000 / APP_FPS)
        self.root.after(interval, self._tick)

    def close(self):
        self._close_current_mode()
        self.stt_engine.stop()
        self.speaker.stop()
        self.root.destroy()


def main():
    app = UnifiedSignEdgeApp()
    app.start()


if __name__ == "__main__":
    main()
