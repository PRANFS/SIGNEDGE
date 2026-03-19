import io
import queue
import shutil
import subprocess
import tempfile
import threading
import wave
from pathlib import Path

from .config import ASSETS_DIR, PIPER_VOICE


class OfflineSpeaker:
    def __init__(self):
        self.command_queue = queue.Queue()
        self.event_queue = queue.Queue()
        self.stop_event = threading.Event()
        self.piper_voice = None
        self.tts_command = None
        self.aplay_command = shutil.which("aplay")

        engine_name, piper_issue = self._load_piper()
        if engine_name is None:
            # BUG FIX: was published as "status" which _drain_tts_events ignores.
            # Changed to "error" so the fallback notice actually appears in the transcript.
            self._publish("error", f"[TTS] Piper not loaded ({piper_issue}); falling back to espeak")
            try:
                engine_name = self._load_espeak()
            except RuntimeError as exc:
                self._publish("error", f"[TTS] No TTS engine available: {exc}")
                engine_name = "none"

        self.thread = threading.Thread(target=self._worker, daemon=True)
        self.thread.start()
        self._publish("ready", f"Speech ready via {engine_name}")

    def _publish(self, event_type, message):
        self.event_queue.put((event_type, message))

    def _load_piper(self):
        try:
            from piper import PiperVoice  # type: ignore[import]
        except ImportError:
            return None, "piper-tts not installed"

        model_path = ASSETS_DIR / f"{PIPER_VOICE}.onnx"
        if not model_path.exists():
            return None, f"model file not found: {model_path}"

        try:
            self.piper_voice = PiperVoice.load(str(model_path))
            return f"piper ({PIPER_VOICE})", None
        except Exception as exc:
            return None, f"PiperVoice.load failed: {exc}"

    def _load_espeak(self):
        self.tts_command = shutil.which("espeak-ng") or shutil.which("espeak")
        if not self.tts_command:
            raise RuntimeError("No TTS engine found. Install piper-tts or espeak-ng.")
        return Path(self.tts_command).name

    def speak(self, text):
        cleaned = (text or "").strip()
        if cleaned:
            self.command_queue.put(cleaned)

    def stop(self):
        self.stop_event.set()
        self.command_queue.put(None)  # unblock the worker if it is waiting
        self.thread.join(timeout=2.0)

    def _worker(self):
        while not self.stop_event.is_set():
            text = self.command_queue.get()
            if text is None:
                break

            try:
                if self.piper_voice is not None:
                    self._speak_piper(text)
                elif self.tts_command is not None:
                    self._speak_espeak(text)
                else:
                    # No engine loaded — silently skip so the app doesn't crash
                    pass
            except Exception as exc:
                self._publish("error", f"Speech failed: {exc}")

    def _speak_piper(self, text):
        chunks = list(self.piper_voice.synthesize(text))
        if not chunks:
            return

        sample_rate = chunks[0].sample_rate
        channels = chunks[0].sample_channels

        # Prefer aplay (Linux, Raspberry Pi)
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
            return

        # Fallback: write a temp WAV and play with ffplay
        ffplay = shutil.which("ffplay")
        if ffplay:
            buf = io.BytesIO()
            with wave.open(buf, "wb") as wav_out:
                wav_out.setnchannels(channels)
                wav_out.setsampwidth(2)
                wav_out.setframerate(sample_rate)
                for chunk in chunks:
                    wav_out.writeframes(chunk.audio_int16_bytes)

            with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
                tmp.write(buf.getvalue())
                tmp_path = tmp.name

            subprocess.run(
                [ffplay, "-nodisp", "-autoexit", tmp_path],
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
                check=False,
            )
            Path(tmp_path).unlink(missing_ok=True)
            return

        # BUG FIX: previously silently did nothing if neither aplay nor ffplay exist.
        self._publish("error", "[TTS] No audio playback tool found (need aplay or ffplay)")

    def _speak_espeak(self, text):
        result = subprocess.run(
            [self.tts_command, "-s", "165", text],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.PIPE,
            check=False,
        )
        if result.returncode != 0:
            self._publish("error", f"espeak failed: {result.stderr.decode(errors='replace').strip()}")
