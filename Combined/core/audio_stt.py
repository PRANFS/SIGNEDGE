import queue

from .config import (
    ASSETS_DIR,
    STT_BLOCK_SIZE,
    STT_MODEL_ARCH,
    STT_MODEL_DIRNAME,
    STT_SAMPLE_RATE,
    STT_UPDATE_INTERVAL,
)


class OfflineSTTEngine:
    def __init__(self, enabled=True):
        self.event_queue = queue.Queue()
        self.enabled = enabled
        self.mic_transcriber = None
        self.listener = None

        if not self.enabled:
            self._publish("error", "Speech-to-text disabled")
            return

        model_path = ASSETS_DIR / STT_MODEL_DIRNAME
        required_files = [
            model_path / "encoder.ort",
            model_path / "decoder_kv.ort",
            model_path / "cross_kv.ort",
            model_path / "adapter.ort",
            model_path / "frontend.ort",
            model_path / "streaming_config.json",
            model_path / "tokenizer.bin",
        ]
        missing = [path.name for path in required_files if not path.exists()]
        if missing:
            self.enabled = False
            self._publish("error", f"Moonshine model missing: {', '.join(missing)}")
            return

        try:
            from moonshine_voice import MicTranscriber, ModelArch, TranscriptEventListener  # type: ignore[import]

            class STTListener(TranscriptEventListener):
                def __init__(self, publish):
                    self.publish = publish

                def on_line_text_changed(self, event):
                    text = event.line.text.strip()
                    if text:
                        self.publish("partial", text)

                def on_line_completed(self, event):
                    text = event.line.text.strip()
                    if text:
                        self.publish("final", text)

                def on_error(self, event):
                    self.publish("error", f"Moonshine error: {event.error}")

            self.mic_transcriber = MicTranscriber(
                model_path=str(model_path),
                model_arch=ModelArch(STT_MODEL_ARCH),
                update_interval=STT_UPDATE_INTERVAL,
                samplerate=STT_SAMPLE_RATE,
                blocksize=STT_BLOCK_SIZE,
                channels=1,
            )
            self.listener = STTListener(self._publish)
            self.mic_transcriber.add_listener(self.listener)
            self.mic_transcriber.start()
            self._publish("ready", "STT ready (Moonshine)")
        except Exception as exc:
            self.enabled = False
            self._publish("error", f"STT unavailable: {exc}")

    def _publish(self, event_type, message):
        self.event_queue.put((event_type, message))

    def stop(self):
        if self.mic_transcriber is None:
            return
        try:
            self.mic_transcriber.stop()
        except Exception:
            pass
        try:
            self.mic_transcriber.close()
        except Exception:
            pass
        self.mic_transcriber = None
