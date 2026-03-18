from collections import deque
from dataclasses import dataclass
from threading import Lock


@dataclass
class TranscriptEntry:
    source: str
    text: str


class TranscriptStore:
    def __init__(self):
        self._entries = deque()
        self._lock = Lock()

    def append(self, source: str, text: str):
        cleaned = (text or "").strip()
        if not cleaned:
            return
        with self._lock:
            self._entries.append(TranscriptEntry(source=source, text=cleaned))

    def snapshot(self):
        with self._lock:
            return list(self._entries)
