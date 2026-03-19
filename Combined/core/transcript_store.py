from collections import deque
from dataclasses import dataclass
from threading import Lock

# Keep at most this many entries in memory.  Old entries are silently dropped
# from the front of the deque once the limit is reached.
_MAX_ENTRIES = 500


@dataclass
class TranscriptEntry:
    source: str
    text: str


class TranscriptStore:
    def __init__(self):
        self._entries = deque(maxlen=_MAX_ENTRIES)
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
