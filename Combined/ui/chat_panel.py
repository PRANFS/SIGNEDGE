import tkinter as tk
from tkinter import ttk

class ChatPanel:
    def __init__(self, parent):
        self.container = ttk.Frame(parent)
        self.container.pack(fill="both", expand=True)

        header = ttk.Label(
            self.container,
            text="Conversation Transcript",
            font=("Segoe UI", 13, "bold"),
        )
        header.pack(anchor="w", padx=8, pady=(8, 4))

        text_frame = ttk.Frame(self.container)
        text_frame.pack(fill="both", expand=True, padx=8, pady=(0, 16))

        self.text = tk.Text(
            text_frame,
            wrap="word",
            state="disabled",
            font=("Consolas", 11),
            bg="#0f1115",
            fg="#f3f4f6",
            insertbackground="#f3f4f6",
        )
        self.scrollbar = tk.Scrollbar(
            text_frame,
            orient="vertical",
            command=self.text.yview,
            width=18,
            bg="#2b2f36",
            activebackground="#4b5563",
            troughcolor="#111317",
            relief="flat",
            borderwidth=0,
        )
        self.text.configure(yscrollcommand=self.scrollbar.set)

        self.text.pack(side="left", fill="both", expand=True)
        self.scrollbar.pack(side="right", fill="y", pady=(0, 12))

        # Colour scheme for each source type
        self.text.tag_configure("STT",      foreground="#7ee787")          # green   — microphone
        self.text.tag_configure("TTS_ASL",  foreground="#79c0ff")          # blue    — ASL typed output
        self.text.tag_configure("TTS_EYE",  foreground="#56d4dd")          # cyan    — eye-typed output
        self.text.tag_configure("TTS",      foreground="#79c0ff")          # blue    — generic TTS
        self.text.tag_configure("SYS",      foreground="#d2a8ff")          # purple  — system events

        self._last_count = 0

    def refresh(self, entries):
        """Append only new entries instead of rebuilding from scratch every tick.

        BUG FIX: the original implementation called self.text.delete("1.0", "end")
        and rewrote every entry on every refresh() call.  For a session with 200+
        entries this does 200 inserts per frame — O(n) work per tick.  The new
        approach is O(1) per tick: we only insert the entries that are genuinely
        new.

        If the backing deque rolled over (entries were dropped from the front) we
        fall back to a full redraw, which is rare in practice.
        """
        n = len(entries)
        if n == self._last_count:
            return  # nothing new

        self.text.configure(state="normal")

        if n < self._last_count:
            # Deque rolled over — full redraw
            self.text.delete("1.0", "end")
            start = 0
        else:
            start = self._last_count

        for entry in entries[start:]:
            source = entry.source.upper()
            tag = self._source_tag(source)
            self.text.insert("end", f"[{source}] ", tag)
            self.text.insert("end", f"{entry.text}\n")

        self.text.see("end")
        self.text.configure(state="disabled")
        self._last_count = n

    @staticmethod
    def _source_tag(source: str) -> str:
        """Map a source string to the nearest colour tag."""
        if source.startswith("STT"):
            return "STT"
        if source == "TTS_ASL":
            return "TTS_ASL"
        if source == "TTS_EYE":
            return "TTS_EYE"
        if source.startswith("TTS"):
            return "TTS"
        return "SYS"
