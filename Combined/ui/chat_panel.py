import tkinter as tk
from tkinter import ttk


class ChatPanel:
    def __init__(self, parent):
        self.container = ttk.Frame(parent)
        self.container.pack(fill="both", expand=True)

        header = ttk.Label(self.container, text="Conversation Transcript", font=("Segoe UI", 13, "bold"))
        header.pack(anchor="w", padx=8, pady=(8, 4))

        text_frame = ttk.Frame(self.container)
        text_frame.pack(fill="both", expand=True, padx=8, pady=(0, 8))

        self.text = tk.Text(
            text_frame,
            wrap="word",
            state="disabled",
            font=("Consolas", 11),
            bg="#0f1115",
            fg="#f3f4f6",
            insertbackground="#f3f4f6",
        )
        self.scrollbar = ttk.Scrollbar(text_frame, orient="vertical", command=self.text.yview)
        self.text.configure(yscrollcommand=self.scrollbar.set)

        self.text.pack(side="left", fill="both", expand=True)
        self.scrollbar.pack(side="right", fill="y")

        self.text.tag_configure("STT", foreground="#7ee787")
        self.text.tag_configure("TTS", foreground="#79c0ff")
        self.text.tag_configure("SYS", foreground="#d2a8ff")

        self._last_count = 0

    def refresh(self, entries):
        if len(entries) == self._last_count:
            return

        self.text.configure(state="normal")
        self.text.delete("1.0", "end")
        for entry in entries:
            source = entry.source.upper()
            tag = "SYS"
            if source.startswith("STT"):
                tag = "STT"
            elif source.startswith("TTS"):
                tag = "TTS"
            self.text.insert("end", f"[{source}] ", tag)
            self.text.insert("end", f"{entry.text}\n")
        self.text.see("end")
        self.text.configure(state="disabled")

        self._last_count = len(entries)
