import cv2
import mediapipe as mp
import numpy as np
import time
import tkinter as tk
from collections import deque
import os
import threading
import queue
from llama_cpp import Llama
import math
import pyttsx3 # NEW: Text-to-Speech library

# ==========================================
# MODULE A: GAZE & BLINK TRACKER
# ==========================================
class GazeTracker:
    def __init__(self, camera_index=0, smoothing_frames=20, ear_threshold=0.2, dwell_time=0.6):
        self.cap = cv2.VideoCapture(camera_index)
        self.mp_face_mesh = mp.solutions.face_mesh
        self.face_mesh = self.mp_face_mesh.FaceMesh(
            max_num_faces=1,
            refine_landmarks=True, 
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5
        )
        self.EAR_THRESHOLD = ear_threshold
        self.DWELL_TIME = dwell_time
        self.blink_start_time = None
        self.x_history = deque(maxlen=smoothing_frames)
        self.y_history = deque(maxlen=smoothing_frames)

    def _calculate_ear(self, landmarks, indices):
        v1 = np.linalg.norm(np.array([landmarks[indices[1]].x, landmarks[indices[1]].y]) - 
                            np.array([landmarks[indices[5]].x, landmarks[indices[5]].y]))
        v2 = np.linalg.norm(np.array([landmarks[indices[2]].x, landmarks[indices[2]].y]) - 
                            np.array([landmarks[indices[4]].x, landmarks[indices[4]].y]))
        h = np.linalg.norm(np.array([landmarks[indices[0]].x, landmarks[indices[0]].y]) - 
                           np.array([landmarks[indices[3]].x, landmarks[indices[3]].y]))
        return (v1 + v2) / (2.0 * h)

    def process_frame(self):
        success, frame = self.cap.read()
        if not success: return None, None, False

        frame = cv2.flip(frame, 1)
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        rgb_frame.flags.writeable = False
        results = self.face_mesh.process(rgb_frame)

        x_center, y_center, is_clicking = None, None, False

        if results.multi_face_landmarks:
            landmarks = results.multi_face_landmarks[0].landmark
            
            left_ear = self._calculate_ear(landmarks, [33, 160, 158, 133, 153, 144])
            right_ear = self._calculate_ear(landmarks, [362, 385, 387, 263, 373, 380])
            is_blinking = ((left_ear + right_ear) / 2.0) < self.EAR_THRESHOLD

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
                elif (time.time() - self.blink_start_time) >= self.DWELL_TIME:
                    is_clicking = True
                    self.blink_start_time = time.time() + 1.5 
            else:
                self.blink_start_time = None
                self.x_history.append(x_raw)
                self.y_history.append(y_raw)
                x_center = sum(self.x_history) / len(self.x_history)
                y_center = sum(self.y_history) / len(self.y_history)

        return x_center, y_center, is_clicking

    def release(self):
        self.cap.release()

# ==========================================
# MODULE C: LOCAL LLM THREAD ENGINE
# ==========================================
class LLMPredictor:
    def __init__(self, model_path="tinyllama-1.1b-chat-v1.0.Q4_K_M.gguf"):
        self.llm = Llama(model_path=model_path, n_ctx=256, n_threads=2, verbose=False)
        self.request_queue = queue.Queue()
        self.result_queue = queue.Queue()
        self.worker_thread = threading.Thread(target=self._worker, daemon=True)
        self.worker_thread.start()

    def _worker(self):
        while True:
            sentence = self.request_queue.get()
            if sentence is None: break
            prompt = f"<|system|>\nYou are an AAC typing assistant. Provide ONLY the next 1 to 2 words to complete the sentence. No punctuation. No explanation.</s>\n<|user|>\nComplete this: {sentence}</s>\n<|assistant|>\n"
            try:
                output = self.llm(prompt, max_tokens=3, stop=["\n", ".", ","], temperature=0.2)
                prediction = output['choices'][0]['text'].strip().upper()
                prediction = prediction.replace('"', '').replace("'", "")
                if prediction:
                    self.result_queue.put(prediction)
            except Exception as e:
                print(f"LLM Error: {e}")

    def request_next_words(self, current_text):
        while not self.request_queue.empty(): 
            try: self.request_queue.get_nowait()
            except queue.Empty: break
        self.request_queue.put(current_text)

    def check_for_results(self):
        if not self.result_queue.empty():
            return self.result_queue.get()
        return None

# ==========================================
# MODULE D: FREQUENCY TRIE
# ==========================================
class TrieNode:
    def __init__(self):
        self.children = {}
        self.is_end_of_word = False
        self.frequency = 0

class FrequencyTrie:
    def __init__(self, dictionary_file="frequencies.txt"):
        self.root = TrieNode()
        self.load_dictionary(dictionary_file)

    def insert(self, word, frequency):
        node = self.root
        for char in word:
            if char not in node.children: node.children[char] = TrieNode()
            node = node.children[char]
        node.is_end_of_word = True
        node.frequency = frequency

    def load_dictionary(self, filepath):
        if not os.path.exists(filepath):
            defaults = [("HELLO", 100), ("HELP", 90), ("WATER", 80), ("FOOD", 70), ("PLEASE", 60)]
            for word, freq in defaults: self.insert(word, freq)
            return
        with open(filepath, 'r') as file:
            for line in file:
                parts = line.strip().split(',')
                if len(parts) == 2:
                    try: self.insert(parts[0].strip().upper(), int(parts[1].strip()))
                    except ValueError: continue

    def search_prefix(self, prefix):
        node = self.root
        for char in prefix:
            if char not in node.children: return []
            node = node.children[char]
        return self._get_all_words_with_freq(node, prefix)

    def _get_all_words_with_freq(self, node, prefix):
        results = []
        if node.is_end_of_word: results.append((prefix, node.frequency))
        for char, child in node.children.items():
            results.extend(self._get_all_words_with_freq(child, prefix + char))
        return results
        
    def get_top_k_predictions(self, prefix, k=4):
        if not prefix: return []
        matches = self.search_prefix(prefix)
        matches.sort(key=lambda x: x[1], reverse=True)
        return [word for word, freq in matches[:k]]

# ==========================================
# MODULE E: TEXT TO SPEECH ENGINE
# ==========================================
class TTSEngine:
    def __init__(self):
        self.speech_queue = queue.Queue()
        self.worker_thread = threading.Thread(target=self._worker, daemon=True)
        self.worker_thread.start()

    def _worker(self):
        # Initialize engine inside the thread
        engine = pyttsx3.init()
        # Slow down the reading speed slightly for better clarity (Default is ~200)
        engine.setProperty('rate', 150) 
        while True:
            text = self.speech_queue.get()
            if text is None: break
            engine.say(text)
            engine.runAndWait()

    def speak(self, text):
        self.speech_queue.put(text)

# ==========================================
# MODULE B: GUI & LOGIC (TKINTER)
# ==========================================
class KeyboardGUI:
    def __init__(self, root, tracker, llm_engine, tts_engine):
        self.root = root
        self.tracker = tracker
        self.llm_engine = llm_engine
        self.tts_engine = tts_engine # Added TTS Engine
        self.root.title("AAC Hybrid Keyboard")
        self.root.attributes("-fullscreen", True)
        
        self.screen_w = self.root.winfo_screenwidth()
        self.screen_h = self.root.winfo_screenheight()
        self.cursor_x, self.cursor_y = self.screen_w / 2, self.screen_h / 2
        
        self.state = "CALIBRATING"
        self.calib_points = ["TOP-LEFT", "TOP-RIGHT", "BOTTOM-LEFT", "BOTTOM-RIGHT"]
        self.calib_index = 0
        self.calib_data = {'x_min': 1.0, 'x_max': 0.0, 'y_min': 1.0, 'y_max': 0.0}
        
        self.current_text = ""
        self.current_word = ""
        self.predictions = ["", "", "", ""]
        self.predictor = FrequencyTrie("frequencies.txt")

        self.canvas = tk.Canvas(self.root, width=self.screen_w, height=self.screen_h, bg="#1E1E1E", highlightthickness=0)
        self.canvas.pack()
        self.keys = []
        self.setup_ui()
        self.update_loop()

    def setup_ui(self):
        self.canvas.delete("all")
        self.keys = []
        
        if self.state == "CALIBRATING":
            target = self.calib_points[self.calib_index]
            self.canvas.create_text(self.screen_w//2, self.screen_h//2, text=f"Look at the {target} corner\nand BLINK to calibrate.", fill="white", font=("Arial", 36), justify="center")
            r = 40
            if target == "TOP-LEFT": cx, cy = r+10, r+10
            elif target == "TOP-RIGHT": cx, cy = self.screen_w-r-10, r+10
            elif target == "BOTTOM-LEFT": cx, cy = r+10, self.screen_h-r-10
            else: cx, cy = self.screen_w-r-10, self.screen_h-r-10
            self.canvas.create_oval(cx-r, cy-r, cx+r, cy+r, fill="red", outline="white", width=4, tags="target")

        elif self.state == "TYPING":
            self.canvas.create_rectangle(50, 20, self.screen_w-50, 120, fill="#333333", outline="gray")
            self.canvas.create_text(60, 70, text=self.current_text + "|", fill="white", font=("Arial", 40), anchor="w", tags="output_text")
            
            pred_w = (self.screen_w - 100) / 4
            for i in range(4):
                px1 = 50 + (i * pred_w)
                py1 = 140
                px2 = px1 + pred_w - 10
                py2 = 220
                text = self.predictions[i] if i < len(self.predictions) else ""
                bg_col = "#4B0082" if " " in text else "#444444" 
                self.canvas.create_rectangle(px1, py1, px2, py2, fill=bg_col, outline="#555555", tags=f"pred_bg_{i}")
                self.canvas.create_text((px1+px2)//2, (py1+py2)//2, text=text, fill="#00FFCC", font=("Arial", 28), tags=f"pred_txt_{i}")
                self.keys.append((px1, py1, px2, py2, f"PRED_{i}"))

            # --- UPDATED LAYOUT WITH SPEAK BUTTON ---
            layout = [
                ["Q", "W", "E", "R", "T", "Y", "U", "I", "O", "P"],
                ["A", "S", "D", "F", "G", "H", "J", "K", "L", "DEL"],
                ["Z", "X", "C", "V", "B", "N", "M", "SPACE", "CLR", "SPEAK"] 
            ]
            start_y = 260
            key_h = (self.screen_h - start_y - 50) // 3
            for row_idx, row in enumerate(layout):
                key_w = (self.screen_w - 100) // len(row)
                for col_idx, char in enumerate(row):
                    x1 = 50 + (col_idx * key_w)
                    y1 = start_y + (row_idx * key_h)
                    x2 = x1 + key_w - 10
                    y2 = y1 + key_h - 10
                    
                    # Colors
                    if char in ["DEL", "CLR"]: bg_color = "#5A2A2A"
                    elif char == "SPACE": bg_color = "#2A5A2A"
                    elif char == "SPEAK": bg_color = "#0055A4" # Blue color for speech
                    else: bg_color = "#2A2A2A"
                    
                    self.canvas.create_rectangle(x1, y1, x2, y2, fill=bg_color, outline="#555555", tags=f"key_bg_{char}")
                    # Make SPEAK text a bit smaller to fit
                    font_size = 24 if char == "SPEAK" else 36
                    self.canvas.create_text((x1+x2)//2, (y1+y2)//2, text=char, fill="white", font=("Arial", font_size, "bold"), tags=f"key_txt_{char}")
                    self.keys.append((x1, y1, x2, y2, char))

    def update_predictions(self):
        if self.current_word:
            matches = self.predictor.get_top_k_predictions(self.current_word, k=4)
            self.predictions = matches + [""] * (4 - len(matches))
        else:
            self.predictions = ["", "", "", ""]
        self.setup_ui() 

    def process_key_click(self, char):
        if char.startswith("PRED_"):
            idx = int(char.split("_")[1])
            if self.predictions[idx]:
                if " " in self.predictions[idx]:
                    self.current_text += self.predictions[idx] + " "
                else:
                    self.current_text = self.current_text[:-len(self.current_word)] + self.predictions[idx] + " "
                self.current_word = ""
                self.llm_engine.request_next_words(self.current_text)
        elif char == "SPACE":
            self.current_text += " "
            self.current_word = ""
            if len(self.current_text.strip()) > 2:
                self.llm_engine.request_next_words(self.current_text)
        elif char == "DEL":
            if self.current_text:
                if self.current_text[-1] != " ": self.current_word = self.current_word[:-1]
                self.current_text = self.current_text[:-1]
        elif char == "CLR":
            self.current_text = ""
            self.current_word = ""
        # --- NEW SPEAK LOGIC ---
        elif char == "SPEAK":
            if self.current_text.strip():
                # Send current text to the audio thread
                self.tts_engine.speak(self.current_text)
        # -----------------------
        else:
            self.current_text += char
            self.current_word += char
        self.update_predictions()

    def update_loop(self):
        llm_result = self.llm_engine.check_for_results()
        if llm_result and not self.current_word:
            self.predictions[0] = llm_result
            self.setup_ui()

        raw_x, raw_y, is_clicking = self.tracker.process_frame()
        if raw_x is not None and raw_y is not None:
            if self.state == "CALIBRATING":
                if is_clicking:
                    self.calib_data['x_min'] = min(self.calib_data['x_min'], raw_x)
                    self.calib_data['x_max'] = max(self.calib_data['x_max'], raw_x)
                    self.calib_data['y_min'] = min(self.calib_data['y_min'], raw_y)
                    self.calib_data['y_max'] = max(self.calib_data['y_max'], raw_y)
                    self.calib_index += 1
                    if self.calib_index >= len(self.calib_points): self.state = "TYPING"
                    self.setup_ui()

            elif self.state == "TYPING":
                target_x = np.interp(raw_x, [self.calib_data['x_min'], self.calib_data['x_max']], [0, self.screen_w])
                target_y = np.interp(raw_y, [self.calib_data['y_min'], self.calib_data['y_max']], [0, self.screen_h])
                
                move_distance = math.hypot(target_x - self.cursor_x, target_y - self.cursor_y)
                
                if move_distance > 25:
                    alpha = 0.022
                    self.cursor_x = (alpha * target_x) + ((1 - alpha) * self.cursor_x)
                    self.cursor_y = (alpha * target_y) + ((1 - alpha) * self.cursor_y)
                
                self.canvas.delete("cursor")
                self.canvas.create_oval(self.cursor_x-10, self.cursor_y-10, self.cursor_x+10, self.cursor_y+10, fill="yellow", tags="cursor")

                hovered_char = None
                for (x1, y1, x2, y2, char) in self.keys:
                    if x1 <= self.cursor_x <= x2 and y1 <= self.cursor_y <= y2:
                        hovered_char = char
                        self.canvas.itemconfig(f"key_bg_{char}", fill="#007ACC") 
                        break 
                
                for (x1, y1, x2, y2, char) in self.keys:
                    if char != hovered_char:
                        if char in ["DEL", "CLR"]: self.canvas.itemconfig(f"key_bg_{char}", fill="#5A2A2A")
                        elif char == "SPACE": self.canvas.itemconfig(f"key_bg_{char}", fill="#2A5A2A")
                        elif char == "SPEAK": self.canvas.itemconfig(f"key_bg_{char}", fill="#0055A4")
                        elif char.startswith("PRED_"):
                            bg_col = "#4B0082" if " " in self.canvas.itemcget(f"pred_txt_{char.split('_')[1]}", "text") else "#444444"
                            self.canvas.itemconfig(f"key_bg_{char}", fill=bg_col)
                        else: self.canvas.itemconfig(f"key_bg_{char}", fill="#2A2A2A")

                if is_clicking and hovered_char:
                    self.canvas.itemconfig(f"key_bg_{hovered_char}", fill="#00FF00") 
                    self.root.update() 
                    self.process_key_click(hovered_char)

        self.root.after(15, self.update_loop)

if __name__ == "__main__":
    root = tk.Tk()
    tracker = GazeTracker(camera_index=0)
    
    try:
        llm_engine = LLMPredictor()
    except Exception as e:
        print(f"Failed to load LLM: {e}")
        exit()
        
    print("Initializing TTS Audio Engine...")
    tts_engine = TTSEngine()
    
    app = KeyboardGUI(root, tracker, llm_engine, tts_engine)
    root.bind("<Escape>", lambda e: root.destroy())
    
    try: root.mainloop() 
    finally: tracker.release()
