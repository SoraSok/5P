import tkinter as tk
from tkinter import ttk
import cv2
from PIL import Image, ImageTk
import numpy as np
import pickle
import os
from scripts.mediapipe_hand_detector import MediaPipeHandDetector
from scripts.landmark_classifier import LandmarkClassifier

class ASLTranslator:
    def __init__(self, root):
        self.root = root
        self.root.title("ASL Fingerspelling Translator - MediaPipe Edition")
        self.root.geometry("1000x700")
        self.root.configure(bg='#2C3E50')
        
        print("Initializing MediaPipe hand detector...")
        self.detector = MediaPipeHandDetector()
        
        print("Loading landmark classifier...")
        try:
            self.classifier = LandmarkClassifier()
            self.classifier.load_model()
            print(f"Model loaded successfully! Classes: {self.classifier.class_names}")
        except Exception as e:
            print(f"Error loading model: {e}")
            print("Please train the landmark classifier first!")
            return
        
        self.cap = cv2.VideoCapture(0)
        
        self.current_prediction = None
        self.conversation = []
        self.prediction_history = []
        
        self.create_widgets()
        
        self.update_frame()
    
    def create_widgets(self):
        title = tk.Label(
            self.root, 
            text="ASL Fingerspelling Translator - MediaPipe Edition",
            font=("Arial", 20, "bold"),
            bg='#2C3E50',
            fg='white'
        )
        title.pack(pady=10)
        
        main_frame = tk.Frame(self.root, bg='#2C3E50')
        main_frame.pack(fill=tk.BOTH, expand=True, padx=20)
        
        video_frame = tk.Frame(main_frame, bg='#34495E', relief=tk.RAISED, bd=2)
        video_frame.pack(side=tk.LEFT, padx=10, pady=10)
        
        self.video_label = tk.Label(video_frame, bg='black')
        self.video_label.pack(padx=5, pady=5)
        
        right_frame = tk.Frame(main_frame, bg='#2C3E50')
        right_frame.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=10)
        
        pred_frame = tk.Frame(right_frame, bg='#34495E', relief=tk.RAISED, bd=2)
        pred_frame.pack(fill=tk.X, pady=10)
        
        tk.Label(
            pred_frame,
            text="Current Sign:",
            font=("Arial", 16),
            bg='#34495E',
            fg='white'
        ).pack(pady=5)
        
        self.pred_label = tk.Label(
            pred_frame,
            text="-",
            font=("Arial", 72, "bold"),
            bg='#34495E',
            fg='#2ECC71'
        )
        self.pred_label.pack(pady=10)
        
        self.confidence_label = tk.Label(
            pred_frame,
            text="Confidence: -",
            font=("Arial", 14),
            bg='#34495E',
            fg='#ECF0F1'
        )
        self.confidence_label.pack(pady=5)
        
        # Add status label
        self.status_label = tk.Label(
            pred_frame,
            text="Status: Ready",
            font=("Arial", 12),
            bg='#34495E',
            fg='#F39C12'
        )
        self.status_label.pack(pady=5)
        
        btn_frame = tk.Frame(right_frame, bg='#2C3E50')
        btn_frame.pack(pady=10)
        
        tk.Button(
            btn_frame,
            text="Add Letter (Space)",
            command=self.add_letter,
            font=("Arial", 12),
            bg='#3498DB',
            fg='white',
            padx=20,
            pady=10
        ).pack(side=tk.LEFT, padx=5)
        
        tk.Button(
            btn_frame,
            text="Space",
            command=self.add_space,
            font=("Arial", 12),
            bg='#95A5A6',
            fg='white',
            padx=20,
            pady=10
        ).pack(side=tk.LEFT, padx=5)
        
        tk.Button(
            btn_frame,
            text="Clear",
            command=self.clear_history,
            font=("Arial", 12),
            bg='#E74C3C',
            fg='white',
            padx=20,
            pady=10
        ).pack(side=tk.LEFT, padx=5)
        
        # Add data collection button
        tk.Button(
            btn_frame,
            text="Collect Data",
            command=self.open_data_collection,
            font=("Arial", 12),
            bg='#9B59B6',
            fg='white',
            padx=20,
            pady=10
        ).pack(side=tk.LEFT, padx=5)
        
        history_frame = tk.Frame(right_frame, bg='#34495E', relief=tk.RAISED, bd=2)
        history_frame.pack(fill=tk.BOTH, expand=True, pady=10)
        
        tk.Label(
            history_frame,
            text="Your Message:",
            font=("Arial", 14),
            bg='#34495E',
            fg='white'
        ).pack(pady=5)
        
        self.history_text = tk.Text(
            history_frame,
            height=8,
            width=40,
            font=("Arial", 16),
            bg='#ECF0F1',
            fg='#2C3E50',
            wrap=tk.WORD
        )
        self.history_text.pack(padx=10, pady=10, fill=tk.BOTH, expand=True)
        
        self.root.bind('<space>', lambda e: self.add_letter())
        self.root.bind('<BackSpace>', lambda e: self.backspace())
    
    def smooth_prediction(self, prediction):
        """Smooth predictions using last 3 frames"""
        self.prediction_history.append(prediction)
        if len(self.prediction_history) > 3:
            self.prediction_history.pop(0)
        
        from collections import Counter
        counts = Counter(self.prediction_history)
        return counts.most_common(1)[0][0]
    
    def update_frame(self):
        ret, frame = self.cap.read()
        if ret:
            frame = cv2.flip(frame, 1)
            
            # Extract hand landmarks
            landmarks, hand_landmarks = self.detector.extract_landmarks(frame)
            
            if landmarks is not None and hand_landmarks is not None:
                # Draw landmarks
                frame = self.detector.draw_landmarks(frame, hand_landmarks)
                
                # Predict sign
                try:
                    letter, confidence = self.classifier.predict(landmarks)
                    
                    if confidence > 0.6:  # Lower threshold for landmarks
                        smoothed_letter = self.smooth_prediction(letter)
                        
                        self.current_prediction = smoothed_letter
                        self.pred_label.config(text=smoothed_letter)
                        self.confidence_label.config(
                            text=f"Confidence: {confidence*100:.1f}%"
                        )
                        self.status_label.config(text="Status: Hand detected", fg='#2ECC71')
                    else:
                        self.pred_label.config(text="-")
                        self.confidence_label.config(text="Hold steady...")
                        self.status_label.config(text="Status: Low confidence", fg='#F39C12')
                        self.prediction_history = []
                        
                except Exception as e:
                    print(f"Prediction error: {e}")
                    self.pred_label.config(text="-")
                    self.confidence_label.config(text="Error in prediction")
                    self.status_label.config(text="Status: Prediction error", fg='#E74C3C')
            else:
                self.pred_label.config(text="-")
                self.confidence_label.config(text="No hand detected")
                self.status_label.config(text="Status: No hand detected", fg='#E74C3C')
                self.prediction_history = []
            
            # Add instruction text
            cv2.putText(frame, "Show ASL sign in front of camera", (10, 30), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
            
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            img = Image.fromarray(frame)
            img = img.resize((640, 480))
            imgtk = ImageTk.PhotoImage(image=img)
            self.video_label.imgtk = imgtk
            self.video_label.configure(image=imgtk)
        
        self.root.after(30, self.update_frame)
    
    def add_letter(self):
        if self.current_prediction and self.current_prediction != '-':
            self.conversation.append(self.current_prediction)
            self.update_history()
    
    def add_space(self):
        self.conversation.append(' ')
        self.update_history()
    
    def backspace(self):
        if self.conversation:
            self.conversation.pop()
            self.update_history()
    
    def update_history(self):
        text = ''.join(self.conversation)
        self.history_text.delete(1.0, tk.END)
        self.history_text.insert(1.0, text)
    
    def clear_history(self):
        self.conversation = []
        self.update_history()
    
    def open_data_collection(self):
        """Open data collection tool"""
        import subprocess
        import sys
        
        try:
            subprocess.Popen([sys.executable, "scripts/mediapipe_hand_detector.py"])
            print("Data collection tool opened in new window")
        except Exception as e:
            print(f"Error opening data collection tool: {e}")
    
    def __del__(self):
        if hasattr(self, 'cap'):
            self.cap.release()
        if hasattr(self, 'detector'):
            self.detector.close()

if __name__ == "__main__":
    root = tk.Tk()
    app = ASLTranslator(root)
    root.mainloop()