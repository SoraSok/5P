import cv2
import mediapipe as mp
import numpy as np
import pickle
import os
from collections import defaultdict

class MediaPipeHandDetector:
    def __init__(self):
        """Initialize MediaPipe hand detection"""
        self.mp_hands = mp.solutions.hands
        self.mp_drawing = mp.solutions.drawing_utils
        self.mp_drawing_styles = mp.solutions.drawing_styles
        
        # Initialize hands solution
        self.hands = self.mp_hands.Hands(
            static_image_mode=False,
            max_num_hands=1,
            min_detection_confidence=0.7,
            min_tracking_confidence=0.5
        )
        
        # Data collection
        self.collected_data = defaultdict(list)
        self.current_class = None
        self.collection_mode = False
        
    def extract_landmarks(self, frame):
        """Extract hand landmarks from frame"""
        # Convert BGR to RGB
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        
        # Process frame
        results = self.hands.process(rgb_frame)
        
        landmarks = None
        hand_landmarks = None
        
        if results.multi_hand_landmarks:
            # Get first hand
            hand_landmarks = results.multi_hand_landmarks[0]
            
            # Extract landmark coordinates
            landmarks = []
            for landmark in hand_landmarks.landmark:
                landmarks.extend([landmark.x, landmark.y, landmark.z])
            
            landmarks = np.array(landmarks)
        
        return landmarks, hand_landmarks
    
    def draw_landmarks(self, frame, hand_landmarks):
        """Draw hand landmarks on frame"""
        if hand_landmarks:
            self.mp_drawing.draw_landmarks(
                frame,
                hand_landmarks,
                self.mp_hands.HAND_CONNECTIONS,
                self.mp_drawing_styles.get_default_hand_landmarks_style(),
                self.mp_drawing_styles.get_default_hand_connections_style()
            )
        return frame
    
    def start_collection(self, class_name):
        """Start collecting data for a specific class"""
        self.current_class = class_name
        self.collection_mode = True
        print(f"Started collecting data for class: {class_name}")
        print("Press 'c' to capture, 'q' to quit, 's' to save")
    
    def stop_collection(self):
        """Stop data collection"""
        self.collection_mode = False
        self.current_class = None
        print("Data collection stopped")
    
    def capture_sample(self):
        """Capture a landmark sample"""
        if self.current_class and self.collection_mode:
            # This will be called from the main loop when 'c' is pressed
            return True
        return False
    
    def save_data(self, filename="data/mediapipe_data/collected_landmarks.pkl"):
        """Save collected landmark data"""
        os.makedirs(os.path.dirname(filename), exist_ok=True)
        
        with open(filename, 'wb') as f:
            pickle.dump(dict(self.collected_data), f)
        
        print(f"Data saved to {filename}")
        print(f"Total classes: {len(self.collected_data)}")
        for class_name, samples in self.collected_data.items():
            print(f"  {class_name}: {len(samples)} samples")
    
    def load_data(self, filename="data/mediapipe_data/collected_landmarks.pkl"):
        """Load collected landmark data"""
        if os.path.exists(filename):
            with open(filename, 'rb') as f:
                self.collected_data = defaultdict(list, pickle.load(f))
            print(f"Data loaded from {filename}")
            return True
        return False
    
    def add_sample(self, landmarks):
        """Add a landmark sample to current class"""
        if self.current_class and landmarks is not None:
            self.collected_data[self.current_class].append(landmarks)
            print(f"Sample added to {self.current_class}. Total: {len(self.collected_data[self.current_class])}")
    
    def close(self):
        """Close MediaPipe resources"""
        if hasattr(self, 'hands'):
            self.hands.close()

def main():
    """Data collection tool"""
    print("ASL Landmark Data Collection Tool")
    print("=" * 40)
    
    detector = MediaPipeHandDetector()
    
    # Load existing data
    detector.load_data()
    
    cap = cv2.VideoCapture(0)
    
    # Set window properties
    cv2.namedWindow('ASL Data Collection', cv2.WINDOW_NORMAL)
    cv2.resizeWindow('ASL Data Collection', 800, 600)
    
    print("\nAvailable commands:")
    print("1-9: Select class (A=1, B=2, ..., I=9)")
    print("c: Capture sample")
    print("s: Save data")
    print("q or ESC: Quit")
    print("r: Reset current class")
    
    # Class mapping
    classes = ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'K', 'L', 'M', 
               'N', 'O', 'P', 'Q', 'R', 'S', 'T', 'U', 'V', 'W', 'X', 'Y', 'Z',
               'del', 'nothing', 'space']
    
    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                print("Failed to read from camera!")
                break
            
            frame = cv2.flip(frame, 1)
            
            # Extract landmarks
            landmarks, hand_landmarks = detector.extract_landmarks(frame)
            
            # Draw landmarks
            if hand_landmarks:
                frame = detector.draw_landmarks(frame, hand_landmarks)
            
            # Add status text
            status_text = f"Class: {detector.current_class or 'None'}"
            cv2.putText(frame, status_text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            
            if detector.current_class:
                sample_count = len(detector.collected_data[detector.current_class])
                cv2.putText(frame, f"Samples: {sample_count}", (10, 70), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            
            # Add instructions
            cv2.putText(frame, "Press 'q' to quit, 'ESC' to close", (10, frame.shape[0] - 20), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
            
            cv2.imshow('ASL Data Collection', frame)
            
            key = cv2.waitKey(1) & 0xFF
            
            if key == ord('q') or key == 27:  # 'q' or ESC
                print("Exiting...")
                break
            elif key == ord('s'):
                detector.save_data()
            elif key == ord('r'):
                detector.stop_collection()
            elif key == ord('c'):
                if landmarks is not None:
                    detector.add_sample(landmarks)
                else:
                    print("No hand detected!")
            elif ord('1') <= key <= ord('9'):
                class_idx = key - ord('1')
                if class_idx < len(classes):
                    detector.start_collection(classes[class_idx])
            elif key >= ord('a') and key <= ord('z'):
                # Handle letters a-z
                letter = chr(key).upper()
                if letter in classes:
                    detector.start_collection(letter)
            elif key == ord(' '):
                detector.start_collection('space')
            elif key == ord('d'):
                detector.start_collection('del')
            elif key == ord('n'):
                detector.start_collection('nothing')
    
    except KeyboardInterrupt:
        print("\nInterrupted by user!")
    except Exception as e:
        print(f"Error: {e}")
    finally:
        # Always clean up
        print("Cleaning up...")
        cap.release()
        cv2.destroyAllWindows()
        detector.close()
        
        # Save data before exit
        detector.save_data()
        print("Data collection completed!")

if __name__ == "__main__":
    main()
