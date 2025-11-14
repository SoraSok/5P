import cv2
import mediapipe as mp
import numpy as np
import os
import pickle
from tqdm import tqdm
import argparse

class DatasetConverter:
    def __init__(self):
        self.mp_hands = mp.solutions.hands
        self.hands = self.mp_hands.Hands(
            static_image_mode=True,
            max_num_hands=1,
            min_detection_confidence=0.5
        )
        print("MediaPipe detector initialized for dataset conversion.")
    
    def extract_landmarks_from_image(self, image_path):
        """Extract landmarks from a single image"""
        image = cv2.imread(image_path)
        if image is None:
            return None
        
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        results = self.hands.process(image_rgb)
        
        if results.multi_hand_landmarks:
            # Get the first hand
            hand_landmarks = results.multi_hand_landmarks[0]
            
            # Extract and flatten landmarks
            landmarks = []
            for landmark in hand_landmarks.landmark:
                landmarks.extend([landmark.x, landmark.y, landmark.z])
            
            return np.array(landmarks)
        
        return None
    
    def convert_dataset(self, dataset_path, output_path, max_samples_per_class=100):
        """Convert the ASL alphabet dataset to MediaPipe landmarks"""
        print(f"Converting dataset from {dataset_path} to {output_path}")
        
        # Create output directory
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        
        # Get all class directories
        class_dirs = [d for d in os.listdir(dataset_path) 
                     if os.path.isdir(os.path.join(dataset_path, d))]
        
        print(f"Found {len(class_dirs)} classes: {class_dirs}")
        
        converted_data = {}
        total_converted = 0
        
        for class_name in tqdm(class_dirs, desc="Converting classes"):
            class_path = os.path.join(dataset_path, class_name)
            
            # Get all image files in the class directory
            image_files = [f for f in os.listdir(class_path) 
                          if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
            
            # Limit samples per class
            if max_samples_per_class:
                image_files = image_files[:max_samples_per_class]
            
            print(f"\nProcessing class '{class_name}' with {len(image_files)} images...")
            
            class_landmarks = []
            successful_conversions = 0
            
            for img_file in tqdm(image_files, desc=f"Converting {class_name}", leave=False):
                img_path = os.path.join(class_path, img_file)
                landmarks = self.extract_landmarks_from_image(img_path)
                
                if landmarks is not None:
                    class_landmarks.append(landmarks)
                    successful_conversions += 1
            
            if class_landmarks:
                converted_data[class_name] = class_landmarks
                print(f"Successfully converted {successful_conversions}/{len(image_files)} images for '{class_name}'")
                total_converted += successful_conversions
            else:
                print(f"No landmarks extracted for '{class_name}'")
        
        # Save converted data
        with open(output_path, 'wb') as f:
            pickle.dump(converted_data, f)
        
        print(f"\nConversion completed!")
        print(f"Total samples converted: {total_converted}")
        print(f"Classes with data: {len(converted_data)}")
        print(f"Data saved to: {output_path}")
        
        # Print summary
        for class_name, landmarks in converted_data.items():
            print(f"  {class_name}: {len(landmarks)} samples")
        
        return converted_data
    
    def close(self):
        """Close MediaPipe resources"""
        self.hands.close()

def main():
    parser = argparse.ArgumentParser(description="Convert ASL alphabet dataset to MediaPipe landmarks")
    parser.add_argument("--dataset_path", default="data/kaggle_datasets/asl_alphabet_test", 
                       help="Path to ASL alphabet dataset")
    parser.add_argument("--output_path", default="data/mediapipe_data/kaggle_landmarks.pkl",
                       help="Output path for converted landmarks")
    parser.add_argument("--max_samples", type=int, default=100,
                       help="Maximum samples per class")
    
    args = parser.parse_args()
    
    # Check if dataset exists
    if not os.path.exists(args.dataset_path):
        print(f"Error: Dataset path {args.dataset_path} does not exist!")
        return
    
    # Initialize converter
    converter = DatasetConverter()
    
    try:
        # Convert dataset
        converted_data = converter.convert_dataset(
            args.dataset_path, 
            args.output_path, 
            args.max_samples
        )
        
        print(f"\nDataset conversion successful!")
        print(f"Ready to train MediaPipe model with {len(converted_data)} classes")
        
    except Exception as e:
        print(f"Error during conversion: {e}")
    finally:
        converter.close()

if __name__ == "__main__":
    main()
