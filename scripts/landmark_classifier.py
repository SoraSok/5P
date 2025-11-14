import numpy as np
import pickle
import os
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, confusion_matrix
import tensorflow as tf
from tensorflow import keras
import matplotlib.pyplot as plt

class LandmarkClassifier:
    def __init__(self, model_type='neural_network'):
        """
        Initialize landmark-based classifier
        
        Args:
            model_type: 'random_forest', 'svm', or 'neural_network'
        """
        self.model_type = model_type
        self.model = None
        self.scaler = StandardScaler()
        self.class_names = None
        
    def load_data(self, data_file):
        """Load landmark data from pickle file"""
        print(f"Loading data from {data_file}")
        
        with open(data_file, 'rb') as f:
            data_dict = pickle.load(f)
        
        X = []
        y = []
        class_names = []
        
        for class_name, landmarks_list in data_dict.items():
            if len(landmarks_list) > 0:
                X.extend(landmarks_list)
                y.extend([len(class_names)] * len(landmarks_list))
                class_names.append(class_name)
                print(f"Loaded {len(landmarks_list)} samples for {class_name}")
        
        X = np.array(X)
        y = np.array(y)
        
        print(f"Total samples: {len(X)}")
        print(f"Feature shape: {X.shape}")
        print(f"Classes: {class_names}")
        
        self.class_names = class_names
        return X, y
    
    def create_neural_network(self, input_dim, num_classes):
        """Create neural network for landmark classification"""
        model = keras.Sequential([
            keras.layers.Input(shape=(input_dim,)),
            keras.layers.Dense(128, activation='relu'),
            keras.layers.Dropout(0.3),
            keras.layers.Dense(64, activation='relu'),
            keras.layers.Dropout(0.3),
            keras.layers.Dense(32, activation='relu'),
            keras.layers.Dense(num_classes, activation='softmax')
        ])
        
        model.compile(
            optimizer='adam',
            loss='sparse_categorical_crossentropy',
            metrics=['accuracy']
        )
        
        return model
    
    def train(self, X, y, test_size=0.2, random_state=42):
        """Train the classifier"""
        print(f"\nTraining {self.model_type} classifier...")
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=random_state, stratify=y
        )
        
        print(f"Training samples: {len(X_train)}")
        print(f"Test samples: {len(X_test)}")
        
        # Scale features
        X_train_scaled = self.scaler.fit_transform(X_train)
        X_test_scaled = self.scaler.transform(X_test)
        
        # Train model based on type
        if self.model_type == 'random_forest':
            self.model = RandomForestClassifier(
                n_estimators=100,
                random_state=random_state,
                max_depth=10
            )
            self.model.fit(X_train_scaled, y_train)
            
        elif self.model_type == 'svm':
            self.model = SVC(
                kernel='rbf',
                C=1.0,
                random_state=random_state,
                probability=True
            )
            self.model.fit(X_train_scaled, y_train)
            
        elif self.model_type == 'neural_network':
            self.model = self.create_neural_network(X.shape[1], len(self.class_names))
            
            # Train neural network
            history = self.model.fit(
                X_train_scaled, y_train,
                epochs=100,
                batch_size=32,
                validation_data=(X_test_scaled, y_test),
                callbacks=[
                    keras.callbacks.EarlyStopping(
                        monitor='val_accuracy',
                        patience=10,
                        restore_best_weights=True
                    )
                ],
                verbose=1
            )
            
            # Plot training history
            self.plot_training_history(history)
        
        # Evaluate model
        self.evaluate(X_test_scaled, y_test)
        
        return X_test_scaled, y_test
    
    def evaluate(self, X_test, y_test):
        """Evaluate model performance"""
        print("\nEvaluating model...")
        
        if self.model_type == 'neural_network':
            y_pred = np.argmax(self.model.predict(X_test), axis=1)
            accuracy = np.mean(y_pred == y_test)
        else:
            y_pred = self.model.predict(X_test)
            accuracy = self.model.score(X_test, y_test)
        
        print(f"Test Accuracy: {accuracy:.4f}")
        
        print("\nClassification Report:")
        print(classification_report(y_test, y_pred, target_names=self.class_names))
        
        return accuracy
    
    def plot_training_history(self, history):
        """Plot training history for neural network"""
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
        
        ax1.plot(history.history['accuracy'], label='Training Accuracy')
        ax1.plot(history.history['val_accuracy'], label='Validation Accuracy')
        ax1.set_title('Model Accuracy')
        ax1.set_xlabel('Epoch')
        ax1.set_ylabel('Accuracy')
        ax1.legend()
        
        ax2.plot(history.history['loss'], label='Training Loss')
        ax2.plot(history.history['val_loss'], label='Validation Loss')
        ax2.set_title('Model Loss')
        ax2.set_xlabel('Epoch')
        ax2.set_ylabel('Loss')
        ax2.legend()
        
        plt.tight_layout()
        plt.savefig('landmark_training_history.png')
        plt.show()
    
    def predict(self, landmarks):
        """Predict class for given landmarks"""
        if self.model is None:
            raise ValueError("Model not trained yet!")
        
        landmarks_scaled = self.scaler.transform([landmarks])
        
        if self.model_type == 'neural_network':
            probabilities = self.model.predict(landmarks_scaled)[0]
            predicted_class_idx = np.argmax(probabilities)
            confidence = probabilities[predicted_class_idx]
        else:
            probabilities = self.model.predict_proba(landmarks_scaled)[0]
            predicted_class_idx = np.argmax(probabilities)
            confidence = probabilities[predicted_class_idx]
        
        predicted_class = self.class_names[predicted_class_idx]
        return predicted_class, confidence
    
    def save_model(self, model_path="models/landmark_classifier"):
        """Save trained model"""
        os.makedirs("models", exist_ok=True)
        
        if self.model_type == 'neural_network':
            self.model.save(f"{model_path}.h5")
        else:
            with open(f"{model_path}.pkl", 'wb') as f:
                pickle.dump(self.model, f)
        
        # Save scaler and class names
        with open(f"{model_path}_scaler.pkl", 'wb') as f:
            pickle.dump(self.scaler, f)
        
        with open(f"{model_path}_classes.pkl", 'wb') as f:
            pickle.dump(self.class_names, f)
        
        print(f"Model saved to {model_path}")
    
    def load_model(self, model_path="models/landmark_classifier"):
        """Load trained model"""
        if self.model_type == 'neural_network':
            self.model = keras.models.load_model(f"{model_path}.h5")
        else:
            with open(f"{model_path}.pkl", 'rb') as f:
                self.model = pickle.load(f)
        
        # Load scaler and class names
        with open(f"{model_path}_scaler.pkl", 'rb') as f:
            self.scaler = pickle.load(f)
        
        with open(f"{model_path}_classes.pkl", 'rb') as f:
            self.class_names = pickle.load(f)
        
        print(f"Model loaded from {model_path}")

def compare_models(data_file):
    """Compare different model types"""
    print("Comparing different classifier models...")
    
    models = ['random_forest', 'svm', 'neural_network']
    results = {}
    
    for model_type in models:
        print(f"\n{'='*50}")
        print(f"Testing {model_type}")
        print(f"{'='*50}")
        
        classifier = LandmarkClassifier(model_type)
        X, y = classifier.load_data(data_file)
        
        # Cross-validation
        X_scaled = classifier.scaler.fit_transform(X)
        if model_type == 'neural_network':
            # For neural network, use a simpler evaluation
            X_train, X_test, y_train, y_test = train_test_split(
                X_scaled, y, test_size=0.2, random_state=42, stratify=y
            )
            classifier.model = classifier.create_neural_network(X.shape[1], len(classifier.class_names))
            classifier.model.fit(X_train, y_train, epochs=50, verbose=0)
            accuracy = classifier.model.evaluate(X_test, y_test, verbose=0)[1]
        else:
            classifier.model = RandomForestClassifier() if model_type == 'random_forest' else SVC(probability=True)
            scores = cross_val_score(classifier.model, X_scaled, y, cv=5)
            accuracy = scores.mean()
        
        results[model_type] = accuracy
        print(f"{model_type} accuracy: {accuracy:.4f}")
    
    print(f"\n{'='*50}")
    print("RESULTS SUMMARY")
    print(f"{'='*50}")
    for model_type, accuracy in results.items():
        print(f"{model_type}: {accuracy:.4f}")
    
    best_model = max(results, key=results.get)
    print(f"\nBest model: {best_model} with accuracy {results[best_model]:.4f}")
    
    return best_model, results

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Train MediaPipe landmark classifier")
    parser.add_argument("--data_file", default="data/mediapipe_data/asl_alphabet_landmarks.pkl",
                       help="Path to landmark data file")
    
    args = parser.parse_args()
    
    # Check if data exists
    if not os.path.exists(args.data_file):
        print(f"Data file {args.data_file} not found!")
        print("Please run the dataset conversion first:")
        print("python scripts/convert_dataset.py")
    else:
        # Compare models
        best_model_type, results = compare_models(args.data_file)
        
        # Train the best model
        print(f"\nTraining final model using {best_model_type}...")
        classifier = LandmarkClassifier(best_model_type)
        X, y = classifier.load_data(args.data_file)
        classifier.train(X, y)
        classifier.save_model()
