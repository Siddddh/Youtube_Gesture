import cv2
import numpy as np
import tensorflow as tf
from tensorflow import keras
import pyautogui
import time
import pickle
import json
from collections import deque
from utils import HandLandmarkExtractor, mp_drawing, mp_hands
import threading
import queue

class AdvancedGestureController:
    def __init__(self, model_path='models/gesture_model.h5'):
        # Load model and preprocessors
        self.model = keras.models.load_model(model_path)
        
        with open('models/label_encoder.pkl', 'rb') as f:
            self.label_encoder = pickle.load(f)
        
        with open('models/scaler.pkl', 'rb') as f:
            self.scaler = pickle.load(f)
        
        # Load metadata
        with open('models/training_metadata.json', 'r') as f:
            self.metadata = json.load(f)
        
        # Initialize components
        self.extractor = HandLandmarkExtractor()
        self.cap = cv2.VideoCapture(0)
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
        
        # Gesture control settings
        self.cooldown_period = 1.5
        self.last_gesture_time = 0
        self.current_gesture = "None"
        self.confidence_threshold = 0.8
        
        # Gesture smoothing
        self.gesture_history = deque(maxlen=5)
        self.prediction_history = deque(maxlen=10)
        
        # Performance monitoring
        self.fps_history = deque(maxlen=30)
        self.inference_time_history = deque(maxlen=30)
        
        # Gesture mapping
        self.gesture_commands = {
            'play': lambda: pyautogui.press('space'),
            'pause': lambda: pyautogui.press('space'),
            'volume_up': lambda: pyautogui.press('up'),
            'volume_down': lambda: pyautogui.press('down'),
            #'next_video': lambda: pyautogui.hotkey('shift', 'n'),
            #'previous_video': lambda: pyautogui.hotkey('shift', 'p'),
            #'fullscreen': lambda: pyautogui.press('f'),
            #'mute': lambda: pyautogui.press('m'),
            'speed_up': lambda: pyautogui.hotkey('shift', '.'),
            'speed_down': lambda: pyautogui.hotkey('shift', ',')
        }
        
        # UI settings
        self.show_debug = True
        self.show_landmarks = True
        
    def predict_gesture(self, landmarks):
        """Predict gesture using the trained model"""
        if landmarks is None:
            return None, 0.0
        
        # Extract features
        features = self.extractor.extract_advanced_features(landmarks)
        if features is None:
            return None, 0.0
        
        # Preprocess
        features_scaled = self.scaler.transform(features.reshape(1, -1))
        
        # Predict
        start_time = time.time()
        predictions = self.model.predict(features_scaled, verbose=0)[0]
        inference_time = (time.time() - start_time) * 1000  # Convert to ms
        self.inference_time_history.append(inference_time)
        
        # Get best prediction
        best_idx = np.argmax(predictions)
        confidence = predictions[best_idx]
        gesture = self.label_encoder.inverse_transform([best_idx])[0]
        
        return gesture, confidence
    
    def smooth_predictions(self, gesture, confidence):
        """Smooth predictions using history"""
        if confidence > self.confidence_threshold:
            self.gesture_history.append(gesture)
            
            # Check if gesture is consistent
            if len(self.gesture_history) >= 3:
                recent_gestures = list(self.gesture_history)[-3:]
                if all(g == gesture for g in recent_gestures):
                    return gesture, confidence
        
        return None, 0.0
    
    def execute_command(self, gesture):
        """Execute YouTube command based on gesture"""
        current_time = time.time()
        
        if current_time - self.last_gesture_time > self.cooldown_period:
            if gesture in self.gesture_commands:
                # Execute in separate thread to avoid blocking
                threading.Thread(
                    target=self.gesture_commands[gesture],
                    daemon=True
                ).start()
                
                self.last_gesture_time = current_time
                self.current_gesture = gesture
                print(f"Executed: {gesture}")
                return True
        
        return False
    
    def draw_ui(self, frame, gesture, confidence):
        """Draw advanced UI with ML metrics"""
        height, width = frame.shape[:2]
        
        # Create semi-transparent overlay
        overlay = frame.copy()
        
        # Top panel - Status
        cv2.rectangle(overlay, (0, 0), (width, 100), (0, 0, 0), -1)
        
        # Current gesture
        if gesture and confidence > self.confidence_threshold:
            cv2.putText(overlay, f"Gesture: {gesture.upper()}", (10, 30),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
            cv2.putText(overlay, f"Confidence: {confidence:.2%}", (10, 60),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
        else:
            cv2.putText(overlay, "No gesture detected", (10, 30),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
        
        # Cooldown indicator
        time_since_last = time.time() - self.last_gesture_time
        if time_since_last < self.cooldown_period:
            remaining = self.cooldown_period - time_since_last
            cv2.putText(overlay, f"Cooldown: {remaining:.1f}s", (10, 90),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
        else:
            cv2.putText(overlay, "Ready", (10, 90),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
        
        # Performance metrics
        if self.show_debug:
            # FPS
            if self.fps_history:
                avg_fps = np.mean(self.fps_history)
                cv2.putText(overlay, f"FPS: {avg_fps:.1f}", (width - 150, 30),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
            
            # Inference time
            if self.inference_time_history:
                avg_inference = np.mean(self.inference_time_history)
                cv2.putText(overlay, f"Inference: {avg_inference:.1f}ms", 
                           (width - 150, 60),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        
        # Gesture list
        y_offset = 150
        cv2.putText(overlay, "Gestures:", (10, y_offset),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        
        gesture_list = [
            "Open Palm: Play/Pause",
            "Fist: Play/Pause",
            "Index+Middle: Volume Up",
            "Ring+Pinky: Volume Down",
            "Pinky: Next Video",
            "All except Pinky: Previous",
            "Thumbs Up: Fullscreen",
            "Peace Sign: Mute",
            "Three Fingers: Speed Up",
            "Four Fingers: Speed Down"
        ]
        
        for i, gesture_text in enumerate(gesture_list[:6]):  # Show first 6
            y_offset += 25
            cv2.putText(overlay, gesture_text, (10, y_offset),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)
        
        # Apply overlay with transparency
        cv2.addWeighted(overlay, 0.7, frame, 0.3, 0, frame)
        
        return frame
    
    def run(self):
        """Main loop for gesture control"""
        print("Advanced YouTube Gesture Controller Started!")
        print("Make sure YouTube is open in your browser")
        print("Press 'Q' to quit, 'D' to toggle debug info, 'L' to toggle landmarks")
        
        prev_time = time.time()
        
        while True:
            ret, frame = self.cap.read()
            if not ret:
                break
            
            # Calculate FPS
            current_time = time.time()
            fps = 1 / (current_time - prev_time)
            prev_time = current_time
            self.fps_history.append(fps)
            
            # Flip frame for mirror effect
            frame = cv2.flip(frame, 1)
            
            # Extract landmarks
            landmarks, hand_landmarks = self.extractor.extract_landmarks(frame)
            
            gesture = None
            confidence = 0.0
            
            if hand_landmarks:
                # Draw landmarks if enabled
                if self.show_landmarks:
                    mp_drawing.draw_landmarks(
                        frame, hand_landmarks, mp_hands.HAND_CONNECTIONS,
                        mp_drawing.DrawingSpec(color=(0, 0, 255), thickness=2),
                        mp_drawing.DrawingSpec(color=(0, 255, 0), thickness=2)
                    )
                
                # Predict gesture
                gesture, confidence = self.predict_gesture(landmarks)
                
                # Smooth predictions
                smoothed_gesture, smoothed_confidence = self.smooth_predictions(
                    gesture, confidence
                )
                
                # Execute command if valid
                if smoothed_gesture:
                    self.execute_command(smoothed_gesture)
            
            # Draw UI
            frame = self.draw_ui(frame, gesture, confidence)
            
            # Display frame
            cv2.imshow('Advanced YouTube Gesture Controller', frame)
            
            # Handle keyboard input
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                break
            elif key == ord('d'):
                self.show_debug = not self.show_debug
            elif key == ord('l'):
                self.show_landmarks = not self.show_landmarks
        
        # Cleanup
        self.cap.release()
        cv2.destroyAllWindows()

if __name__ == "__main__":
    try:
        controller = AdvancedGestureController()
        controller.run()
    except FileNotFoundError as e:
        print(f"Error: {e}")
        print("Please make sure you have trained the model first!")
        print("Run: python train_model.py")