import cv2
import numpy as np
import os
import time
from datetime import datetime
from utils import HandLandmarkExtractor, mp_drawing, mp_hands, create_directories
import json

class GestureDataCollector:
    def __init__(self):
        self.cap = cv2.VideoCapture(0)
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
        self.extractor = HandLandmarkExtractor()
        self.gestures = {
            '0': 'play',
            '1': 'pause',
            '2': 'volume_up',
            '3': 'volume_down',
            '4': 'next_video',
            '5': 'previous_video',
            '6': 'fullscreen',
            '7': 'mute',
            '8': 'speed_up',
            '9': 'speed_down'
        }
        self.collected_data = {gesture: [] for gesture in self.gestures.values()}
        self.is_collecting = False
        self.current_gesture = None
        create_directories()
        
    def collect_data(self):
        """Main data collection loop"""
        print("=== Gesture Data Collection ===")
        print("\nGestures to collect:")
        for key, gesture in self.gestures.items():
            print(f"Press '{key}' to collect: {gesture}")
        print("\nPress 'SPACE' to start/stop collection")
        print("Press 'S' to save collected data")
        print("Press 'Q' to quit\n")
        
        samples_per_gesture = 100
        collection_delay = 0.1
        last_collection_time = 0
        
        while True:
            ret, frame = self.cap.read()
            if not ret:
                break
            
            frame = cv2.flip(frame, 1)
            
            # Extract landmarks and features
            landmarks, hand_landmarks = self.extractor.extract_landmarks(frame)
            
            if hand_landmarks:
                mp_drawing.draw_landmarks(
                    frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)
                
                # Collect data if enabled
                if self.is_collecting and self.current_gesture and \
                   time.time() - last_collection_time > collection_delay:
                    
                    features = self.extractor.extract_advanced_features(landmarks)
                    if features is not None:
                        self.collected_data[self.current_gesture].append(features)
                        last_collection_time = time.time()
            
            # Display UI
            self._draw_ui(frame)
            
            cv2.imshow('Gesture Data Collection', frame)
            
            # Handle keyboard input
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                break
            elif key == ord(' '):
                self.is_collecting = not self.is_collecting
                if not self.is_collecting:
                    self.current_gesture = None
            elif key == ord('s'):
                self.save_data()
            elif chr(key) in self.gestures:
                self.current_gesture = self.gestures[chr(key)]
                print(f"Selected gesture: {self.current_gesture}")
        
        self.cap.release()
        cv2.destroyAllWindows()
    
    def _draw_ui(self, frame):
        """Draw UI elements on frame"""
        # Status
        status = "COLLECTING" if self.is_collecting else "PAUSED"
        color = (0, 255, 0) if self.is_collecting else (0, 0, 255)
        cv2.putText(frame, f"Status: {status}", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
        
        # Current gesture
        if self.current_gesture:
            cv2.putText(frame, f"Gesture: {self.current_gesture}", (10, 60),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)
            
            # Sample count
            count = len(self.collected_data[self.current_gesture])
            cv2.putText(frame, f"Samples: {count}", (10, 90),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        
        # Data summary
        y_offset = 150
        cv2.putText(frame, "Collected Samples:", (10, y_offset),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
        
        for gesture, data in self.collected_data.items():
            y_offset += 25
            cv2.putText(frame, f"{gesture}: {len(data)}", (10, y_offset),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)
    
    def save_data(self):
        """Save collected data"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # Save raw data
        data_dict = {
            'gestures': [],
            'features': [],
            'metadata': {
                'timestamp': timestamp,
                'gesture_names': list(self.gestures.values()),
                'total_samples': sum(len(data) for data in self.collected_data.values())
            }
        }
        
        for gesture, features_list in self.collected_data.items():
            for features in features_list:
                data_dict['gestures'].append(gesture)
                data_dict['features'].append(features.tolist())
        
        # Save as numpy arrays
        if data_dict['features']:
            np.save(f'data/raw/features_{timestamp}.npy', 
                    np.array(data_dict['features']))
            np.save(f'data/raw/labels_{timestamp}.npy', 
                    np.array(data_dict['gestures']))
            
            # Save metadata
            with open(f'data/raw/metadata_{timestamp}.json', 'w') as f:
                json.dump(data_dict['metadata'], f, indent=4)
            
            print(f"\nData saved successfully!")
            print(f"Total samples: {data_dict['metadata']['total_samples']}")
            print(f"Files saved with timestamp: {timestamp}")
        else:
            print("No data to save!")

if __name__ == "__main__":
    collector = GestureDataCollector()
    collector.collect_data()