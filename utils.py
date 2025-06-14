# Utility functions for gesture recognition
import numpy as np
import cv2
import mediapipe as mp
from sklearn.preprocessing import StandardScaler
import pickle
import os

mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils

class HandLandmarkExtractor:
    def __init__(self):
        self.hands = mp_hands.Hands(
            static_image_mode=False,
            max_num_hands=1,
            min_detection_confidence=0.7,
            min_tracking_confidence=0.5
        )
        self.scaler = StandardScaler()
        
    def extract_landmarks(self, image):
        """Extract hand landmarks from image"""
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        results = self.hands.process(image_rgb)
        
        if results.multi_hand_landmarks:
            landmarks = results.multi_hand_landmarks[0]
            # Extract 21 landmarks with x, y, z coordinates (63 features)
            landmark_array = []
            for landmark in landmarks.landmark:
                landmark_array.extend([landmark.x, landmark.y, landmark.z])
            return np.array(landmark_array), landmarks
        return None, None
    
    def extract_advanced_features(self, landmarks_array):
        """Extract advanced features from landmarks"""
        if landmarks_array is None:
            return None
        
        landmarks_3d = landmarks_array.reshape(21, 3)
        
        # Additional features
        features = []
        
        # 1. Original landmarks (63 features)
        features.extend(landmarks_array)
        
        # 2. Distances between fingertips and palm center (5 features)
        palm_center = np.mean(landmarks_3d[[0, 1, 5, 9, 13, 17]], axis=0)
        fingertips = landmarks_3d[[4, 8, 12, 16, 20]]
        distances = np.linalg.norm(fingertips - palm_center, axis=1)
        features.extend(distances)
        
        # 3. Angles between fingers (10 features)
        angles = []
        finger_indices = [4, 8, 12, 16, 20]  # Fingertips
        for i in range(len(finger_indices)):
            for j in range(i+1, len(finger_indices)):
                v1 = landmarks_3d[finger_indices[i]] - landmarks_3d[0]
                v2 = landmarks_3d[finger_indices[j]] - landmarks_3d[0]
                angle = np.arccos(np.clip(np.dot(v1, v2) / 
                        (np.linalg.norm(v1) * np.linalg.norm(v2)), -1, 1))
                angles.append(angle)
        features.extend(angles)
        
        # 4. Hand orientation (3 features)
        hand_plane_normal = np.cross(
            landmarks_3d[5] - landmarks_3d[0],
            landmarks_3d[17] - landmarks_3d[0]
        )
        hand_plane_normal = hand_plane_normal / np.linalg.norm(hand_plane_normal)
        features.extend(hand_plane_normal)
        
        # 5. Finger curvature (5 features)
        finger_roots = [1, 5, 9, 13, 17]
        finger_tips = [4, 8, 12, 16, 20]
        for root, tip in zip(finger_roots, finger_tips):
            curvature = np.linalg.norm(landmarks_3d[tip] - landmarks_3d[root])
            features.append(curvature)
        
        return np.array(features)
    
    def save_scaler(self, path='models/scaler.pkl'):
        """Save the scaler for later use"""
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, 'wb') as f:
            pickle.dump(self.scaler, f)
    
    def load_scaler(self, path='models/scaler.pkl'):
        """Load the scaler"""
        with open(path, 'rb') as f:
            self.scaler = pickle.load(f)

def create_directories():
    """Create necessary directories"""
    dirs = ['data/raw', 'data/processed', 'models', 'logs']
    for dir in dirs:
        os.makedirs(dir, exist_ok=True)