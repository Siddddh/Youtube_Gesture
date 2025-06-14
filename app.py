from flask import Flask, render_template, Response, jsonify, request
from flask_socketio import SocketIO, emit
from flask_cors import CORS
import cv2
import numpy as np
import base64
import json
import threading
import time
from datetime import datetime
import os
from utils import HandLandmarkExtractor, mp_drawing, mp_hands, create_directories
from gesture_controller import AdvancedGestureController
import tensorflow as tf

# Initialize Flask app
app = Flask(__name__)
app.config['SECRET_KEY'] = 'your-secret-key-here'
CORS(app)
socketio = SocketIO(app, cors_allowed_origins="*")

# Global variables
controller = None
is_running = False
camera_thread = None

class WebGestureController(AdvancedGestureController):
    """Modified gesture controller for web deployment"""
    
    def __init__(self, model_path='models/gesture_model.h5'):
        super().__init__(model_path)
        self.frame_queue = []
        self.result_queue = []
        self.is_processing = False
        
    def process_frame_for_web(self, frame):
        """Process frame and return results for web display"""
        # Flip frame for mirror effect
        frame = cv2.flip(frame, 1)
        
        # Extract landmarks
        landmarks, hand_landmarks = self.extractor.extract_landmarks(frame)
        
        result = {
            'gesture': None,
            'confidence': 0.0,
            'landmarks': None,
            'fps': 0,
            'inference_time': 0
        }
        
        if hand_landmarks:
            # Draw landmarks
            if self.show_landmarks:
                mp_drawing.draw_landmarks(
                    frame, hand_landmarks, mp_hands.HAND_CONNECTIONS,
                    mp_drawing.DrawingSpec(color=(0, 0, 255), thickness=2),
                    mp_drawing.DrawingSpec(color=(0, 255, 0), thickness=2)
                )
            
            # Predict gesture
            gesture, confidence = self.predict_gesture(landmarks)
            
            if gesture and confidence > self.confidence_threshold:
                result['gesture'] = gesture
                result['confidence'] = float(confidence)
                
                # Smooth predictions
                smoothed_gesture, _ = self.smooth_predictions(gesture, confidence)
                
                # Execute command if valid
                if smoothed_gesture:
                    self.execute_command(smoothed_gesture)
        
        # Calculate metrics
        if self.fps_history:
            result['fps'] = float(np.mean(self.fps_history))
        if self.inference_time_history:
            result['inference_time'] = float(np.mean(self.inference_time_history))
        
        # Draw UI
        frame = self.draw_ui(frame, result['gesture'], result['confidence'])
        
        return frame, result

# Initialize controller
try:
    controller = WebGestureController()
    print("Gesture controller initialized successfully!")
except Exception as e:
    print(f"Error initializing controller: {e}")

@app.route('/')
def index():
    """Main page"""
    return render_template('index.html')

@app.route('/dashboard')
def dashboard():
    """Dashboard page with analytics"""
    return render_template('dashboard.html')

@app.route('/api/status')
def api_status():
    """Get system status"""
    status = {
        'model_loaded': controller is not None,
        'camera_active': is_running,
        'timestamp': datetime.now().isoformat()
    }
    
    if controller:
        status['model_info'] = {
            'classes': controller.metadata['classes'],
            'accuracy': controller.metadata.get('final_accuracy', 0),
            'num_classes': controller.metadata['num_classes']
        }
    
    return jsonify(status)

@app.route('/api/gestures')
def api_gestures():
    """Get available gestures"""
    if controller:
        return jsonify({
            'gestures': controller.metadata['classes'],
            'commands': list(controller.gesture_commands.keys())
        })
    return jsonify({'error': 'Controller not initialized'})

def generate_frames():
    """Generate frames for video streaming"""
    global controller, is_running
    
    while is_running:
        ret, frame = controller.cap.read()
        if not ret:
            break
        
        # Process frame
        frame, result = controller.process_frame_for_web(frame)
        
        # Send result via WebSocket
        socketio.emit('gesture_update', result)
        
        # Encode frame
        ret, buffer = cv2.imencode('.jpg', frame)
        frame = buffer.tobytes()
        
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')

@app.route('/video_feed')
def video_feed():
    """Video streaming route"""
    return Response(generate_frames(),
                    mimetype='multipart/x-mixed-replace; boundary=frame')

@socketio.on('connect')
def handle_connect():
    """Handle client connection"""
    print('Client connected')
    emit('connected', {'data': 'Connected to server'})

@socketio.on('disconnect')
def handle_disconnect():
    """Handle client disconnection"""
    print('Client disconnected')

@socketio.on('start_camera')
def handle_start_camera():
    """Start camera streaming"""
    global is_running
    is_running = True
    emit('camera_status', {'active': True})

@socketio.on('stop_camera')
def handle_stop_camera():
    """Stop camera streaming"""
    global is_running
    is_running = False
    emit('camera_status', {'active': False})

@socketio.on('toggle_landmarks')
def handle_toggle_landmarks():
    """Toggle landmark display"""
    if controller:
        controller.show_landmarks = not controller.show_landmarks
        emit('landmarks_status', {'show': controller.show_landmarks})

@socketio.on('toggle_debug')
def handle_toggle_debug():
    """Toggle debug info"""
    if controller:
        controller.show_debug = not controller.show_debug
        emit('debug_status', {'show': controller.show_debug})

if __name__ == '__main__':
    create_directories()
    socketio.run(app, debug=True, host='0.0.0.0', port=5000)