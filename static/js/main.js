const socket = io();

const videoFeed = document.getElementById('video-feed');
const cameraPlaceholder = document.getElementById('camera-placeholder');
const toggleCameraBtn = document.getElementById('toggle-camera');
const toggleLandmarksBtn = document.getElementById('toggle-landmarks');
const toggleDebugBtn = document.getElementById('toggle-debug');
const currentGestureEl = document.getElementById('current-gesture');
const confidenceBar = document.getElementById('confidence-bar');
const confidenceValue = document.getElementById('confidence-value');
const fpsValue = document.getElementById('fps-value');
const inferenceValue = document.getElementById('inference-value');

let isCameraActive = false;
let gestureHistory = [];
const maxHistoryLength = 50;

socket.on('connect', () => {
    console.log('Connected to server');
});

socket.on('gesture_update', (data) => {
    updateGestureDisplay(data);
    updateMetrics(data);
    addToHistory(data);
});

socket.on('camera_status', (data) => {
    isCameraActive = data.active;
    updateCameraUI();
});

toggleCameraBtn.addEventListener('click', () => {
    isCameraActive ? socket.emit('stop_camera') : socket.emit('start_camera');
});

toggleLandmarksBtn.addEventListener('click', () => {
    socket.emit('toggle_landmarks');
});

toggleDebugBtn.addEventListener('click', () => {
    socket.emit('toggle_debug');
});

function updateGestureDisplay(data) {
    if (data.gesture && data.confidence > 0.8) {
        currentGestureEl.textContent = data.gesture;
        currentGestureEl.classList.add('gesture-detected');
        setTimeout(() => {
            currentGestureEl.classList.remove('gesture-detected');
        }, 500);

        const confidencePercent = Math.round(data.confidence * 100);
        confidenceBar.style.width = `${confidencePercent}%`;
        confidenceValue.textContent = `${confidencePercent}%`;

        if (confidencePercent >= 90) {
            confidenceBar.className = 'progress-bar bg-success';
        } else if (confidencePercent >= 70) {
            confidenceBar.className = 'progress-bar bg-warning';
        } else {
            confidenceBar.className = 'progress-bar bg-danger';
        }
    } else {
        currentGestureEl.textContent = 'None';
        confidenceBar.style.width = '0%';
        confidenceValue.textContent = '0%';
    }
}

function updateMetrics(data) {
    if (data.fps) fpsValue.textContent = data.fps.toFixed(1);
    if (data.inference_time) inferenceValue.textContent = `${data.inference_time.toFixed(1)}ms`;
}

function updateCameraUI() {
    if (isCameraActive) {
        videoFeed.style.display = 'block';
        cameraPlaceholder.style.display = 'none';
        toggleCameraBtn.innerHTML = '<i class="fas fa-video-slash"></i>';
        toggleCameraBtn.classList.add('text-danger');
    } else {
        videoFeed.style.display = 'none';
        cameraPlaceholder.style.display = 'flex';
        toggleCameraBtn.innerHTML = '<i class="fas fa-video"></i>';
        toggleCameraBtn.classList.remove('text-danger');
    }
}

function addToHistory(data) {
    if (data.gesture) {
        gestureHistory.push({
            gesture: data.gesture,
            confidence: data.confidence,
            timestamp: new Date()
        });

        if (gestureHistory.length > maxHistoryLength) {
            gestureHistory.shift();
        }
    }
}

document.addEventListener('keydown', (e) => {
    switch(e.key) {
        case 'c': toggleCameraBtn.click(); break;
        case 'l': toggleLandmarksBtn.click(); break;
        case 'd': toggleDebugBtn.click(); break;
    }
});

window.addEventListener('load', () => {
    fetch('/api/status')
        .then(response => response.json())
        .then(data => console.log('System status:', data));
});
