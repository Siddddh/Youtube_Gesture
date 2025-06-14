// dashboard.js
const chartOptions = {
    responsive: true,
    maintainAspectRatio: false,
    plugins: {
        legend: {
            position: 'bottom',
            labels: {
                color: '#fff'
            }
        }
    },
    scales: {
        x: {
            ticks: { color: '#ccc' },
            grid: { color: '#333' }
        },
        y: {
            beginAtZero: true,
            ticks: { color: '#ccc' },
            grid: { color: '#333' }
        }
    }
};

const gestureCtx = document.getElementById('gestureChart').getContext('2d');
const gestureChart = new Chart(gestureCtx, {
    type: 'doughnut',
    data: {
        labels: [],
        datasets: [{
            data: [],
            backgroundColor: [
                '#1DB954', '#36A2EB', '#FFCE56', '#4BC0C0',
                '#9966FF', '#FF9F40', '#FF6384', '#C9CBCF',
                '#4BC0C0', '#36A2EB'
            ]
        }]
    },
    options: chartOptions
});

const perfCtx = document.getElementById('performanceChart').getContext('2d');
const performanceChart = new Chart(perfCtx, {
    type: 'line',
    data: {
        labels: [],
        datasets: [
            {
                label: 'FPS',
                data: [],
                borderColor: '#1DB954',
                fill: false
            },
            {
                label: 'Inference Time (ms)',
                data: [],
                borderColor: '#FF6384',
                fill: false
            }
        ]
    },
    options: chartOptions
});

async function loadModelInfo() {
    try {
        const response = await fetch('/api/status');
        const data = await response.json();

        if (data.model_info) {
            const modelInfoEl = document.getElementById('model-info');
            modelInfoEl.innerHTML = `
                <div class="row">
                    <div class="col-md-4">
                        <h6>Model Accuracy</h6>
                        <p class="h3 text-success">${(data.model_info.accuracy * 100).toFixed(2)}%</p>
                    </div>
                    <div class="col-md-4">
                        <h6>Number of Classes</h6>
                        <p class="h3 text-info">${data.model_info.num_classes}</p>
                    </div>
                    <div class="col-md-4">
                        <h6>Supported Gestures</h6>
                        <ul class="list-unstyled">
                            ${data.model_info.classes.map(c => `<li>${c}</li>`).join('')}
                        </ul>
                    </div>
                </div>
            `;
        }
    } catch (error) {
        console.error('Error loading model info:', error);
    }
}

let performanceData = [];
const maxDataPoints = 20;

function updateCharts(data) {
    const timestamp = new Date().toLocaleTimeString();
    performanceData.push({
        time: timestamp,
        fps: data.fps || 0,
        inference: data.inference_time || 0
    });

    if (performanceData.length > maxDataPoints) {
        performanceData.shift();
    }

    performanceChart.data.labels = performanceData.map(d => d.time);
    performanceChart.data.datasets[0].data = performanceData.map(d => d.fps);
    performanceChart.data.datasets[1].data = performanceData.map(d => d.inference);
    performanceChart.update();
}

const socket = io();

socket.on('gesture_update', (data) => {
    updateCharts(data);

    if (data.gesture) {
        const labels = gestureChart.data.labels;
        const index = labels.indexOf(data.gesture);

        if (index !== -1) {
            gestureChart.data.datasets[0].data[index]++;
        } else {
            labels.push(data.gesture);
            gestureChart.data.datasets[0].data.push(1);
        }

        gestureChart.update();
    }
});

window.addEventListener('load', () => {
    loadModelInfo();
});
