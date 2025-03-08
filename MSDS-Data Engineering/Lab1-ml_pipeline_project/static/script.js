const predictBtn = document.getElementById('predict-btn');
const randomBtn = document.getElementById('random-btn');
const resultDiv = document.getElementById('prediction-result');
const responseTimeChart = document.getElementById('response-time-chart').getContext('2d');

let iterationData = [];
let chartInstance = null;  // Store chart instance here

// Handle Predict Button Click
predictBtn.addEventListener('click', async () => {
    const sepalLength = parseFloat(document.getElementById('sepal-length').value);
    const sepalWidth = parseFloat(document.getElementById('sepal-width').value);
    const petalLength = parseFloat(document.getElementById('petal-length').value);
    const petalWidth = parseFloat(document.getElementById('petal-width').value);

    const features = [sepalLength, sepalWidth, petalLength, petalWidth];

    try {
        const response = await fetch('http://127.0.0.1:5000/predict', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ features })
        });
        const data = await response.json();

        if (data.error) {
            resultDiv.innerHTML = `<p>Error: ${data.error}</p>`;
        } else {
            resultDiv.innerHTML = `
                <p>Prediction: ${data.prediction}</p>
                <p>Flower Type: ${data.flower_type}</p>
                <p>Model Type: ${data.model_type}</p>
                <p>Confidence: ${data.confidence_level}%</p>
                <p>Response Time: ${data.processing_time} seconds</p>
            `;

            // Store the processing time for plotting
            iterationData.push(data.processing_time);
            updateChart();  // Update the chart with the new data
        }
    } catch (error) {
        resultDiv.innerHTML = `<p>Error: ${error.message}</p>`;
    }
});

// Handle Random Prediction Button Click
randomBtn.addEventListener('click', async () => {
    try {
        const response = await fetch('http://127.0.0.1:5000/random_prediction');
        const data = await response.json();

        if (data.error) {
            resultDiv.innerHTML = `<p>Error: ${data.error}</p>`;
        } else {
            resultDiv.innerHTML = `
                <p>Prediction: ${data.prediction}</p>
                <p>Flower Type: ${data.flower_type}</p>
                <p>Model Type: ${data.model_type}</p>
                <p>Confidence: ${data.confidence_level}%</p>
            `;
        }
    } catch (error) {
        resultDiv.innerHTML = `<p>Error: ${error.message}</p>`;
    }
});

// Update the response time chart
function updateChart() {
    // If chart instance exists, update it
    if (chartInstance) {
        chartInstance.data.labels = Array.from({ length: iterationData.length }, (_, i) => i + 1);  // x-axis labels (iterations)
        chartInstance.data.datasets[0].data = iterationData;  // Update the y-axis data (response times)
        chartInstance.update();  // Just update the chart, don't recreate it
    } else {
        // Initialize a new chart instance if not already created
        chartInstance = new Chart(responseTimeChart, {
            type: 'line',
            data: {
                labels: Array.from({ length: iterationData.length }, (_, i) => i + 1),  // x-axis labels (iterations)
                datasets: [{
                    label: 'Response Time (seconds)',
                    data: iterationData,  // Data points (response times)
                    borderColor: 'rgba(75, 192, 192, 1)',
                    fill: false,
                }],
            },
            options: {
                scales: {
                    y: {
                        beginAtZero: true,
                        title: {
                            display: true,
                            text: 'Response Time (seconds)'
                        }
                    },
                    x: {
                        title: {
                            display: true,
                            text: 'Iteration'
                        }
                    }
                }
            }
        });
    }
}
