// dashboard.js - JavaScript for the Dashboard page

document.addEventListener('DOMContentLoaded', function() {
    // Initialize dashboard
    initDashboard();
    
    // Set up refresh button event
    document.getElementById('refresh-btn').addEventListener('click', refreshData);
    
    // Set up timeframe selector event
    document.getElementById('history-timeframe').addEventListener('change', function() {
        loadHistoryData(this.value);
    });
});

/**
 * Initialize dashboard with data
 */
function initDashboard() {
    // Fetch current vitals data
    fetchVitalsData();
    
    // Load history data with default timeframe (day)
    loadHistoryData('day');
    
    // Update last updated time
    updateLastUpdated();
}

/**
 * Fetch and display current vitals data
 */
function fetchVitalsData() {
    fetch('/api/vitals/current')
        .then(response => {
            if (!response.ok) {
                throw new Error('Network response was not ok');
            }
            return response.json();
        })
        .then(data => {
            updateVitalsDisplay(data);
            initCharts(data);
        })
        .catch(error => {
            console.error('Error fetching vitals data:', error);
            showErrorMessage('Unable to load current vitals data. Please try again.');
        });
}

/**
 * Update the vitals display with the given data
 * @param {Object} data - The vitals data from the API
 */
function updateVitalsDisplay(data) {
    // Update heart rate
    document.getElementById('heart-rate').textContent = data.heart_rate ? `${data.heart_rate} bpm` : '-- bpm';
    
    // Update SpO2
    document.getElementById('spo2').textContent = data.spo2 ? `${data.spo2}%` : '-- %';
    
    // Update temperature
    document.getElementById('temperature').textContent = data.temperature ? `${data.temperature.toFixed(1)} °C` : '-- °C';
    
    // Update respiratory rate
    document.getElementById('respiratory-rate').textContent = data.respiratory_rate ? `${data.respiratory_rate} breaths/min` : '-- breaths/min';
    
    // Update blood pressure
    document.getElementById('blood-pressure').textContent = (data.systolic_bp && data.diastolic_bp) ? 
        `${data.systolic_bp}/${data.diastolic_bp} mmHg` : '--/-- mmHg';
    
    // Update BMI
    document.getElementById('bmi').textContent = data.bmi ? data.bmi.toFixed(1) : '--';
    
    // Update BMI classification
    if (data.bmi) {
        let bmiClass = '';
        if (data.bmi < 18.5) bmiClass = 'Underweight';
        else if (data.bmi < 25) bmiClass = 'Normal weight';
        else if (data.bmi < 30) bmiClass = 'Overweight';
        else bmiClass = 'Obese';
        
        document.getElementById('bmi-classification').textContent = bmiClass;
    } else {
        document.getElementById('bmi-classification').textContent = 'Unknown';
    }
    
    // Update risk level
    const riskElement = document.getElementById('risk-level');
    riskElement.classList.remove('risk-low', 'risk-moderate', 'risk-high');
    
    if (data.risk_level) {
        riskElement.querySelector('.value').textContent = data.risk_level;
        
        if (data.risk_level === 'Low') {
            riskElement.classList.add('risk-low');
        } else if (data.risk_level === 'Moderate') {
            riskElement.classList.add('risk-moderate');
        } else if (data.risk_level === 'High') {
            riskElement.classList.add('risk-high');
        }
    } else {
        riskElement.querySelector('.value').textContent = 'Unknown';
    }
    
    // Update facial expression
    const expressionElement = document.getElementById('facial-expression');
    expressionElement.classList.remove('expression-positive', 'expression-neutral', 'expression-negative');
    
    if (data.facial_expression) {
        expressionElement.querySelector('.value').textContent = data.facial_expression;
        
        if (data.facial_expression === 'Happy' || data.facial_expression === 'Calm') {
            expressionElement.classList.add('expression-positive');
        } else if (data.facial_expression === 'Neutral') {
            expressionElement.classList.add('expression-neutral');
        } else {
            expressionElement.classList.add('expression-negative');
        }
    } else {
        expressionElement.querySelector('.value').textContent = 'Unknown';
    }
    
    // Update combined assessment
    const combinedElement = document.getElementById('combined-assessment');
    combinedElement.classList.remove('risk-low', 'risk-moderate', 'risk-high');
    
    if (data.combined_assessment) {
        combinedElement.querySelector('.value').textContent = data.combined_assessment;
        
        if (data.combined_assessment === 'Good') {
            combinedElement.classList.add('risk-low');
        } else if (data.combined_assessment === 'Fair') {
            combinedElement.classList.add('risk-moderate');
        } else if (data.combined_assessment === 'Poor') {
            combinedElement.classList.add('risk-high');
        }
    } else {
        combinedElement.querySelector('.value').textContent = 'Unknown';
    }
}

/**
 * Initialize charts with the latest vitals data
 * @param {Object} data - The vitals data from the API
 */
function initCharts(data) {
    // Heart rate chart
    const heartRateCtx = document.getElementById('heart-rate-chart').getContext('2d');
    new Chart(heartRateCtx, {
        type: 'line',
        data: {
            labels: generateTimeLabels(12),
            datasets: [{
                label: 'Heart Rate (bpm)',
                data: generateRandomData(data.heart_rate, 12, 5),
                borderColor: '#e74c3c',
                backgroundColor: 'rgba(231, 76, 60, 0.1)',
                tension: 0.4,
                fill: true
            }]
        },
        options: getChartOptions('Heart Rate (bpm)')
    });
    
    // SpO2 chart
    const spo2Ctx = document.getElementById('spo2-chart').getContext('2d');
    new Chart(spo2Ctx, {
        type: 'line',
        data: {
            labels: generateTimeLabels(12),
            datasets: [{
                label: 'SpO2 (%)',
                data: generateRandomData(data.spo2, 12, 1, true, 95, 100),
                borderColor: '#3498db',
                backgroundColor: 'rgba(52, 152, 219, 0.1)',
                tension: 0.4,
                fill: true
            }]
        },
        options: getChartOptions('SpO2 (%)')
    });
    
    // Temperature chart
    const tempCtx = document.getElementById('temperature-chart').getContext('2d');
    new Chart(tempCtx, {
        type: 'line',
        data: {
            labels: generateTimeLabels(12),
            datasets: [{
                label: 'Temperature (°C)',
                data: generateRandomData(data.temperature, 12, 0.2),
                borderColor: '#f39c12',
                backgroundColor: 'rgba(243, 156, 18, 0.1)',
                tension: 0.4,
                fill: true
            }]
        },
        options: getChartOptions('Temperature (°C)')
    });
    
    // Respiratory rate chart
    const respCtx = document.getElementById('respiratory-chart').getContext('2d');
    new Chart(respCtx, {
        type: 'line',
        data: {
            labels: generateTimeLabels(12),
            datasets: [{
                label: 'Respiratory Rate (breaths/min)',
                data: generateRandomData(data.respiratory_rate, 12, 1),
                borderColor: '#2ecc71',
                backgroundColor: 'rgba(46, 204, 113, 0.1)',
                tension: 0.4,
                fill: true
            }]
        },
        options: getChartOptions('Respiratory Rate (breaths/min)')
    });
    
    // Blood pressure chart
    const bpCtx = document.getElementById('bp-chart').getContext('2d');
    new Chart(bpCtx, {
        type: 'line',
        data: {
            labels: generateTimeLabels(12),
            datasets: [
                {
                    label: 'Systolic (mmHg)',
                    data: generateRandomData(data.systolic_bp, 12, 3),
                    borderColor: '#9b59b6',
                    backgroundColor: 'rgba(155, 89, 182, 0.1)',
                    tension: 0.4,
                    fill: false
                },
                {
                    label: 'Diastolic (mmHg)',
                    data: generateRandomData(data.diastolic_bp, 12, 2),
                    borderColor: '#16a085',
                    backgroundColor: 'rgba(22, 160, 133, 0.1)',
                    tension: 0.4,
                    fill: false
                }
            ]
        },
        options: getChartOptions('Blood Pressure (mmHg)')
    });
}

/**
 * Generate chart options
 * @param {string} title - The chart title
 * @return {Object} Chart.js options object
 */
function getChartOptions(title) {
    return {
        plugins: {
            legend: {
                display: false
            },
            tooltip: {
                mode: 'index',
                intersect: false
            }
        },
        scales: {
            x: {
                display: false
            },
            y: {
                beginAtZero: false
            }
        },
        responsive: true,
        maintainAspectRatio: false
    };
}

/**
 * Generate time labels for charts
 * @param {number} count - Number of time labels to generate
 * @return {Array} Array of time labels
 */
function generateTimeLabels(count) {
    const labels = [];
    const now = new Date();
    
    for (let i = count - 1; i >= 0; i--) {
        const time = new Date(now - i * 5 * 60000); // 5 minutes intervals
        labels.push(time.toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' }));
    }
    
    return labels;
}

/**
 * Generate random data points for charts
 * @param {number} baseValue - Base value to fluctuate around
 * @param {number} count - Number of data points to generate
 * @param {number} variance - Maximum variance from base value
 * @param {boolean} constrained - Whether to constrain values within min and max
 * @param {number} min - Minimum value if constrained
 * @param {number} max - Maximum value if constrained
 * @return {Array} Array of data points
 */
function generateRandomData(baseValue, count, variance, constrained = false, min = 0, max = 200) {
    if (!baseValue) return Array(count).fill(null);
    
    const data = [];
    
    for (let i = 0; i < count; i++) {
        let value = baseValue + (Math.random() * 2 - 1) * variance;
        
        if (constrained) {
            value = Math.min(Math.max(value, min), max);
        }
        
        data.push(Number(value.toFixed(1)));
    }
    
    return data;
}

/**
 * Load historical health data
 * @param {string} timeframe - The timeframe to load (day, week, month)
 */
function loadHistoryData(timeframe) {
    fetch(`/api/history/${timeframe}`)
        .then(response => {
            if (!response.ok) {
                throw new Error('Network response was not ok');
            }
            return response.json();
        })
        .then(data => {
            updateHistoryChart(data, timeframe);
            updateHistoryTable(data);
        })
        .catch(error => {
            console.error('Error fetching history data:', error);
            showErrorMessage('Unable to load history data. Please try again.');
        });
}

/**
 * Update the history chart with the given data
 * @param {Array} data - The history data from the API
 * @param {string} timeframe - The timeframe (day, week, month)
 */
function updateHistoryChart(data, timeframe) {
    // Get timestamps and format based on timeframe
    const timestamps = data.map(entry => {
        const date = new Date(entry.timestamp);
        if (timeframe === 'day') {
            return date.toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' });
        } else if (timeframe === 'week') {
            return date.toLocaleDateString([], { weekday: 'short', month: 'numeric', day: 'numeric' });
        } else {
            return date.toLocaleDateString([], { month: 'short', day: 'numeric' });
        }
    });
    
    // Get heart rate and SpO2 data
    const heartRateData = data.map(entry => entry.heart_rate);
    const spo2Data = data.map(entry => entry.spo2);
    
    // Get or create chart
    const ctx = document.getElementById('history-chart').getContext('2d');
    
    if (window.historyChart) {
        window.historyChart.destroy();
    }
    
    window.historyChart = new Chart(ctx, {
        type: 'line',
        data: {
            labels: timestamps,
            datasets: [
                {
                    label: 'Heart Rate (bpm)',
                    data: heartRateData,
                    borderColor: '#e74c3c',
                    backgroundColor: 'rgba(231, 76, 60, 0.1)',
                    tension: 0.4,
                    yAxisID: 'y'
                },
                {
                    label: 'SpO2 (%)',
                    data: spo2Data,
                    borderColor: '#3498db',
                    backgroundColor: 'rgba(52, 152, 219, 0.1)',
                    tension: 0.4,
                    yAxisID: 'y1'
                }
            ]
        },
        options: {
            responsive: true,
            maintainAspectRatio: false,
            plugins: {
                legend: {
                    position: 'top'
                },
                tooltip: {
                    mode: 'index',
                    intersect: false
                }
            },
            scales: {
                x: {
                    title: {
                        display: true,
                        text: timeframe === 'day' ? 'Time' : 'Date'
                    }
                },
                y: {
                    type: 'linear',
                    display: true,
                    position: 'left',
                    title: {
                        display: true,
                        text: 'Heart Rate (bpm)'
                    }
                },
                y1: {
                    type: 'linear',
                    display: true,
                    position: 'right',
                    title: {
                        display: true,
                        text: 'SpO2 (%)'
                    },
                    min: 90,
                    max: 100,
                    grid: {
                        drawOnChartArea: false
                    }
                }
            }
        }
    });
}

/**
 * Update the history table with the given data
 * @param {Array} data - The history data from the API
 */
function updateHistoryTable(data) {
    const tableBody = document.querySelector('#history-table tbody');
    tableBody.innerHTML = '';
    
    // Only show the most recent 10 entries
    const recentData = data.slice(0, 10);
    
    recentData.forEach(entry => {
        const date = new Date(entry.timestamp);
        const formattedTime = date.toLocaleString();
        
        // Determine risk class for styling
        let riskClass = '';
        if (entry.risk_level === 'Low') {
            riskClass = 'risk-low';
        } else if (entry.risk_level === 'Moderate') {
            riskClass = 'risk-moderate';
        } else if (entry.risk_level === 'High') {
            riskClass = 'risk-high';
        }
        
        const row = `
            <tr>
                <td>${formattedTime}</td>
                <td>${entry.heart_rate || '--'} bpm</td>
                <td>${entry.spo2 || '--'}%</td>
                <td>${entry.temperature ? entry.temperature.toFixed(1) : '--'} °C</td>
                <td>${entry.systolic_bp || '--'}/${entry.diastolic_bp || '--'}</td>
                <td class="${riskClass}">${entry.risk_level || 'Unknown'}</td>
            </tr>
        `;
        
        tableBody.innerHTML += row;
    });
    
    // If no data available
    if (recentData.length === 0) {
        tableBody.innerHTML = `
            <tr>
                <td colspan="6" class="no-data">No historical data available for this timeframe</td>
            </tr>
        `;
    }
}

/**
 * Show error message
 * @param {string} message - The error message to display
 */
function showErrorMessage(message) {
    // You could implement a toast notification or other error display here
    console.error(message);
    alert(message);
}

/**
 * Update the "last updated" timestamp
 */
function updateLastUpdated() {
    const lastUpdatedElement = document.getElementById('last-updated');
    lastUpdatedElement.textContent = `Last updated: ${new Date().toLocaleTimeString()}`;
}

/**
 * Refresh all dashboard data
 */
function refreshData() {
    // Show loading state
    const refreshBtn = document.getElementById('refresh-btn');
    const originalContent = refreshBtn.innerHTML;
    refreshBtn.innerHTML = '<i class="fas fa-spinner fa-spin"></i> Refreshing...';
    refreshBtn.disabled = true;
    
    // Fetch new data
    Promise.all([
        fetchVitalsData(),
        loadHistoryData(document.getElementById('history-timeframe').value)
    ])
        .then(() => {
            // Update last updated time
            updateLastUpdated();
        })
        .catch(error => {
            console.error('Error refreshing data:', error);
            showErrorMessage('Failed to refresh data. Please try again.');
        })
        .finally(() => {
            // Reset button state
            refreshBtn.innerHTML = originalContent;
            refreshBtn.disabled = false;
        });
}