// main.js - JavaScript for the Home page

document.addEventListener('DOMContentLoaded', function() {
    // Fetch latest health data for the home page
    fetchLatestHealthData();
    
    // Set copyright year
    document.querySelector('footer .container p').innerHTML = `&copy; ${new Date().getFullYear()} Health Monitoring System. All rights reserved.`;
});

/**
 * Fetches the latest health data from the API and displays it on the home page
 */
function fetchLatestHealthData() {
    const latestDataContainer = document.getElementById('latest-data');
    
    // Show loading state
    latestDataContainer.innerHTML = '<p><i class="fas fa-spinner fa-spin"></i> Loading health data...</p>';
    
    fetch('/api/latest')
        .then(response => {
            if (!response.ok) {
                throw new Error('Network response was not ok');
            }
            return response.json();
        })
        .then(data => {
            displayLatestData(data, latestDataContainer);
        })
        .catch(error => {
            console.error('Error fetching latest health data:', error);
            latestDataContainer.innerHTML = `
                <div class="error-message">
                    <i class="fas fa-exclamation-triangle"></i>
                    <p>Unable to load health data. Please check your connection and try again.</p>
                    <button class="btn" onclick="fetchLatestHealthData()">Retry</button>
                </div>
            `;
        });
}

/**
 * Displays the latest health data on the home page
 * @param {Object} data - The health data from the API
 * @param {HTMLElement} container - The container element for the data
 */
function displayLatestData(data, container) {
    // If no data available
    if (!data || Object.keys(data).length === 0) {
        container.innerHTML = `
            <div class="no-data-message">
                <i class="fas fa-info-circle"></i>
                <p>No health data available yet. Please connect your health monitoring device.</p>
            </div>
        `;
        return;
    }
    
    // Format timestamp
    const timestamp = new Date(data.timestamp);
    const timeAgo = getTimeAgo(timestamp);
    
    // Create risk level indicator class
    let riskClass = 'neutral';
    if (data.vitals_risk_level === 'Low') {
        riskClass = 'low-risk';
    } else if (data.vitals_risk_level === 'Moderate') {
        riskClass = 'moderate-risk';
    } else if (data.vitals_risk_level === 'High') {
        riskClass = 'high-risk';
    }
    
    // Build HTML for stats display
    const statsHTML = `
        <div class="stats-header">
            <h3>Last Updated: ${timeAgo}</h3>
            <p class="timestamp">${timestamp.toLocaleString()}</p>
        </div>
        
        <div class="stats-grid">
            <div class="stat-item">
                <div class="stat-value">${data.heart_rate || '--'}</div>
                <div class="stat-label">Heart Rate (bpm)</div>
            </div>
            
            <div class="stat-item">
                <div class="stat-value">${data.spo2 || '--'}%</div>
                <div class="stat-label">Blood Oxygen</div>
            </div>
            
            <div class="stat-item">
                <div class="stat-value">${data.temperature ? data.temperature.toFixed(1) : '--'}°C</div>
                <div class="stat-label">Temperature</div>
            </div>
            
            <div class="stat-item">
                <div class="stat-value">${data.systolic_bp || '--'}/${data.diastolic_bp || '--'}</div>
                <div class="stat-label">Blood Pressure (mmHg)</div>
            </div>
            
            <div class="stat-item">
                <div class="stat-value">${data.respiratory_rate || '--'}</div>
                <div class="stat-label">Resp. Rate (br/min)</div>
            </div>
            
            <div class="stat-item">
                <div class="stat-value ${riskClass}">${data.vitals_risk_level || 'Unknown'}</div>
                <div class="stat-label">Risk Level</div>
            </div>
        </div>
        
        <div class="stats-footer">
            <a href="/dashboard" class="btn"><i class="fas fa-chart-line"></i> View Full Dashboard</a>
            <a href="/chatbot" class="btn btn-secondary"><i class="fas fa-robot"></i> Ask Health Advisor</a>
        </div>
    `;
    
    container.innerHTML = statsHTML;
}

/**
 * Returns a friendly string representing time elapsed since the given date
 * @param {Date} date - The date to compare against current time
 * @return {string} A string like "2 minutes ago" or "5 hours ago"
 */
function getTimeAgo(date) {
    const seconds = Math.floor((new Date() - date) / 1000);
    
    let interval = Math.floor(seconds / 31536000);
    if (interval > 1) return interval + ' years ago';
    if (interval === 1) return '1 year ago';
    
    interval = Math.floor(seconds / 2592000);
    if (interval > 1) return interval + ' months ago';
    if (interval === 1) return '1 month ago';
    
    interval = Math.floor(seconds / 86400);
    if (interval > 1) return interval + ' days ago';
    if (interval === 1) return '1 day ago';
    
    interval = Math.floor(seconds / 3600);
    if (interval > 1) return interval + ' hours ago';
    if (interval === 1) return '1 hour ago';
    
    interval = Math.floor(seconds / 60);
    if (interval > 1) return interval + ' minutes ago';
    if (interval === 1) return '1 minute ago';
    
    if (seconds < 10) return 'just now';
    
    return Math.floor(seconds) + ' seconds ago';
}