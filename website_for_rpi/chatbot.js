// chatbot.js - JavaScript for the Health Advisor (chatbot) page

document.addEventListener('DOMContentLoaded', function() {
    // Initialize chatbot
    initChatbot();
    
    // Set up chat form submission
    document.getElementById('chat-form').addEventListener('submit', handleChatSubmit);
    
    // Set up suggestion chips
    document.querySelectorAll('.chip').forEach(chip => {
        chip.addEventListener('click', function() {
            const message = this.getAttribute('data-message');
            document.getElementById('user-input').value = message;
            document.getElementById('chat-form').dispatchEvent(new Event('submit'));
        });
    });
    
    // Fetch health summary
    fetchHealthSummary();
});

/**
 * Initialize chatbot
 */
function initChatbot() {
    // Load conversation history
    loadConversationHistory();
    
    // Scroll to bottom of chat
    scrollToBottom();
}

/**
 * Handle chat form submission
 * @param {Event} event - Form submit event
 */
function handleChatSubmit(event) {
    event.preventDefault();
    
    const inputElement = document.getElementById('user-input');
    const userMessage = inputElement.value.trim();
    
    if (userMessage === '') return;
    
    // Add user message to chat
    addUserMessage(userMessage);
    
    // Clear input
    inputElement.value = '';
    
    // Show bot typing indicator
    showBotTyping();
    
    // Send message to chatbot API
    sendMessage(userMessage)
        .then(response => {
            // Remove typing indicator
            removeBotTyping();
            
            // Add bot response to chat
            addBotMessage(response.message);
            
            // Save conversation to history
            saveToHistory(userMessage, response.message);
        })
        .catch(error => {
            console.error('Error sending message:', error);
            
            // Remove typing indicator
            removeBotTyping();
            
            // Add error message
            addBotMessage("I'm sorry, I'm having trouble connecting to the health analysis system. Please try again later.");
        });
}

/**
 * Send message to chatbot API
 * @param {string} message - User message
 * @return {Promise} Promise resolving to response object
 */
function sendMessage(message) {
    // For demonstration purposes, we'll simulate an API response
    return new Promise((resolve) => {
        setTimeout(() => {
            const responses = {
                "What does my risk level mean?": "Your current risk level is an assessment based on your vital signs and health metrics. Low risk means your vitals are within normal ranges. Moderate risk indicates some metrics are outside normal ranges, and you should monitor them closely. High risk suggests immediate attention may be needed.",
                
                "Is my heart rate normal?": "Your heart rate is currently 72 bpm, which falls within the normal resting range (60-100 bpm) for adults. Based on your age and activity level, this is considered healthy.",
                
                "What exercises should I do?": "Based on your current health metrics and risk assessment, I recommend moderate aerobic exercises like brisk walking for 30 minutes daily, along with light strength training twice a week. Remember to stay hydrated and monitor your heart rate during exercise.",
                
                "How can I improve my health metrics?": "To improve your health metrics, consider these recommendations: 1) Stay hydrated by drinking at least 8 glasses of water daily, 2) Aim for 7-8 hours of quality sleep, 3) Include more fruits and vegetables in your diet, 4) Practice stress reduction techniques like meditation, and 5) Maintain regular physical activity with at least 150 minutes of moderate exercise weekly."
            };
            
            // Check for exact matches in our response dictionary
            if (responses[message]) {
                resolve({ message: responses[message] });
            } else {
                // Check for partial matches
                for (const key in responses) {
                    if (message.toLowerCase().includes(key.toLowerCase())) {
                        resolve({ message: responses[key] });
                        return;
                    }
                }
                
                // Default response
                resolve({ message: "I understand you're asking about your health. Based on your current metrics, everything looks stable. Is there a specific aspect of your health you'd like to know more about?" });
            }
        }, 1000);
    });
}

/**
 * Add user message to chat
 * @param {string} message - User message
 */
function addUserMessage(message) {
    const messagesContainer = document.getElementById('chat-messages');
    const messageElement = document.createElement('div');
    messageElement.classList.add('message', 'user-message');
    
    messageElement.innerHTML = `
        <div class="message-avatar">
            <i class="fas fa-user"></i>
        </div>
        <div class="message-content">
            <p>${escapeHtml(message)}</p>
        </div>
    `;
    
    messagesContainer.appendChild(messageElement);
    scrollToBottom();
}

/**
 * Add bot message to chat
 * @param {string} message - Bot message
 */
function addBotMessage(message) {
    const messagesContainer = document.getElementById('chat-messages');
    const messageElement = document.createElement('div');
    messageElement.classList.add('message', 'bot-message');
    
    messageElement.innerHTML = `
        <div class="message-avatar">
            <i class="fas fa-robot"></i>
        </div>
        <div class="message-content">
            <p>${message}</p>
        </div>
    `;
    
    messagesContainer.appendChild(messageElement);
    scrollToBottom();
}

/**
 * Show bot typing indicator
 */
function showBotTyping() {
    const messagesContainer = document.getElementById('chat-messages');
    const typingElement = document.createElement('div');
    typingElement.classList.add('message', 'bot-message', 'typing-indicator');
    typingElement.id = 'typing-indicator';
    
    typingElement.innerHTML = `
        <div class="message-avatar">
            <i class="fas fa-robot"></i>
        </div>
        <div class="message-content">
            <div class="typing-dots">
                <span class="dot"></span>
                <span class="dot"></span>
                <span class="dot"></span>
            </div>
        </div>
    `;
    
    messagesContainer.appendChild(typingElement);
    scrollToBottom();
}

/**
 * Remove bot typing indicator
 */
function removeBotTyping() {
    const typingIndicator = document.getElementById('typing-indicator');
    if (typingIndicator) {
        typingIndicator.remove();
    }
}

/**
 * Scroll to bottom of chat messages
 */
function scrollToBottom() {
    const messagesContainer = document.getElementById('chat-messages');
    messagesContainer.scrollTop = messagesContainer.scrollHeight;
}

/**
 * Save conversation to history
 * @param {string} userMessage - User message
 * @param {string} botMessage - Bot response
 */
function saveToHistory(userMessage, botMessage) {
    const historyItems = getHistoryFromStorage();
    
    const newItem = {
        id: Date.now(),
        timestamp: new Date().toISOString(),
        userMessage: userMessage,
        botMessage: botMessage
    };
    
    // Add new item to beginning of array
    historyItems.unshift(newItem);
    
    // Keep only the latest 5 conversations
    const trimmedHistory = historyItems.slice(0, 5);
    
    // Save to local storage
    localStorage.setItem('chatHistory', JSON.stringify(trimmedHistory));
    
    // Update history display
    displayConversationHistory();
}

/**
 * Get conversation history from local storage
 * @return {Array} Array of conversation history items
 */
function getHistoryFromStorage() {
    const history = localStorage.getItem('chatHistory');
    return history ? JSON.parse(history) : [];
}

/**
 * Load conversation history display
 */
function loadConversationHistory() {
    displayConversationHistory();
}

/**
 * Display conversation history on the page
 */
function displayConversationHistory() {
    const historyContainer = document.getElementById('conversation-history');
    const historyItems = getHistoryFromStorage();
    
    // Clear current history display
    historyContainer.innerHTML = '';
    
    if (historyItems.length === 0) {
        const emptyMessage = document.createElement('p');
        emptyMessage.classList.add('empty-history');
        emptyMessage.textContent = 'No conversation history yet.';
        historyContainer.appendChild(emptyMessage);
        return;
    }
    
    // Add history items to display
    historyItems.forEach(item => {
        const historyElement = document.createElement('div');
        historyElement.classList.add('history-item');
        
        const date = new Date(item.timestamp);
        const formattedDate = `${date.toLocaleDateString()} ${date.toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' })}`;
        
        historyElement.innerHTML = `
            <div class="history-header">
                <span class="history-time">${formattedDate}</span>
                <button class="history-replay" data-message="${escapeHtml(item.userMessage)}">
                    <i class="fas fa-redo-alt"></i>
                </button>
            </div>
            <div class="history-content">
                <p class="history-user-message">${escapeHtml(item.userMessage)}</p>
                <p class="history-bot-message">${item.botMessage}</p>
            </div>
        `;
        
        historyContainer.appendChild(historyElement);
        
        // Add event listener to replay button
        historyElement.querySelector('.history-replay').addEventListener('click', function() {
            const message = this.getAttribute('data-message');
            document.getElementById('user-input').value = message;
            document.getElementById('chat-form').dispatchEvent(new Event('submit'));
        });
    });
}

/**
 * Fetch health summary data from API
 */
function fetchHealthSummary() {
    // For demonstration, we'll use simulated data
    setTimeout(() => {
        const healthData = {
            risk: "Low",
            heartRate: 72,
            bloodOxygen: 98,
            temperature: 36.6,
            bloodPressure: {
                systolic: 120,
                diastolic: 80
            }
        };
        
        updateHealthSummary(healthData);
    }, 800);
}

/**
 * Update health summary display
 * @param {Object} data - Health data object
 */
function updateHealthSummary(data) {
    document.getElementById('summary-risk').textContent = data.risk;
    document.getElementById('summary-hr').textContent = `${data.heartRate} bpm`;
    document.getElementById('summary-spo2').textContent = `${data.bloodOxygen}%`;
    document.getElementById('summary-temp').textContent = `${data.temperature} °C`;
    document.getElementById('summary-bp').textContent = `${data.bloodPressure.systolic}/${data.bloodPressure.diastolic} mmHg`;
    
    // Add risk level class
    const riskElement = document.getElementById('summary-risk');
    riskElement.className = 'value'; // Reset classes
    
    if (data.risk === 'Low') {
        riskElement.classList.add('risk-low');
    } else if (data.risk === 'Moderate') {
        riskElement.classList.add('risk-moderate');
    } else if (data.risk === 'High') {
        riskElement.classList.add('risk-high');
    }
}

/**
 * Escape HTML special characters
 * @param {string} text - Text to escape
 * @return {string} Escaped text
 */
function escapeHtml(text) {
    const div = document.createElement('div');
    div.textContent = text;
    return div.innerHTML;
}