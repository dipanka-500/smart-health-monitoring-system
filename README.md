🩺 Smart Health Monitoring System
An IoT + AI-powered health monitoring system that collects patient vitals using an ESP32, sends data to a Raspberry Pi running a Flask server, and leverages Machine Learning models for health risk analysis and wellness recommendations via an AI chatbot interface powered by LLaMA 4 API.

📦 Project Structure in rpi
php
Copy
Edit
health_monitoring_system/
├── server.py           # Flask backend server
├── templates/          # HTML template files
│   ├── index.html      # User data input page
│   ├── dashboard.html  # Health dashboard visualization
│   └── chatbot.html    # Chat interface for recommendations
├── static/
│   ├── css/
│   │   └── style.css   # Styling for UI
│   └── js/
│       ├── index.js    # Input page JS
│       ├── dashboard.js# Dashboard JS
│       └── chatbot.js  # Chatbot frontend logic
└── health_data.db      # SQLite database (auto-generated)
🛠️ Features
ESP32-based sensor integration:

📟 MAX30102: Heart rate and SpO2

🌡️ MAX30205: Body temperature

🌬️ Respiratory sensor: Breathing rate

🩸 Blood pressure sensor

Local ESP32 Web UI for entering:

Age, Gender, Weight, Height

Data transmission to Raspberry Pi using serial/Wi-Fi

Raspberry Pi with Flask:

Stores data in SQLite

Runs ML models for risk prediction

Frontend Dashboard for visualization

Machine Learning Models:

📊 Risk Prediction Model:

Trained using Kaggle Human Vital Signs Dataset

Classifies risk as High, Moderate, or Low

😊 Facial Expression Model:

Trained using AffectNet Dataset

Detects emotional state for behavioral context

LLM-powered chatbot:

Calls LLaMA 4 API with all sensor + ML model data

Recommends diet, exercises, and general wellness plans

Natural language interaction via chatbot page

🚀 How It Works
ESP32 Setup:

Collects sensor values

Hosts a local webpage for user input

Sends collected data to Raspberry Pi over serial or network

Raspberry Pi (Flask Server):

Receives and stores data

Runs trained ML models for health risk and emotion analysis

Displays everything on a live dashboard

Provides a chat interface to interact with the AI model

Chatbot + LLaMA 4 API:

Uses risk levels + emotional state to generate personalized advice

Example queries:

"What should I eat today?"

"What exercises are safe for me?"

"Why am I feeling anxious?"

📸 Screenshots
(You can add screenshots here from the UI like index page, dashboard, chatbot, etc.)

🧠 Technologies Used
Hardware: ESP32, MAX30102, MAX30205, Blood Pressure Sensor, Respiratory Sensor

Backend: Python (Flask), SQLite, scikit-learn, PyTorch/TensorFlow

Frontend: HTML, CSS, JavaScript

ML Models: Trained on Kaggle datasets

AI API: LLaMA 4 for chat integration

🧪 Future Improvements
Integration with cloud-based health records (e.g., Firebase or AWS)

Real-time alert system for critical conditions

BLE support for wearable integration

Mobile app for cross-platform support

📜 License
MIT License
