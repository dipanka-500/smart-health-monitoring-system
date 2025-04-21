"""
Health Monitoring Data Server
Receives health data from ESP32, processes with ML models,
stores in SQLite database and serves a web interface with LLM-powered chatbot
"""

# ===== IMPORTS =====
from flask import Flask, request, jsonify, render_template, send_from_directory
import sqlite3
from datetime import datetime
import logging
import numpy as np
import tensorflow as tf
import cv2
import threading
import time
import json
import os
import requests

# ===== CONFIGURATION =====
app = Flask(__name__, 
            static_folder='static',
            template_folder='templates')

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# ML Model paths - replace with your actual paths
VITALS_MODEL_PATH = "path/to/vitals_risk_model.h5"
FACIAL_MODEL_PATH = "path/to/facial_expression_model.h5"

# Camera settings
CAMERA_PORT = 0
FRAME_WIDTH = 640
FRAME_HEIGHT = 480
FACE_CASCADE_PATH = "haarcascade_frontalface_default.xml"

# Expression labels from AffectNet
EXPRESSION_LABELS = ['Neutral', 'Happy', 'Sad', 'Surprise', 'Fear', 'Disgust', 'Anger', 'Contempt']

# Health risk categories
RISK_CATEGORIES = ['Low', 'Moderate', 'High']

# LLM API configuration - using Llama 2
LLM_API_URL = "https://api.together.xyz/v1/completions"
LLM_API_KEY = "your_api_key_here"  # Replace with your actual API key
LLM_MODEL = "togethercomputer/llama-2-7b-chat"

# Global variables
latest_face_expression = "Unknown"
face_detection_active = True

# ===== DATABASE FUNCTIONS =====
def init_db():
    """Initialize the SQLite database with required tables"""
    conn = sqlite3.connect('health_data.db')
    c = conn.cursor()
    
    # Create table with expanded columns for ML outputs
    c.execute('''CREATE TABLE IF NOT EXISTS health_records
                 (id INTEGER PRIMARY KEY AUTOINCREMENT,
                  timestamp DATETIME DEFAULT CURRENT_TIMESTAMP,
                  age INTEGER,
                  gender TEXT,
                  weight REAL,
                  height REAL,
                  heart_rate REAL,
                  avg_heart_rate REAL,
                  spo2 INTEGER,
                  spo2_valid INTEGER,
                  temperature REAL,
                  respiratory_rate INTEGER,
                  systolic_bp INTEGER,
                  diastolic_bp INTEGER,
                  pulse_pressure INTEGER,
                  map REAL,
                  derived_bmi REAL,
                  derived_hrv REAL,
                  vitals_risk_level TEXT,
                  facial_expression TEXT,
                  combined_assessment TEXT)''')
    
    # Create table for chat history
    c.execute('''CREATE TABLE IF NOT EXISTS chat_history
                 (id INTEGER PRIMARY KEY AUTOINCREMENT,
                  timestamp DATETIME DEFAULT CURRENT_TIMESTAMP,
                  user_message TEXT,
                  bot_response TEXT,
                  health_data_id INTEGER,
                  FOREIGN KEY (health_data_id) REFERENCES health_records (id))''')
    
    conn.commit()
    conn.close()
    logger.info("Database initialized with ML output columns and chat history")

# ===== ML MODEL FUNCTIONS =====
def load_ml_models():
    """Load the machine learning models"""
    try:
        # Load health vitals risk assessment model
        vitals_model = tf.keras.models.load_model(VITALS_MODEL_PATH)
        logger.info("Vitals risk assessment model loaded successfully")
        
        # Load facial expression recognition model
        facial_model = tf.keras.models.load_model(FACIAL_MODEL_PATH)
        logger.info("Facial expression model loaded successfully")
        
        return vitals_model, facial_model
    
    except Exception as e:
        logger.error(f"Error loading ML models: {str(e)}")
        return None, None

def preprocess_health_data(data):
    """Prepare health data for ML model input"""
    # Extract relevant features for the model
    features = np.array([[
        float(data.get('age', 0)),
        1.0 if data.get('gender') == 'M' else 0.0,  # Gender encoding
        float(data.get('heart_rate', 0)),
        float(data.get('avg_heart_rate', 0)),
        float(data.get('spo2', 0)),
        float(data.get('temperature', 0)), 
        float(data.get('respiratory_rate', 0)),
        float(data.get('systolic_bp', 0)),
        float(data.get('diastolic_bp', 0)),
        float(data.get('pulse_pressure', 0)),
        float(data.get('map', 0)),
        float(data.get('derived_bmi', 0))
    ]])
    
    return features

def assess_health_risk(vitals_model, data):
    """Assess health risk using the vitals ML model"""
    try:
        # Preprocess the input data
        processed_data = preprocess_health_data(data)
        
        # Get model prediction
        prediction = vitals_model.predict(processed_data)
        risk_index = np.argmax(prediction[0])
        risk_level = RISK_CATEGORIES[risk_index]
        confidence = float(prediction[0][risk_index])
        
        logger.info(f"Health risk assessment: {risk_level} (confidence: {confidence:.2f})")
        return risk_level
        
    except Exception as e:
        logger.error(f"Error in health risk assessment: {str(e)}")
        return "Unknown"

def combine_assessments(vitals_risk, face_expression):
    """Combine vitals risk and facial expression into overall assessment"""
    # Map facial expressions to potential health states
    negative_expressions = ['Sad', 'Fear', 'Disgust', 'Anger']
    positive_expressions = ['Neutral', 'Happy']
    
    combined = vitals_risk  # Start with vitals risk level
    
    # Adjust assessment based on facial expression
    if vitals_risk == "Low" and face_expression in negative_expressions:
        combined = "Low-Moderate (Physical metrics good, negative expression)"
    elif vitals_risk == "High" and face_expression in positive_expressions:
        combined = "Moderate-High (Physical metrics concerning despite positive expression)"
    elif vitals_risk == "Moderate":
        if face_expression in negative_expressions:
            combined = "Moderate-High (Emotional distress detected)"
        elif face_expression in positive_expressions:
            combined = "Low-Moderate (Positive emotional state)"
    
    return combined

# ===== FACIAL EXPRESSION DETECTION =====
def init_camera():
    """Initialize camera for facial expression monitoring"""
    try:
        cap = cv2.VideoCapture(CAMERA_PORT)
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, FRAME_WIDTH)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, FRAME_HEIGHT)
        
        if not cap.isOpened():
            logger.error("Failed to open camera")
            return None
            
        return cap
    except Exception as e:
        logger.error(f"Camera initialization error: {str(e)}")
        return None

def detect_face_expression(facial_model, frame):
    """Detect and classify facial expression from camera frame"""
    try:
        # Load face cascade for detection
        face_cascade = cv2.CascadeClassifier(FACE_CASCADE_PATH)
        
        # Convert to grayscale for face detection
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
        # Detect faces
        faces = face_cascade.detectMultiScale(gray, 1.1, 4)
        
        if len(faces) > 0:
            # Process the first face found
            x, y, w, h = faces[0]
            
            # Extract face ROI
            face_roi = frame[y:y+h, x:x+w]
            
            # Preprocess for model input
            resized_face = cv2.resize(face_roi, (224, 224))  # Adjust size to match model input
            normalized_face = resized_face / 255.0  # Normalize pixel values
            
            # Add batch dimension
            model_input = np.expand_dims(normalized_face, axis=0)
            
            # Get prediction
            prediction = facial_model.predict(model_input)
            expression_index = np.argmax(prediction[0])
            expression = EXPRESSION_LABELS[expression_index]
            
            return expression
        
        return "No face detected"
    
    except Exception as e:
        logger.error(f"Error in facial expression detection: {str(e)}")
        return "Error in detection"

def facial_expression_monitor(facial_model):
    """Background thread to continuously monitor facial expressions"""
    global latest_face_expression, face_detection_active
    
    cap = init_camera()
    if cap is None:
        logger.error("Facial expression monitoring disabled - camera initialization failed")
        return
    
    logger.info("Starting facial expression monitoring")
    
    try:
        while face_detection_active:
            ret, frame = cap.read()
            
            if not ret:
                logger.warning("Failed to capture frame")
                time.sleep(1)
                continue
                
            # Process every 5th frame to reduce CPU usage
            if int(time.time()) % 5 == 0:
                expression = detect_face_expression(facial_model, frame)
                latest_face_expression = expression
                logger.info(f"Current facial expression: {expression}")
            
            time.sleep(0.1)  # Small delay to prevent 100% CPU usage
    
    except Exception as e:
        logger.error(f"Facial monitoring thread error: {str(e)}")
    
    finally:
        cap.release()
        logger.info("Facial expression monitoring stopped")

# ===== LLM CHATBOT FUNCTIONS =====
def get_chat_prompt(user_message, health_data):
    """Create a prompt for the LLM with health data context"""
    
    # Format the health data into a readable summary
    health_summary = f"""
Current Health Data:
- Age: {health_data.get('age')}
- Gender: {health_data.get('gender')}
- BMI: {health_data.get('derived_bmi', 'Unknown')}
- Heart Rate: {health_data.get('heart_rate', 'Unknown')} bpm
- Blood Oxygen (SpO2): {health_data.get('spo2', 'Unknown')}%
- Temperature: {health_data.get('temperature', 'Unknown')}°C
- Blood Pressure: {health_data.get('systolic_bp', 'Unknown')}/{health_data.get('diastolic_bp', 'Unknown')} mmHg
- Respiratory Rate: {health_data.get('respiratory_rate', 'Unknown')} breaths/min
- Facial Expression: {health_data.get('facial_expression', 'Unknown')}
- Risk Assessment: {health_data.get('vitals_risk_level', 'Unknown')}
- Combined Assessment: {health_data.get('combined_assessment', 'Unknown')}
    """
    
    # Build the LLM prompt
    prompt = f"""<|im_start|>system
You are a helpful, accurate, and friendly health assistant. You provide personalized health advice based on health monitoring data. 
You should give specific, actionable recommendations based on the health data provided.
When giving advice about exercises, diet, or health precautions, be specific but emphasize that your advice is general guidance and not a replacement for medical consultation.
Focus only on providing helpful health advice based on the data and question.
Here is the person's current health data:
{health_summary}
<|im_end|>

<|im_start|>user
{user_message}
<|im_end|>

<|im_start|>assistant
"""
    
    return prompt

def query_llm(prompt):
    """Send a prompt to the LLM API and get a response"""
    try:
        headers = {
            "Authorization": f"Bearer {LLM_API_KEY}",
            "Content-Type": "application/json"
        }
        
        payload = {
            "model": LLM_MODEL,
            "prompt": prompt,
            "max_tokens": 1024,
            "temperature": 0.7,
            "stop": ["<|im_end|>"]
        }
        
        response = requests.post(LLM_API_URL, json=payload, headers=headers)
        
        if response.status_code == 200:
            result = response.json()
            return result["choices"][0]["text"]
        else:
            logger.error(f"LLM API error: {response.status_code} - {response.text}")
            return "I'm sorry, I'm having trouble processing your question right now. Please try again later."
    
    except Exception as e:
        logger.error(f"Error querying LLM: {str(e)}")
        return "I'm sorry, I encountered an error while processing your question. Please try again later."

def save_chat_interaction(user_message, bot_response, health_data_id):
    """Save chat interaction to database"""
    try:
        conn = sqlite3.connect('health_data.db')
        c = conn.cursor()
        
        c.execute('''INSERT INTO chat_history 
                    (timestamp, user_message, bot_response, health_data_id)
                    VALUES (?, ?, ?, ?)''',
                 (datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
                  user_message,
                  bot_response,
                  health_data_id))
        
        conn.commit()
        conn.close()
        
        logger.info(f"Chat interaction saved: Question about {user_message[:30]}...")
        
    except Exception as e:
        logger.error(f"Error saving chat interaction: {str(e)}")

# ===== API ROUTES =====
@app.route('/api/health_data', methods=['POST'])
def receive_health_data():
    """Endpoint to receive health data from ESP32"""
    try:
        # Get data from request
        data = request.get_json()
        logger.info(f"Received health data: {data}")
        
        # Validate required fields
        required_fields = ['age', 'gender', 'weight', 'height']
        if not all(field in data for field in required_fields):
            return jsonify({'error': 'Missing required fields'}), 400
        
        # Process with ML models
        vitals_model, _ = load_ml_models()
        risk_level = assess_health_risk(vitals_model, data)
        
        # Get the latest facial expression from the monitoring thread
        face_expression = latest_face_expression
        
        # Combine assessments
        combined_assessment = combine_assessments(risk_level, face_expression)
        
        # Connect to database
        conn = sqlite3.connect('health_data.db')
        c = conn.cursor()
        
        # Insert data into database with ML outputs
        c.execute('''INSERT INTO health_records 
                    (timestamp, age, gender, weight, height, 
                     heart_rate, avg_heart_rate, spo2, spo2_valid,
                     temperature, respiratory_rate, systolic_bp, diastolic_bp,
                     pulse_pressure, map, derived_bmi, 
                     vitals_risk_level, facial_expression, combined_assessment)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)''',
                 (datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
                  data.get('age'),
                  data.get('gender'),
                  data.get('weight'),
                  data.get('height'),
                  data.get('heart_rate'),
                  data.get('avg_heart_rate'),
                  data.get('spo2'),
                  data.get('spo2_valid'),
                  data.get('temperature'),
                  data.get('respiratory_rate'),
                  data.get('systolic_bp'),
                  data.get('diastolic_bp'),
                  data.get('pulse_pressure'),
                  data.get('map'),
                  data.get('derived_bmi'),
                  risk_level,
                  face_expression,
                  combined_assessment))
        
        conn.commit()
        conn.close()
        
        # Prepare response with ML insights
        response = {
            'message': 'Data received and stored successfully',
            'analysis': {
                'risk_level': risk_level,
                'facial_expression': face_expression,
                'combined_assessment': combined_assessment
            }
        }
        
        logger.info("Data and analysis stored successfully")
        return jsonify(response), 200
    
    except Exception as e:
        logger.error(f"Error processing data: {str(e)}")
        return jsonify({'error': str(e)}), 500

@app.route('/api/health_data', methods=['GET'])
def get_health_data():
    """Retrieve all health records with ML assessments"""
    try:
        conn = sqlite3.connect('health_data.db')
        c = conn.cursor()
        
        # Get all records sorted by timestamp
        c.execute('SELECT * FROM health_records ORDER BY timestamp DESC')
        records = c.fetchall()
        
        # Convert to list of dictionaries
        columns = [column[0] for column in c.description]
        result = [dict(zip(columns, row)) for row in records]
        
        conn.close()
        
        return jsonify(result), 200
    
    except Exception as e:
        logger.error(f"Error retrieving data: {str(e)}")
        return jsonify({'error': str(e)}), 500

@app.route('/api/latest', methods=['GET'])
def get_latest_record():
    """Get the most recent health record with assessment"""
    try:
        conn = sqlite3.connect('health_data.db')
        c = conn.cursor()
        
        # Get the most recent record
        c.execute('SELECT * FROM health_records ORDER BY timestamp DESC LIMIT 1')
        record = c.fetchone()
        
        if record:
            columns = [column[0] for column in c.description]
            result = dict(zip(columns, record))
            conn.close()
            return jsonify(result), 200
        else:
            conn.close()
            return jsonify({'message': 'No records found'}), 404
    
    except Exception as e:
        logger.error(f"Error retrieving latest record: {str(e)}")
        return jsonify({'error': str(e)}), 500

@app.route('/api/facial_expression', methods=['GET'])
def get_current_expression():
    """Get the current facial expression from camera monitoring"""
    return jsonify({
        'facial_expression': latest_face_expression,
        'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    })

@app.route('/api/chat', methods=['POST'])
def chat_with_llm():
    """Endpoint to interact with the LLM chatbot"""
    try:
        # Get user message from request
        user_message = request.json.get('message', '')
        if not user_message:
            return jsonify({'error': 'Message cannot be empty'}), 400
        
        # Get the latest health data for context
        conn = sqlite3.connect('health_data.db')
        conn.row_factory = sqlite3.Row  # Get results as dictionaries
        c = conn.cursor()
        
        c.execute('SELECT * FROM health_records ORDER BY timestamp DESC LIMIT 1')
        health_data = dict(c.fetchone())
        health_data_id = health_data['id']
        
        # Create a prompt with the health data context
        prompt = get_chat_prompt(user_message, health_data)
        
        # Query the LLM
        bot_response = query_llm(prompt)
        
        # Save the interaction
        save_chat_interaction(user_message, bot_response, health_data_id)
        
        conn.close()
        
        return jsonify({
            'response': bot_response,
            'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        }), 200
    
    except Exception as e:
        logger.error(f"Error in chat interaction: {str(e)}")
        return jsonify({'error': str(e)}), 500

@app.route('/api/chat_history', methods=['GET'])
def get_chat_history():
    """Get chat history for the web interface"""
    try:
        limit = request.args.get('limit', default=10, type=int)
        
        conn = sqlite3.connect('health_data.db')
        c = conn.cursor()
        
        c.execute('''SELECT * FROM chat_history 
                     ORDER BY timestamp DESC 
                     LIMIT ?''', (limit,))
        
        records = c.fetchall()
        
        # Convert to list of dictionaries
        columns = [column[0] for column in c.description]
        result = [dict(zip(columns, row)) for row in records]
        
        conn.close()
        
        return jsonify(result), 200
    
    except Exception as e:
        logger.error(f"Error retrieving chat history: {str(e)}")
        return jsonify({'error': str(e)}), 500

# ===== WEB INTERFACE ROUTES =====
@app.route('/')
def index():
    """Serve the main web interface"""
    return render_template('index.html')

@app.route('/dashboard')
def dashboard():
    """Serve the health dashboard page"""
    return render_template('dashboard.html')

@app.route('/chatbot')
def chatbot():
    """Serve the chatbot interface"""
    return render_template('chatbot.html')

# ===== MAIN ENTRY POINT =====
if __name__ == '__main__':
    # Create directories if they don't exist
    os.makedirs('templates', exist_ok=True)
    os.makedirs('static', exist_ok=True)
    os.makedirs('static/js', exist_ok=True)
    os.makedirs('static/css', exist_ok=True)
    
    # Initialize database
    init_db()
    logger.info("Database initialized")
    
    # Load ML models
    vitals_model, facial_model = load_ml_models()
    
    # Start facial expression monitoring in a background thread
    if facial_model is not None:
        face_monitor_thread = threading.Thread(
            target=facial_expression_monitor,
            args=(facial_model,),
            daemon=True
        )
        face_monitor_thread.start()
        logger.info("Facial expression monitoring thread started")
    
    # Start Flask application
    app.run(host='0.0.0.0', port=8080, debug=True)