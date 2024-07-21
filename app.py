import os
from flask import Flask, request, jsonify, render_template
from flask_cors import CORS
from joblib import load
import numpy as np
import sklearn

app = Flask(__name__)
CORS(app)  # Enable CORS for all domains on all routes

# Print current working directory and list files
print("Current working directory:", os.getcwd())
print("Files in the current directory:", os.listdir())
print("Files in the templates directory:", os.listdir('templates'))  # Add this line

# Print numpy and scikit-learn versions
print(f"Numpy version: {np.__version__}")
print(f"Scikit-learn version: {sklearn.__version__}")

# Set up absolute paths for model and scaler
model_path = os.path.join(os.getcwd(), 'cricket_performance_model.joblib')
scaler_path = os.path.join(os.getcwd(), 'scaler.joblib')

# Print debug information about file paths
print(f"Model path: {model_path}")
print(f"Scaler path: {scaler_path}")
print(f"Model file exists: {os.path.exists(model_path)}")
print(f"Scaler file exists: {os.path.exists(scaler_path)}")

# Load the model and scaler with detailed error handling
try:
    model = load(model_path)
    print("Model loaded successfully")
except Exception as e:
    print(f"Error loading model: {str(e)}")
    model = None

try:
    scaler = load(scaler_path)
    print("Scaler loaded successfully")
except Exception as e:
    print(f"Error loading scaler: {str(e)}")
    scaler = None

# Check if model and scaler loaded successfully
if model is None or scaler is None:
    print("Failed to load model or scaler. Exiting.")
    exit(1)

@app.route('/')
def home():
    return render_template('home.html')  # Make sure this matches your HTML file name

@app.route('/predict', methods=['POST'])
def predict():
    try:
        data = request.json
        print("Received data:", data)  # Debug print
        required_features = ['Batting_Position', 'Runs', 'Balls', '_4s', '_6s', 'Strike_Rate', 
                             'Maidens', 'Wickets', 'Economy', 'Overs']
        
        # Check if all required features are present
        if not all(feature in data for feature in required_features):
            return jsonify({'error': 'Missing required features'}), 400

        features = [float(data[feature]) for feature in required_features]
        features_scaled = scaler.transform([features])
        prediction = model.predict(features_scaled)[0]
        print("Prediction:", prediction)  # Debug print
        return jsonify({'predicted_performance_score': float(prediction)})
    except Exception as e:
        print("Error:", str(e))  # Debug print
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True)