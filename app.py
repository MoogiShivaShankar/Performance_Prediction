from flask import Flask, request, jsonify, render_template
from flask_cors import CORS
import pickle
import numpy as np
import os

app = Flask(__name__)
CORS(app)  # Enable CORS for all domains on all routes

# Load the model and scaler
try:
    with open('cricket_model.pkl', 'rb') as f:
        model = pickle.load(f)
    with open('scaler.pkl', 'rb') as f:
        scaler = pickle.load(f)
except FileNotFoundError:
    print("Error: Model or scaler file not found.")
    exit(1)
except Exception as e:
    print(f"Error loading model or scaler: {str(e)}")
    exit(1)

@app.route('/')
def home():
    return render_template('home.html')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        data = request.json
        required_features = ['Batting_Position', 'Runs', 'Balls', '_4s', '_6s', 'Strike_Rate', 
                             'Maidens', 'Wickets', 'Economy', 'Overs']
        
        # Check if all required features are present
        if not all(feature in data for feature in required_features):
            return jsonify({'error': 'Missing required features'}), 400

        features = [data[feature] for feature in required_features]
        features_scaled = scaler.transform([features])
        prediction = model.predict(features_scaled)[0]
        return jsonify({'predicted_performance_score': float(prediction)})
    except Exception as e:
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True)