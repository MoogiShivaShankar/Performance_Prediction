import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from joblib import dump
import os

# Load your data
# Ensure you have the correct path to your data file
data_path = 'world_cup_data.xlsx'  # Update this path
df = pd.read_excel(data_path)

# Preprocess your data
df['Runs'] = df['Runs'].fillna(0)
df['Wickets'] = df['Wickets'].fillna(0)

# Create a simple performance score
df['Performance_Score'] = df['Runs'] + df['Wickets'] * 20

# Select features for prediction
features = ['Batting_Position', 'Runs', 'Balls', '_4s', '_6s', 'Strike_Rate', 'Maidens', 'Wickets', 'Economy', 'Overs']

# Separate features and target
X = df[features]
y = df['Performance_Score']

# Handle missing values in features (X)
imputer_X = SimpleImputer(strategy='mean')
X_imputed = pd.DataFrame(imputer_X.fit_transform(X), columns=X.columns)

# Split the data
X_train, X_test, y_train, y_test = train_test_split(X_imputed, y, test_size=0.2, random_state=42)

# Scale the features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Train the model
rf_model = RandomForestRegressor(n_estimators=100, random_state=42)
rf_model.fit(X_train_scaled, y_train)

# Save the model and scaler using joblib
dump(rf_model, 'cricket_performance_model.joblib')
dump(scaler, 'scaler.joblib')

print("Model and scaler saved successfully.")