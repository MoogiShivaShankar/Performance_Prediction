# model_training.py

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestRegressor
from joblib import dump

# Load your data
df = pd.read_csv('your_data.csv')  # Replace 'your_data.csv' with your actual data file name

# Prepare your features and target
X = df[['Team', 'Batting_Position', 'Runs', 'Balls', '4s', '6s', 'Strike_Rate', 
        'Maidens', 'Wickets', 'Economy', 'Overs']]
y = df['Performance_Score']  # Replace 'Performance_Score' with your actual target column name

# Split the data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Scale the features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Train the model
model = RandomForestRegressor(n_estimators=100, random_state=42)
model.fit(X_train_scaled, y_train)

# Save the model and scaler
dump(model, 'cricket_model.joblib')
dump(scaler, 'scaler.joblib')

print("Model and scaler saved successfully.")