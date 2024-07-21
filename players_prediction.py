import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from joblib import dump
import pickle
import os

# Get the current directory
current_dir = os.path.dirname(os.path.abspath(__file__))

# Construct the full path to the Excel file
file_path = os.path.join(current_dir, 'world_cup_data.xlsx')

# Load the data
df = pd.read_excel('D:\Projects\Performance_Prediction\Performance_Prediction\world_cup_data.xlsx')

# Print the column names
print("Columns in the dataset:", df.columns.tolist())

# Fill NaN values in 'Runs' and 'Wickets' with 0
df['Runs'] = df['Runs'].fillna(0)
df['Wickets'] = df['Wickets'].fillna(0)

# Create a simple performance score
df['Performance_Score'] = df['Runs'] + df['Wickets'] * 20  # Assuming a wicket is worth 20 runs

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
dump(rf_model, 'cricket_model.joblib')
dump(scaler, 'scaler.joblib')

# Save the model and scaler using pickle with protocol 4
with open('cricket_model.pkl', 'wb') as f:
    pickle.dump(rf_model, f, protocol=4)
with open('scaler.pkl', 'wb') as f:
    pickle.dump(scaler, f, protocol=4)

print("Model and scaler saved successfully using both joblib and pickle.")

# Print feature importances
feature_importance = pd.DataFrame({'feature': features, 'importance': rf_model.feature_importances_})
print("\nFeature Importances:")
print(feature_importance.sort_values('importance', ascending=False))