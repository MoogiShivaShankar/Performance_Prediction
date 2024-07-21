# Cricket Player Performance Predictor

This project uses machine learning to predict a cricket player's performance based on their match statistics.

## Features
- Developed a Random Forest Regression model for performance prediction
- Created a Flask API for real-time predictions
- Implemented data preprocessing and feature scaling
- Utilized joblib for model serialization

## Technical Skills Used
- Python
- Flask for API development
- Scikit-learn for machine learning
- Pandas for data manipulation
- Joblib for model serialization

## Setup and Usage
1. Clone the repository
2. Install dependencies: `pip install -r requirements.txt`
3. Run the Flask app: `python app.py`
4. Open `http://localhost:5000` in your browser

## API Usage
Send a POST request to `/predict` with JSON data containing player statistics to get a performance prediction.

## Model Details
- We use a Random Forest Regressor to predict player performance.
- The performance score is calculated based on runs, boundaries, and strike rate.
- Features used: Team, Batting Position, Runs, Balls, 4s, 6s, Strike Rate, Maidens, Wickets, Economy, Overs

## Future Improvements
- Incorporate more historical data for better predictions
- Implement feature importance analysis
- Add more advanced models and compare their performance
- Extend the model to predict match outcomes

Feel free to contribute or provide feedback! 