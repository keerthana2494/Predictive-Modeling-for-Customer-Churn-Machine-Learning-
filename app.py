from flask import Flask, request, jsonify
import joblib
import numpy as np

# Initialize Flask app
app = Flask(__name__)

# Load the trained model
model = joblib.load('customer_churn_model.pkl')

@app.route('/')
def home():
    return "Welcome to the Customer Churn Prediction API!"

@app.route('/predict', methods=['POST'])
def predict():
    # Get JSON data from request
    data = request.get_json()
    features = np.array(data['features']).reshape(1, -1)

    # Make prediction
    prediction = model.predict(features)[0]
    probability = model.predict_proba(features)[0]

    # Return result
    return jsonify({
        'churn': bool(prediction),
        'probability': float(probability[1])  # Probability of churn
    })

if __name__ == '__main__':
    app.run(debug=True)
