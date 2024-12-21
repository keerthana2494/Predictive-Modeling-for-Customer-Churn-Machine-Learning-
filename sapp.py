import streamlit as st
import joblib
import numpy as np

# Load the trained model
model = joblib.load('customer_churn_model.pkl')

# App title
st.title("Customer Churn Prediction")

# Input fields
st.header("Enter Customer Details:")
features = [
    st.number_input("Feature 1 (e.g., tenure):"),
    st.number_input("Feature 2 (e.g., monthly charges):"),
    st.number_input("Feature 3 (e.g., total charges):"),
    # Add more input fields as per your feature set
]

# Prediction button
if st.button("Predict"):
    # Reshape input for the model
    features = np.array(features).reshape(1, -1)
    prediction = model.predict(features)[0]
    probability = model.predict_proba(features)[0]

    # Display result
    if prediction:
        st.error(f"The customer is likely to churn. (Probability: {probability[1]:.2f})")
    else:
        st.success(f"The customer is unlikely to churn. (Probability: {probability[1]:.2f})")
