# app.py

import streamlit as st
import numpy as np
import pandas as pd
import joblib

# Load the saved models and preprocessing objects
@st.cache_resource
def load_model():
    rfc = joblib.load("models/best_rfc.pkl")
    scaler = joblib.load("models/scaler.pkl")
    le = joblib.load("models/label_encoder.pkl")
    return rfc, scaler, le

model, scaler, le = load_model()

# Set the title of the app
st.title("🌾 Crop Recommendation System")
st.write("""
This app predicts the best crop to grow based on the given input parameters. Please fill in the values in the sidebar.
""")

# Sidebar for user input
st.sidebar.header("Input Parameters")

def user_input_features():
    N = st.sidebar.number_input(
        'Nitrogen (N)', 
        min_value=0, 
        max_value=140, 
        value=0, 
        step=1,
        help="Nitrogen is essential for plant growth."
    )
    P = st.sidebar.number_input(
        'Phosphorus (P)', 
        min_value=5, 
        max_value=145, 
        value=5,  # Updated default value
        step=1,
        help="Phosphorus helps in root development."
    )
    K = st.sidebar.number_input(
        'Potassium (K)', 
        min_value=5, 
        max_value=205, 
        value=5,  # Updated default value
        step=1,
        help="Potassium regulates various processes in plants."
    )
    temperature = st.sidebar.slider(
        'Temperature (°C)', 
        min_value=0.0, 
        max_value=50.0, 
        value=25.0, 
        step=0.1,
        help="Ideal temperature for crop growth."
    )
    humidity = st.sidebar.slider(
        'Humidity (%)', 
        min_value=0.0, 
        max_value=100.0, 
        value=50.0, 
        step=0.1,
        help="Humidity level affecting crop yield."
    )
    ph = st.sidebar.slider(
        'Soil pH Level', 
        min_value=0.0, 
        max_value=14.0, 
        value=6.5, 
        step=0.1,
        help="Soil pH affects nutrient availability."
    )
    rainfall = st.sidebar.number_input(
        'Rainfall (mm)', 
        min_value=0, 
        max_value=3000, 
        value=0, 
        step=1,
        help="Amount of rainfall during the crop growing season."
    )
    
    data = {
        'N': N,
        'P': P,
        'K': K,
        'temperature': temperature,
        'humidity': humidity,
        'ph': ph,
        'rainfall': rainfall
    }
    features = pd.DataFrame(data, index=[0])
    return features

input_df = user_input_features()

# Display user input
st.subheader('User Input Parameters')
st.write(input_df)

# Prediction
if st.button('Predict'):
    try:
        # Preprocess the input data
        input_scaled = scaler.transform(input_df)
        
        # Make prediction
        prediction = model.predict(input_scaled)[0]
        crop = le.inverse_transform([prediction])[0]
        
        # Display the result
        st.success(f"The best crop to grow is: **{crop.capitalize()}**")
    except Exception as e:
        st.error(f"An error occurred: {e}")