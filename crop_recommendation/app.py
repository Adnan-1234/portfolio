import streamlit as st
import pandas as pd
import joblib
import os

# Define paths
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIR, 'best_crop_recommendation_model.pkl')
SCALER_PATH = os.path.join(BASE_DIR, 'scaler.pkl')
NORMALIZER_PATH = os.path.join(BASE_DIR, 'normalizer.pkl')
LABEL_ENCODER_PATH = os.path.join(BASE_DIR, 'label_encoder.pkl')

# Load model and preprocessors
try:
    model = joblib.load(MODEL_PATH)
    scaler = joblib.load(SCALER_PATH)
    normalizer = joblib.load(NORMALIZER_PATH)
    label_encoder = joblib.load(LABEL_ENCODER_PATH)
except FileNotFoundError as e:
    st.error(f"Error loading files: {e}")
    st.stop()

# Streamlit app
st.title("Crop Recommendation System")
st.write("Predict the best crop based on soil and climate conditions")

# Input fields
N = st.number_input("Nitrogen (N) content in soil", min_value=0.0, max_value=200.0, value=90.0)
P = st.number_input("Phosphorus (P) content in soil", min_value=0.0, max_value=200.0, value=42.0)
K = st.number_input("Potassium (K) content in soil", min_value=0.0, max_value=200.0, value=43.0)
temperature = st.number_input("Temperature (°C)", min_value=0.0, max_value=50.0, value=20.8)
humidity = st.number_input("Humidity (%)", min_value=0.0, max_value=100.0, value=82.0)
ph = st.number_input("Soil pH", min_value=0.0, max_value=14.0, value=6.5)
rainfall = st.number_input("Rainfall (mm)", min_value=0.0, max_value=1000.0, value=202.9)

# Prediction function
def predict_crop(N, P, K, temperature, humidity, ph, rainfall):
    input_data = pd.DataFrame({
        'N': [N],
        'P': [P],
        'K': [K],
        'temperature': [temperature],
        'humidity': [humidity],
        'ph': [ph],
        'rainfall': [rainfall]
    })
    input_scaled = scaler.transform(input_data)
    input_normalized = normalizer.transform(input_scaled)
    prediction = model.predict(input_normalized)[0]
    return label_encoder.inverse_transform([prediction])[0]

# Predict
if st.button("Predict"):
    try:
        result = predict_crop(N, P, K, temperature, humidity, ph, rainfall)
        st.success(f"Recommended Crop: {result}")
    except Exception as e:
        st.error(f"Error making prediction: {e}")