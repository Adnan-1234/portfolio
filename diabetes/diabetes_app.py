import streamlit as st
import pickle
import pandas as pd
import os

# Define paths
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIR, 'model.pkl')
SCALER_PATH = os.path.join(BASE_DIR, 'scaler.pkl')
ACCURACY_PATH = os.path.join(BASE_DIR, 'accuracy.pkl')

# Load model, scaler, and accuracy
try:
    model = pickle.load(open(MODEL_PATH, 'rb'))
    scaler = pickle.load(open(SCALER_PATH, 'rb'))
    accuracy = pickle.load(open(ACCURACY_PATH, 'rb'))
except FileNotFoundError as e:
    st.error(f"Error loading files: {e}")
    st.stop()

# Streamlit app
st.title("Diabetes Progression Prediction")
st.write("Predict diabetes progression based on health metrics")

# Input fields for all 10 features
age = st.number_input("Age (years)", min_value=18, max_value=100, value=50)
sex = st.selectbox("Sex", ["Male", "Female"])
bmi = st.number_input("BMI", min_value=10.0, max_value=50.0, value=25.0)
bp = st.number_input("Blood Pressure (mm Hg)", min_value=80, max_value=200, value=120)
s1 = st.number_input("Total Cholesterol (mg/dL)", min_value=100, max_value=300, value=200)
s2 = st.number_input("LDL Cholesterol (mg/dL)", min_value=50, max_value=200, value=100)
s3 = st.number_input("HDL Cholesterol (mg/dL)", min_value=20, max_value=100, value=50)
s4 = st.number_input("Total Cholesterol / HDL", min_value=1.0, max_value=10.0, value=4.0)
s5 = st.number_input("Log of Serum Triglycerides", min_value=4.0, max_value=6.0, value=5.0)
s6 = st.number_input("Blood Sugar Level (mg/dL)", min_value=50, max_value=200, value=100)

# Encode sex
sex_value = 1 if sex == "Male" else 0

# Create input DataFrame
input_data = pd.DataFrame({
    'age': [age],
    'sex': [sex_value],
    'bmi': [bmi],
    'bp': [bp],
    's1': [s1],
    's2': [s2],
    's3': [s3],
    's4': [s4],
    's5': [s5],
    's6': [s6]
})

# Ensure column order
expected_columns = ['age', 'sex', 'bmi', 'bp', 's1', 's2', 's3', 's4', 's5', 's6']
input_data = input_data[expected_columns]

# Scale input
try:
    input_scaled = scaler.transform(input_data)
except Exception as e:
    st.error(f"Error scaling input data: {e}")
    st.stop()

# Predict
if st.button("Predict"):
    try:
        prediction = model.predict(input_scaled)[0]
        st.success(f"Predicted Diabetes Progression Score: {prediction:.2f}")
        st.write(f"Model R² Score: {accuracy*100:.2f}%")
    except Exception as e:
        st.error(f"Error making prediction: {e}")