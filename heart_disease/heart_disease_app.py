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
st.title("Heart Disease Prediction")
st.write("Predict heart disease risk based on health data")

# Input fields
age = st.number_input("Age (years)", min_value=18, max_value=100, value=50)
sex = st.selectbox("Sex", ['Male', 'Female'])
cp = st.selectbox("Chest Pain Type", ['Typical angina', 'Atypical angina', 'Non-anginal pain', 'Asymptomatic'])
trestbps = st.number_input("Resting Blood Pressure (mm Hg)", min_value=80, max_value=200, value=120)
chol = st.number_input("Serum Cholesterol (mg/dl)", min_value=100, max_value=400, value=200)
fbs = st.selectbox("Fasting Blood Sugar > 120 mg/dl", ['No', 'Yes'])
restecg = st.selectbox("Resting ECG Results", ['Normal', 'ST-T wave abnormality', 'Left ventricular hypertrophy'])
thalach = st.number_input("Maximum Heart Rate Achieved", min_value=60, max_value=220, value=150)
exang = st.selectbox("Exercise-Induced Angina", ['No', 'Yes'])
oldpeak = st.number_input("ST Depression Induced by Exercise", min_value=0.0, max_value=6.0, value=1.0, step=0.1)
slope = st.selectbox("Slope of Peak Exercise ST Segment", ['Upsloping', 'Flat', 'Downsloping'])
ca = st.number_input("Number of Major Vessels Colored by Fluoroscopy", min_value=0, max_value=3, value=0)
thal = st.selectbox("Thalassemia", ['Normal', 'Fixed defect', 'Reversible defect'])

# Encode inputs
sex_map = {'Male': 1, 'Female': 0}
cp_map = {'Typical angina': 0, 'Atypical angina': 1, 'Non-anginal pain': 2, 'Asymptomatic': 3}
fbs_map = {'Yes': 1, 'No': 0}
restecg_map = {'Normal': 0, 'ST-T wave abnormality': 1, 'Left ventricular hypertrophy': 2}
exang_map = {'Yes': 1, 'No': 0}
slope_map = {'Upsloping': 0, 'Flat': 1, 'Downsloping': 2}
thal_map = {'Normal': 0, 'Fixed defect': 1, 'Reversible defect': 2}

# Create input DataFrame
input_data = pd.DataFrame({
    'age': [age],
    'sex': [sex_map[sex]],
    'cp': [cp_map[cp]],
    'trestbps': [trestbps],
    'chol': [chol],
    'fbs': [fbs_map[fbs]],
    'restecg': [restecg_map[restecg]],
    'thalach': [thalach],
    'exang': [exang_map[exang]],
    'oldpeak': [oldpeak],
    'slope': [slope_map[slope]],
    'ca': [ca],
    'thal': [thal_map[thal]],
    'age_chol': [age * chol]
})

# Ensure column order
expected_columns = ['age', 'sex', 'cp', 'trestbps', 'chol', 'fbs', 'restecg', 'thalach', 'exang', 'oldpeak', 'slope', 'ca', 'thal', 'age_chol']
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
        result = "High Risk" if prediction == 1 else "Low Risk"
        st.success(f"Prediction: {result} of heart disease")
        st.write(f"Model Accuracy: {accuracy*100:.2f}%")
    except Exception as e:
        st.error(f"Error making prediction: {e}")