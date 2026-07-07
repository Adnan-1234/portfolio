import streamlit as st
import numpy as np
from PIL import Image
import tensorflow as tf
import os

# Define paths
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIR, 'skin_cancer_model.h5')

# Load model
try:
    model = tf.keras.models.load_model(MODEL_PATH)
except Exception as e:
    st.error(f"Error loading model: {e}")
    st.stop()

# Streamlit app
st.title("Skin Cancer Detection")
st.write("Upload an image to predict if it is malignant or benign")

# Image upload
uploaded_file = st.file_uploader("Choose an image...", type=['jpg', 'jpeg', 'png'])

# Prediction function
def predict_skin_cancer(image, img_size=(128, 128)):
    img = Image.open(image)
    img = img.resize(img_size)
    img_array = np.array(img).astype('float32') / 255.0
    img_array = img_array.reshape(1, 128, 128, 3)
    prediction = model.predict(img_array)[0][0]
    result = 'Malignant' if prediction > 0.5 else 'Benign'
    confidence = prediction if prediction > 0.5 else 1 - prediction
    return result, confidence

# Predict
if uploaded_file is not None:
    try:
        result, confidence = predict_skin_cancer(uploaded_file)
        st.image(uploaded_file, caption='Uploaded Image', use_column_width=True)
        st.success(f"Prediction: {result} (Confidence: {confidence:.4f})")
    except Exception as e:
        st.error(f"Error making prediction: {e}")