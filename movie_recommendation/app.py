import streamlit as st
import joblib
import os

# Define paths
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIR, 'best_sentiment_model.pkl')
VECTORIZER_PATH = os.path.join(BASE_DIR, 'tfidf_vectorizer.pkl')
LABEL_ENCODER_PATH = os.path.join(BASE_DIR, 'label_encoder.pkl')
SCALER_PATH = os.path.join(BASE_DIR, 'scaler.pkl')

# Load model and preprocessors
try:
    model = joblib.load(MODEL_PATH)
    vectorizer = joblib.load(VECTORIZER_PATH)
    label_encoder = joblib.load(LABEL_ENCODER_PATH)
except FileNotFoundError as e:
    st.error(f"Error loading files: {e}")
    st.stop()

# Scaler is optional - only needed for certain model types
try:
    scaler = joblib.load(SCALER_PATH)
except FileNotFoundError:
    scaler = None

# Streamlit app
st.title("Movie Review Sentiment Analysis")
st.write("Enter a movie review to predict its sentiment")

# Input field
review = st.text_area("Enter Movie Review", height=200)

# Prediction function
def predict_sentiment(review_text):
    input_vector = vectorizer.transform([review_text])
    if scaler is not None and model.__class__.__name__ in ['LogisticRegression', 'SVC', 'KNeighborsClassifier']:
        input_vector = scaler.transform(input_vector)
    prediction = model.predict(input_vector)[0]
    sentiment = label_encoder.inverse_transform([prediction])[0]
    probabilities = model.predict_proba(input_vector)[0]
    confidence = max(probabilities)
    return sentiment, confidence
