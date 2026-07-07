import streamlit as st
import pickle
import re
import os

# Define paths
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIR, 'model.pkl')
VECTORIZER_PATH = os.path.join(BASE_DIR, 'vectorizer.pkl')
ACCURACY_PATH = os.path.join(BASE_DIR, 'accuracy.pkl')

# Clean resume function
def clean_resume(resume_text):
    resume_text = re.sub(r'http[s]?://\S+', '', resume_text)
    resume_text = re.sub(r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,4}\b', '', resume_text)
    resume_text = re.sub(r'\(?\+?[0-9]*\)?[0-9_\- \(\)]*', '', resume_text)
    resume_text = re.sub(r'[^\w\s]', '', resume_text)
    resume_text = re.sub(r'\s+', ' ', resume_text).strip()
    return resume_text

# Load model, vectorizer, and accuracy
try:
    model = pickle.load(open(MODEL_PATH, 'rb'))
    tfidf = pickle.load(open(VECTORIZER_PATH, 'rb'))
    accuracy = pickle.load(open(ACCURACY_PATH, 'rb'))
except FileNotFoundError as e:
    st.error(f"Error loading files: {e}")
    st.stop()

# Category mapping
mapping = {
    0: 'Advocate', 1: 'Arts', 2: 'Automation Testing', 3: 'Blockchain', 4: 'Business Analyst',
    5: 'Civil Engineer', 6: 'Data Science', 7: 'Database', 8: 'DevOps Engineer', 9: 'DotNet Developer',
    10: 'ETL Developer', 11: 'Electrical Engineering', 12: 'HR', 13: 'Hadoop', 14: 'Health and fitness',
    15: 'Java Developer', 16: 'Mechanical Engineer', 17: 'Network Security Engineer', 18: 'Operations Manager',
    19: 'PMO', 20: 'Python Developer', 21: 'SAP Developer', 22: 'Sales', 23: 'Testing', 24: 'Web Designing'
}

# Streamlit app
st.title("Resume Screening App")
st.write("Predict job category from resume text")

resume_input = st.text_area("Enter Resume Text", height=200)
if st.button("Predict"):
    if resume_input:
        try:
            cleaned_resume = clean_resume(resume_input)
            input_feature = tfidf.transform([cleaned_resume])
            predicted = model.predict(input_feature)[0]
            category_name = mapping.get(predicted, 'Unknown')
            st.success(f"Predicted Job Category: {category_name}")
            st.write(f"Model Accuracy: {accuracy*100:.2f}%")
        except Exception as e:
            st.error(f"Error making prediction: {e}")
    else:
        st.error("Please enter resume text")