"""
🐕 Dog Skin Disease Classifier
Professional Streamlit App powered by a custom CNN model hosted on Hugging Face Hub.
Model repo: Adnan-official/DOG_SKIN_DISEASE
"""

import os
import pickle
import numpy as np
import streamlit as st
from PIL import Image
import tensorflow as tf
from huggingface_hub import hf_hub_download

# ============================================================
# PAGE CONFIG
# ============================================================
st.set_page_config(
    page_title="Dog Skin Disease Classifier",
    page_icon="🐕",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ============================================================
# CUSTOM CSS — PROFESSIONAL LOOK
# ============================================================
st.markdown("""
<style>
    .main {
        background-color: #0e1117;
    }
    .stApp {
        background: linear-gradient(180deg, #f7f9fc 0%, #eef2f7 100%);
    }
    .app-header {
        text-align: center;
        padding: 1.6rem 1rem 1.2rem 1rem;
        background: linear-gradient(135deg, #2b6777 0%, #52ab98 100%);
        border-radius: 18px;
        color: white;
        margin-bottom: 1.6rem;
        box-shadow: 0 8px 24px rgba(0,0,0,0.12);
    }
    .app-header h1 {
        font-size: 2.3rem;
        margin-bottom: 0.3rem;
        font-weight: 700;
    }
    .app-header p {
        font-size: 1.02rem;
        opacity: 0.92;
        margin: 0;
    }
    .result-card {
        background: white;
        border-radius: 16px;
        padding: 1.5rem;
        box-shadow: 0 4px 18px rgba(0,0,0,0.08);
        border-left: 6px solid #52ab98;
        margin-top: 1rem;
    }
    .result-card h2 {
        color: #2b6777;
        margin-bottom: 0.2rem;
    }
    .confidence-badge {
        display: inline-block;
        padding: 0.3rem 0.9rem;
        border-radius: 20px;
        background: #52ab98;
        color: white;
        font-weight: 600;
        font-size: 0.95rem;
    }
    .stButton>button {
        background: linear-gradient(135deg, #2b6777 0%, #52ab98 100%);
        color: white;
        border: none;
        border-radius: 10px;
        padding: 0.6rem 1.8rem;
        font-weight: 600;
        font-size: 1rem;
        transition: all 0.2s ease-in-out;
    }
    .stButton>button:hover {
        transform: translateY(-2px);
        box-shadow: 0 6px 14px rgba(43,103,119,0.35);
    }
    footer {visibility: hidden;}
    .footer-note {
        text-align: center;
        color: #888;
        font-size: 0.85rem;
        margin-top: 2.5rem;
        padding-top: 1rem;
        border-top: 1px solid #ddd;
    }
    .disclaimer-box {
        background: #fff8e6;
        border: 1px solid #ffd966;
        border-radius: 10px;
        padding: 0.8rem 1rem;
        font-size: 0.9rem;
        color: #6b5600;
        margin-top: 1rem;
    }
</style>
""", unsafe_allow_html=True)

# ============================================================
# CONFIG
# ============================================================
HF_REPO_ID = "Adnan-official/DOG_SKIN_DISEASE"
MODEL_FILENAME = "best_dog_cnn_model.h5"
CLASSES_FILENAME = "dog_class_names.pkl"
IMG_SIZE = (224, 224)

# Optional: short info about common classes (edit / extend freely).
# If a predicted class isn't listed here, a generic message is shown instead.
DISEASE_INFO = {
    "healthy": "No visible signs of skin disease detected. Skin appears normal.",
    "hotspot": "A hotspot (acute moist dermatitis) is a red, inflamed, often oozing patch caused by self-trauma from licking, biting, or scratching.",
    "fungal": "Possible fungal infection (e.g. ringworm). Typically appears as circular, hairless, scaly patches.",
    "mange": "Mange is caused by mites and often leads to hair loss, intense itching, and crusty skin.",
    "dermatitis": "General skin inflammation which can be triggered by allergies, irritants, or infections.",
    "flea": "Flea-related skin irritation, often causing redness, itching, and small scabs.",
}

# ============================================================
# LOAD MODEL & CLASS NAMES (cached)
# ============================================================
@st.cache_resource(show_spinner=False)
def load_model_and_classes():
    model_path = hf_hub_download(repo_id=HF_REPO_ID, filename=MODEL_FILENAME)
    classes_path = hf_hub_download(repo_id=HF_REPO_ID, filename=CLASSES_FILENAME)

    model = tf.keras.models.load_model(model_path)

    with open(classes_path, "rb") as f:
        class_names = pickle.load(f)

    return model, class_names


def preprocess_image(image: Image.Image):
    image = image.convert("RGB").resize(IMG_SIZE)
    arr = np.array(image).astype("float32") / 255.0
    arr = np.expand_dims(arr, axis=0)
    return arr


def get_disease_info(label: str) -> str:
    key = label.lower().strip()
    for k, v in DISEASE_INFO.items():
        if k in key:
            return v
    return "No additional information available for this class. Please consult a veterinarian for a proper diagnosis."


# ============================================================
# HEADER
# ============================================================
st.markdown("""
<div class="app-header">
    <h1>🐕 Dog Skin Disease Classifier</h1>
    <p>Upload a photo of your dog's skin — an AI-powered CNN model will analyze it instantly.</p>
</div>
""", unsafe_allow_html=True)

# ============================================================
# SIDEBAR
# ============================================================
with st.sidebar:
    st.markdown("## ℹ️ About This App")
    st.write(
        "This tool uses a **custom-trained Convolutional Neural Network (CNN)** "
        "to classify dog skin conditions from an uploaded image."
    )
    st.markdown("---")
    st.markdown("### 🧠 Model Details")
    st.write(f"**Hosted on:** Hugging Face Hub")
    st.write(f"**Repo:** `{HF_REPO_ID}`")
    st.write(f"**Input size:** {IMG_SIZE[0]}x{IMG_SIZE[1]} px")
    st.markdown("---")
    st.markdown("### 📝 How to Use")
    st.write(
        "1. Upload a clear image of the affected skin area\n"
        "2. Click **Analyze Image**\n"
        "3. View predicted condition & confidence"
    )
    st.markdown("---")
    st.caption("⚠️ This app is for educational/informational purposes only and is **not** a substitute for professional veterinary diagnosis.")

# ============================================================
# LOAD MODEL
# ============================================================
with st.spinner("🔄 Loading model from Hugging Face Hub... (first load may take a minute)"):
    try:
        model, class_names = load_model_and_classes()
        model_loaded = True
    except Exception as e:
        model_loaded = False
        st.error(f"❌ Failed to load model from Hugging Face Hub.\n\nError: {e}")

# ============================================================
# MAIN LAYOUT
# ============================================================
if model_loaded:
    col1, col2 = st.columns([1, 1], gap="large")

    with col1:
        st.markdown("### 📤 Upload Image")
        uploaded_file = st.file_uploader(
            "Choose a dog skin image (JPG, PNG)",
            type=["jpg", "jpeg", "png"],
        )

        if uploaded_file is not None:
            image = Image.open(uploaded_file)
            st.image(image, caption="Uploaded Image", use_container_width=True)
            analyze_btn = st.button("🔍 Analyze Image", use_container_width=True)
        else:
            analyze_btn = False
            st.info("👆 Please upload an image to get started.")

    with col2:
        st.markdown("### 📊 Prediction Result")

        if uploaded_file is not None and analyze_btn:
            with st.spinner("🧪 Analyzing image..."):
                processed = preprocess_image(image)
                predictions = model.predict(processed, verbose=0)[0]
                top_idx = int(np.argmax(predictions))
                top_label = class_names[top_idx]
                top_confidence = float(predictions[top_idx]) * 100

            st.markdown(f"""
            <div class="result-card">
                <h2>{top_label.replace('_', ' ').title()}</h2>
                <span class="confidence-badge">Confidence: {top_confidence:.2f}%</span>
                <p style="margin-top:1rem; color:#444;">{get_disease_info(top_label)}</p>
            </div>
            """, unsafe_allow_html=True)

            st.markdown("#### 🔢 Class Probabilities")
            prob_dict = {
                class_names[i].replace('_', ' ').title(): float(predictions[i]) * 100
                for i in range(len(class_names))
            }
            sorted_probs = dict(sorted(prob_dict.items(), key=lambda x: x[1], reverse=True))
            st.bar_chart(sorted_probs)

            st.markdown("""
            <div class="disclaimer-box">
                ⚠️ <b>Disclaimer:</b> This prediction is generated by an AI model and should not replace
                professional veterinary advice. Please consult a licensed veterinarian for an accurate diagnosis.
            </div>
            """, unsafe_allow_html=True)

        elif uploaded_file is not None and not analyze_btn:
            st.info("Click **Analyze Image** to run the prediction.")
        else:
            st.write("Prediction results will appear here once you upload and analyze an image.")

# ============================================================
# FOOTER
# ============================================================
st.markdown("""
<div class="footer-note">
    Built with ❤️ using TensorFlow, Streamlit & Hugging Face Hub<br>
    Model repository: Adnan-official/DOG_SKIN_DISEASE
</div>
""", unsafe_allow_html=True)
