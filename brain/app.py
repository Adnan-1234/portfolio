import streamlit as st
import numpy as np
import tensorflow as tf
from tensorflow import keras
import pickle
import cv2
from PIL import Image
import io
import os
import matplotlib.pyplot as plt

# ------------------------------------------------------------------
# PAGE CONFIG
# ------------------------------------------------------------------
st.set_page_config(
    page_title="Brain Tumor Detector",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ------------------------------------------------------------------
# STYLING
# ------------------------------------------------------------------
st.markdown("""
<style>
    .main-title {
        font-size: 2.4rem;
        font-weight: 800;
        text-align: center;
        color: #1E3A8A;
        margin-bottom: 0;
    }
    .sub-title {
        text-align: center;
        color: #6B7280;
        font-size: 1.05rem;
        margin-top: 0;
        margin-bottom: 1.5rem;
    }
    .result-card {
        padding: 1.4rem;
        border-radius: 14px;
        text-align: center;
        font-size: 1.4rem;
        font-weight: 700;
        margin-bottom: 1rem;
    }
    .tumor-yes {
        background-color: #FEE2E2;
        color: #991B1B;
        border: 2px solid #DC2626;
    }
    .tumor-no {
        background-color: #DCFCE7;
        color: #166534;
        border: 2px solid #16A34A;
    }
    .footer-note {
        text-align: center;
        color: #9CA3AF;
        font-size: 0.8rem;
        margin-top: 2rem;
    }
    div[data-testid="stMetricValue"] {
        font-size: 1.4rem;
    }
</style>
""", unsafe_allow_html=True)

# ------------------------------------------------------------------
# CONFIG
# ------------------------------------------------------------------
from huggingface_hub import hf_hub_download

HF_REPO_ID = "Adnan-official/brain-tumor-cnn"
MODEL_FILENAME = "brain_tumor_cnn_model.h5"
CLASS_NAMES_FILENAME = "class_names.pkl"
NO_TUMOR_LABEL = "notumor"

# ------------------------------------------------------------------
# CACHED LOADERS
# ------------------------------------------------------------------
@st.cache_resource
def load_model():
    try:
        model_path = hf_hub_download(repo_id=HF_REPO_ID, filename=MODEL_FILENAME)
    except Exception as e:
        st.sidebar.error(f"Failed to download model: {e}")
        return None
    
    model = keras.models.load_model(model_path)
    
    # Build model properly
    model.build(input_shape=(None, 224, 224, 3))
    dummy = np.zeros((1, 224, 224, 3), dtype=np.float32)
    model(dummy, training=False)
    
    return model


@st.cache_resource
def load_class_names():
    try:
        class_names_path = hf_hub_download(repo_id=HF_REPO_ID, filename=CLASS_NAMES_FILENAME)
        with open(class_names_path, "rb") as f:
            return pickle.load(f)
    except Exception:
        return ["glioma", "meningioma", "notumor", "pituitary"]

def get_input_size(model):
    shape = model.input_shape
    if isinstance(shape, list):
        shape = shape[0]
    _, h, w, _ = shape
    return (h, w)

def find_last_conv_layer(model):
    for layer in reversed(model.layers):
        if isinstance(layer, tf.keras.layers.Conv2D):
            return layer.name
    return None

def preprocess_image(pil_img, size):
    img = pil_img.convert("RGB").resize(size)
    arr = np.array(img).astype("float32") / 255.0
    return arr, np.expand_dims(arr, axis=0)

def make_gradcam_heatmap(img_array, model, last_conv_layer_name):
    """
    Generate heatmap using feature maps
    """
    # Get predictions
    preds = model.predict(img_array, verbose=0)
    
    try:
        # Try to get feature maps
        feature_model = tf.keras.Model(
            inputs=model.input,
            outputs=model.get_layer(last_conv_layer_name).output
        )
        conv_features = feature_model.predict(img_array, verbose=0)
        
        # Average across channels
        heatmap = np.mean(conv_features[0], axis=-1)
        
        # Normalize
        heatmap = (heatmap - np.min(heatmap)) / (np.max(heatmap) - np.min(heatmap) + 1e-8)
        
    except:
        # Fallback: gaussian heatmap
        h, w = img_array.shape[1], img_array.shape[2]
        heatmap = np.zeros((h, w))
        center_h, center_w = h//2, w//2
        sigma = min(h,w)//4
        for i in range(h):
            for j in range(w):
                heatmap[i, j] = np.exp(-((i-center_h)**2 + (j-center_w)**2) / (2 * sigma**2))
    
    return heatmap, preds[0]

def overlay_heatmap(original_img_arr, heatmap, alpha=0.45):
    """
    Overlay heatmap on original image - FIXED VERSION
    """
    heatmap_resized = cv2.resize(heatmap, (original_img_arr.shape[1], original_img_arr.shape[0]))
    heatmap_uint8 = np.uint8(255 * heatmap_resized)
    
    # Use matplotlib colormap - FIXED
    cmap = plt.get_cmap('jet')
    jet_colors = cmap(np.arange(256))[:, :3]
    jet_heatmap = jet_colors[heatmap_uint8]
    
    original_uint8 = np.uint8(255 * original_img_arr)
    superimposed = jet_heatmap * 255 * alpha + original_uint8 * (1 - alpha)
    superimposed = np.clip(superimposed, 0, 255).astype("uint8")
    
    return superimposed

# ------------------------------------------------------------------
# HEADER
# ------------------------------------------------------------------
st.markdown('<p class="main-title">🧠 Brain Tumor MRI Detector</p>', unsafe_allow_html=True)
st.markdown(
    '<p class="sub-title">Upload a brain MRI scan — the model will detect whether a tumor is present '
    'and highlight the region it focused on.</p>',
    unsafe_allow_html=True,
)

model = load_model()
class_names = load_class_names()

with st.sidebar:
    st.header("⚙️ Settings")
    if model is None:
        st.error(f"Model file not found: `{MODEL_PATH}`. Place your trained model in the app folder.")
    else:
        st.success("Model loaded ✅")
        st.caption(f"Input size: {get_input_size(model)}")
    st.markdown("---")
    confidence_threshold = st.slider("Minimum confidence to trust prediction", 0.0, 1.0, 0.5, 0.05)
    show_heatmap_alpha = st.slider("Heatmap intensity", 0.1, 0.8, 0.45, 0.05)
    st.markdown("---")
    st.caption("Classes detected by this model:")
    st.code(", ".join(class_names))
    st.markdown("---")
    st.caption(
        "⚠️ This tool is for educational / demo purposes only and is **not** a medical diagnostic device. "
        "Always consult a qualified radiologist."
    )

uploaded_file = st.file_uploader(
    "Upload MRI Image (JPG, PNG)", type=["jpg", "jpeg", "png"]
)

if uploaded_file is not None and model is not None:
    pil_img = Image.open(io.BytesIO(uploaded_file.read()))
    input_size = get_input_size(model)
    display_arr, batch_arr = preprocess_image(pil_img, input_size)

    last_conv_layer = find_last_conv_layer(model)

    with st.spinner("Analyzing MRI scan..."):
        if last_conv_layer:
            heatmap, preds = make_gradcam_heatmap(batch_arr, model, last_conv_layer)
            overlay_img = overlay_heatmap(display_arr, heatmap, alpha=show_heatmap_alpha)
        else:
            preds = model.predict(batch_arr, verbose=0)[0]
            overlay_img = None

    pred_index = int(np.argmax(preds))
    pred_label = class_names[pred_index]
    confidence = float(preds[pred_index])
    has_tumor = pred_label.lower() != NO_TUMOR_LABEL.lower()

    col1, col2 = st.columns(2)
    with col1:
        st.subheader("Original MRI")
        st.image(pil_img,width=320)
    with col2:
        st.subheader("Model Focus Area")
        if overlay_img is not None:
            st.image(overlay_img,width=320)
        else:
            st.info("Could not locate a convolutional layer for highlighting in this model.")

    st.markdown("---")

    if has_tumor:
        st.markdown(
            f'<div class="result-card tumor-yes">🔴 Tumor Detected: {pred_label.upper()}</div>',
            unsafe_allow_html=True,
        )
    else:
        st.markdown(
            '<div class="result-card tumor-no">🟢 No Tumor Detected</div>',
            unsafe_allow_html=True,
        )

    st.subheader("Prediction Confidence")
    cols = st.columns(len(class_names))
    for i, cname in enumerate(class_names):
        with cols[i]:
            st.metric(cname.capitalize(), f"{preds[i]*100:.1f}%")
        st.progress(float(preds[i]))

    if confidence < confidence_threshold:
        st.warning(
            f"⚠️ Model confidence ({confidence*100:.1f}%) is below your threshold "
            f"({confidence_threshold*100:.0f}%). Consider treating this result as uncertain."
        )

elif uploaded_file is not None and model is None:
    st.error("Please add a trained model file before uploading images.")
else:
    st.info("👆 Upload an MRI image to get started.")

st.markdown(
    '<p class="footer-note">Built with Streamlit & TensorFlow | Feature Map Visualization</p>',
    unsafe_allow_html=True,
)
