import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image
import time

# --- CONFIGURATION ---
MODEL_PATH = 'tomato_disease_model.h5'
IMG_SIZE = 128

# --- PAGE SETUP ---
st.set_page_config(
    page_title="AgriMind AI",
    page_icon="🍅",
    layout="centered"
)

# --- SIDEBAR (Professional Touch) ---
st.sidebar.title("🌱 AgriMind Dashboard")
st.sidebar.write("### Project Info:")
st.sidebar.info(
    """
    **Project:** AI-Based Crop Disease Detection
    **Model:** Convolutional Neural Network (CNN)
    **Accuracy:** >90% (Achieved on Test Set)
    **Developer: Smacky's Bro
    """
)

# Initialize Session State for History (Memory during the session)
if 'history' not in st.session_state:
    st.session_state.history = []

st.sidebar.write("### Recent Scans:")
if st.session_state.history:
    for item in st.session_state.history:
        st.sidebar.write(f"🔸 {item}")
else:
    st.sidebar.write("No scans yet.")

# --- LOAD MODEL ---
@st.cache_resource
def load_model():
    model = tf.keras.models.load_model(MODEL_PATH)
    return model

model = load_model()

# --- FUNCTIONS ---
def preprocess_image(image):
    image = image.resize((IMG_SIZE, IMG_SIZE))
    image = np.array(image) / 255.0
    image = np.expand_dims(image, axis=0)
    return image

# --- MAIN UI ---
st.title("🍅 AgriMind: Advanced Disease Detector")
st.markdown("---") # Horizontal line

uploaded_file = st.file_uploader("Upload a Tomato Leaf Image", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    col1, col2 = st.columns(2)

    with col1:
        image = Image.open(uploaded_file)
        st.image(image, caption='Uploaded Image', use_column_width=True)

    with col2:
        st.write("### Analysis Report")
        with st.spinner('AI is thinking...'):
            time.sleep(1) # Simulate processing time for effect
            processed_image = preprocess_image(image)
            prediction = model.predict(processed_image)
            confidence = prediction[0][0]

            # LOGIC: 
            # 0 = Early Blight, 1 = Healthy
            
            if confidence < 0.5:
                result = "Early Blight 🚨"
                color = "red"
                conf_score = (1 - confidence) * 100
                remedy = "Apply Copper Fungicide immediately. Remove affected leaves."
                log_entry = "Early Blight"
            else:
                result = "Healthy ✅"
                color = "green"
                conf_score = confidence * 100
                remedy = "Plant is healthy. Continue regular monitoring."
                log_entry = "Healthy"

            # Display Result
            st.markdown(f"<h2 style='color:{color};text-align: center;'>{result}</h2>", unsafe_allow_html=True)
            st.metric("Confidence Score", f"{conf_score:.2f}%")
            
            st.info(f"**Recommendation:** {remedy}")

            # Add to History
            st.session_state.history.insert(0, log_entry) # Add to top of list
            # Keep only last 5 scans
            if len(st.session_state.history) > 5:
                st.session_state.history.pop()

else:
    st.write("👈 Upload an image to start the diagnosis.")

st.markdown("---")
st.caption("Powered by TensorFlow & Streamlit | Mini Project Semester 6")