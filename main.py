import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image
from streamlit_extras.colored_header import colored_header
from streamlit_extras.stylable_container import stylable_container

# Page Config
st.set_page_config(page_title="Mask Detection", page_icon="😷", layout="centered")

# Load Model
@st.cache_resource
def load_model():
    model = tf.keras.models.load_model(r"E:\NTI\Mask_withoutmask_proj\myModel (1).keras")
    return model

model = load_model()

# Custom CSS
st.markdown("""
    <style>
        .stApp {
             background: linear-gradient(135deg, #001f3f, #003366);
        }
        .result-card {
            padding: 20px;
            border-radius: 20px;
            text-align: center;
            font-size: 22px;
            font-weight: bold;
            margin-top: 20px;
        }
        .mask {
            background-color: red;
            color: #2e7d32;
            box-shadow: 0px 4px 10px rgba(46, 125, 50, 0.4);
        }
        .no-mask {
            background-color:green;
            color: black;
            box-shadow: 0px 4px 10px rgba(183, 28, 28, 0.4);
        }
    </style>
""", unsafe_allow_html=True)

# Header
colored_header(
    label="😷 Mask Detection App",
    description="Upload an image and check if a mask is detected!",
    color_name="blue-70",
)

# File uploader
uploaded_file = st.file_uploader("📤 Upload your image", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    # Show uploaded image
    img = Image.open(uploaded_file).convert("RGB")
    st.image(img, caption="🖼️ Uploaded Image", use_container_width=True)

    # Preprocess image
    img_resized = img.resize((224, 224))
    img_array = np.array(img_resized)
    img_array = np.expand_dims(img_array, axis=0)
    img_array = img_array / 255.0

    # Prediction
    prediction = model.predict(img_array)
    pred_class = 0 if prediction[0] < 0.5 else 1

    # Show result inside card
    if pred_class == 1:
        st.markdown('<div class="result-card mask">✅ Mask Detected</div>', unsafe_allow_html=True)
    else:
        st.markdown('<div class="result-card no-mask">❌ No Mask Detected</div>', unsafe_allow_html=True)
