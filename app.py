import streamlit as st
import numpy as np
import os
import cv2
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import tensorflow as tf
from utils.gradcam import get_gradcam_heatmap, overlay_heatmap
import warnings
warnings.filterwarnings("ignore")
import logging
logging.getLogger("tensorflow").setLevel(logging.ERROR)
from utils.pdf_generator import generate_pdf
# PAGE CONFIG

st.set_page_config(
    page_title="Brain Tumor Detection",
    page_icon="🧠",
    layout="wide"
)

st.markdown("""
<style>
.block-container {
    padding-top: 1.5rem;
    max-width: 100%;
}

/* Glass Card */
.card {
    background: rgba(255, 255, 255, 0.06);
    padding: 22px;
    border-radius: 18px;
    backdrop-filter: blur(12px);
    border: 1px solid rgba(255,255,255,0.08);
    margin-bottom: 18px;
    box-shadow: 0 4px 20px rgba(0,0,0,0.25);
}

/* Title */
.title {
    text-align: center;
    font-size: 38px;
    font-weight: 700;
    letter-spacing: 0.5px;
}

/* Subtitle */
.subtitle {
    text-align: center;
    color: #9aa0a6;
    margin-bottom: 25px;
    font-size: 15px;
}

/* Result Cards */
.result-success {
    background: rgba(0, 200, 83, 0.15);
    padding: 15px;
    border-radius: 12px;
    text-align: center;
    font-weight: 600;
    color: #00e676;
}

.result-error {
    background: rgba(255, 82, 82, 0.15);
    padding: 15px;
    border-radius: 12px;
    text-align: center;
    font-weight: 600;
    color: #ff5252;
}

/* Upload Box */
[data-testid="stFileUploader"] {
    border: 1px dashed rgba(255,255,255,0.2);
    border-radius: 12px;
    padding: 10px;
}

/* Progress Bar */
.stProgress > div > div {
    border-radius: 10px;
}
</style>
""", unsafe_allow_html=True)


# LOAD MODEL

@st.cache_resource
def load_model():
    return tf.keras.models.load_model("model/brain_tumor_model.keras")

model = load_model()

# UI HEADER
st.markdown('<div class="title">🧠 Brain Tumor Detection</div>', unsafe_allow_html=True)
st.markdown('<div class="subtitle">AI-powered MRI Analysis with Explainability (Grad-CAM)</div>', unsafe_allow_html=True)
st.divider()

# FILE UPLOAD
col1, col2, col3 = st.columns([2.5,1,1])

with col1:
    st.markdown("### 📤 Upload MRI")
    uploaded_file = st.file_uploader(
        "Upload MRI Image",
        type=["jpg", "png", "jpeg"],
        label_visibility="collapsed"
    )

with col2:
    threshold = st.slider("Threshold", 0.0, 1.0, 0.5)

with col3:
    analyze = st.button("🔍 Analyze", width= "stretch")

# PREDICTION
if uploaded_file is not None and analyze:
    st.markdown("## 🧪 Analysis Result")
    st.caption("AI prediction and confidence based on uploaded MRI scan")
    left, right = st.columns([1, 1.2])

    # -------- LEFT: IMAGE --------
    with left:
        file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
        img = cv2.imdecode(file_bytes, 1)

        st.markdown('<div class="card">', unsafe_allow_html=True)
        st.markdown("### 🖼 Uploaded MRI")
        st.image(img, width="stretch")
        st.markdown('</div>', unsafe_allow_html=True)

    # -------- RIGHT: RESULT --------
    with right:
        with st.spinner("🧠 Analyzing MRI scan..."):

            img_resized = cv2.resize(img, (224, 224)) / 255.0
            img_reshaped = np.reshape(img_resized, (1, 224, 224, 3))

            pred = model.predict(img_reshaped)[0][0]
            confidence = pred if pred > threshold else 1 - pred
            result_text = "Tumor Detected" if pred > threshold else "No Tumor Detected"

        st.markdown('<div class="card">', unsafe_allow_html=True)

        if pred > threshold:
            st.markdown('<div class="result-error">⚠️ Tumor Detected</div>', unsafe_allow_html=True)
        else:
            st.markdown('<div class="result-success">✅ No Tumor Detected</div>', unsafe_allow_html=True)

        st.metric("Confidence Score", f"{confidence*100:.1f}%")
        st.progress(float(confidence))

        st.markdown('</div>', unsafe_allow_html=True)

    # GRAD-CAM
    st.markdown('<div class="card">', unsafe_allow_html=True)

    st.markdown("### 🔍 Model Explanation")
    st.info("🧠 Highlighted regions indicate where the model focused to make its prediction.")
    st.caption("Grad-CAM highlights regions influencing the model's decision")

    heatmap = get_gradcam_heatmap(model, img_reshaped, "Conv_1")

    temp_path = "temp.jpg"
    cv2.imwrite(temp_path, img)

    superimposed_img = overlay_heatmap(heatmap, temp_path)
    original_path = "original.jpg"
    gradcam_path = "gradcam.jpg"

    cv2.imwrite(original_path, img)
    cv2.imwrite(gradcam_path, superimposed_img)
    pdf_file = generate_pdf(result_text, confidence, original_path, gradcam_path)
    with open(pdf_file, "rb") as f:
        st.download_button(
            label="📄 Download Report",
            data=f,
            file_name="Brain_Tumor_Report.pdf",
            mime="application/pdf"
        )

    col1, col2 = st.columns(2)

    with col1:
        st.image(img, caption="Original MRI", width= "stretch")

    with col2:
        st.image(superimposed_img, caption="Grad-CAM 🔥", width="stretch")

    st.markdown('</div>', unsafe_allow_html=True)
    
# FOOTER

st.divider()
st.caption("⚠️ This tool is for educational purposes only. Consult a medical professional for diagnosis.")