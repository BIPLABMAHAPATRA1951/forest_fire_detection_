import streamlit as st
from ultralytics import YOLO
from PIL import Image
import numpy as np
import cv2
import os
import tempfile

# Page config
st.set_page_config(
    page_title="Forest Fire Detection",
    page_icon="🔥",
    layout="wide"
)

# Model load karo
@st.cache_resource
def load_model():
    model_path = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                              "runs", "detect", "runs", "forest_fire6", "weights", "best.pt")
    return YOLO(model_path)

model = load_model()

# UI
st.title("🔥 Forest Fire Detection System")
st.markdown("Upload a forest image to detect fire and smoke!")

# Sidebar
st.sidebar.title("⚙️ Settings")
confidence = st.sidebar.slider("Confidence Threshold", 0.1, 0.9, 0.25, 0.05)
st.sidebar.markdown("---")
st.sidebar.markdown("### Classes:")
st.sidebar.markdown("🔴 Large Fire")
st.sidebar.markdown("🟠 Medium Fire")
st.sidebar.markdown("🟡 Small Fire")
st.sidebar.markdown("⚫ Heavy Smoke")
st.sidebar.markdown("🔵 Low Smoke")

# Upload
uploaded_file = st.file_uploader(
    "Choose an image...",
    type=['jpg', 'jpeg', 'png']
)

if uploaded_file is not None:
    image = Image.open(uploaded_file)

    col1, col2 = st.columns(2)

    with col1:
        st.subheader("📷 Original Image")
        st.image(image, use_column_width=True)

    # Prediction
    with st.spinner("🔍 Analyzing image..."):
        with tempfile.NamedTemporaryFile(delete=False, suffix='.jpg') as tmp:
            tmp_path = tmp.name
            image.save(tmp_path)

        results = model.predict(
            source=tmp_path,
            conf=confidence,
            save=False
        )

        try:
            os.unlink(tmp_path)
        except:
            pass

    # Result image
    result_img = results[0].plot()
    result_img_rgb = cv2.cvtColor(result_img, cv2.COLOR_BGR2RGB)

    with col2:
        st.subheader("🔥 Detection Result")
        st.image(result_img_rgb, use_column_width=True)

    # Detection details
    st.markdown("---")
    boxes = results[0].boxes

    if len(boxes) == 0:
        st.success("✅ No fire or smoke detected! Forest looks safe.")
    else:
        class_names = [results[0].names[int(b.cls[0])] for b in boxes]
        if any('fire' in c.lower() for c in class_names):
            st.error("🚨 FIRE DETECTED! Immediate action required!")
        else:
            st.warning("⚠️ SMOKE DETECTED! Monitor the situation.")

        st.subheader(f"📊 Detected {len(boxes)} Object(s):")
        cols = st.columns(3)
        cols[0].markdown("**Class**")
        cols[1].markdown("**Confidence**")
        cols[2].markdown("**Risk Level**")

        for box in boxes:
            cls_name = results[0].names[int(box.cls[0])]
            conf_val = float(box.conf[0])

            if 'Large fire' in cls_name:
                risk = "🔴 CRITICAL"
            elif 'Medium fire' in cls_name:
                risk = "🟠 HIGH"
            elif 'Small fire' in cls_name:
                risk = "🟡 MEDIUM"
            elif 'Heavy smoke' in cls_name:
                risk = "⚫ HIGH"
            else:
                risk = "🔵 LOW"

            c1, c2, c3 = st.columns(3)
            c1.write(cls_name)
            c2.write(f"{conf_val:.2%}")
            c3.write(risk)

else:
    st.info("👆 Upload an image to get started!")
    st.markdown("""
    ### How it works:
    1. 📤 Upload a forest image
    2. 🤖 AI model analyzes it
    3. 🔥 Fire & smoke detected with bounding boxes
    4. 📊 Risk level shown
    """)