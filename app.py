import streamlit as st
from detect_and_classify import detect_and_classify
from PIL import Image

st.set_page_config(page_title="Chest X-ray Detection & Classification", layout="centered")
st.title("🩺 Chest X-ray Detection + Pneumonia Classification")

uploaded_file = st.file_uploader("Upload an X-ray image", type=["jpg","jpeg","png"])

if uploaded_file:
    # Save uploaded image
    img_path = f"uploads/{uploaded_file.name}"
    with open(img_path, "wb") as f:
        f.write(uploaded_file.getbuffer())

    st.image(Image.open(img_path), caption="Uploaded Image", use_column_width=True)

    # Run YOLO + ResNet
    st.info("Running detection and classification...")
    annotated_path = detect_and_classify(img_path, save_path="results")

    st.success("✅ Done!")
    st.image(Image.open(annotated_path), caption="Annotated Image", use_column_width=True)
