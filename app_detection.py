'''import streamlit as st
from ultralytics import YOLO
from PIL import Image

# ---------------------------
# Load YOLO Model
# ---------------------------
@st.cache_resource
def load_model():
    # Load your trained YOLO model
    model = YOLO("runs/detect/train/weights/last.pt")
    return model

model = load_model()

# Use your dataset class names
class_names = [
    "No finding", "Aortic enlargement", "Atelectasis", "Calcification", "Cardiomegaly",
    "Consolidation", "ILD", "Infiltration", "Lung Opacity", "Nodule/Mass",
    "Other lesion", "Pleural effusion", "Pleural thickening", "Pneumothorax", "Pulmonary fibrosis"
]

# ---------------------------
# Streamlit UI
# ---------------------------
st.title("🩻 Chest X-Ray Abnormality Detection (YOLO)")
st.write("Upload a Chest X-ray image and the model will detect abnormalities.")

uploaded_file = st.file_uploader("Upload an image", type=["jpg", "jpeg", "png"])

if uploaded_file:
    # Load and display image
    image = Image.open(uploaded_file).convert("RGB")
    st.image(image, caption="Uploaded Image", use_column_width=True)

    # Run YOLO inference
    results = model.predict(image, imgsz=640, conf=0.25)
    res = results[0]

    # Show annotated image
    annotated = res.plot()  # numpy array with bounding boxes
    st.image(annotated, caption="Detections", use_column_width=True)

    # Show detection details
    if len(res.boxes) > 0:
        st.write("### Detected Findings")
        det_summary = {}
        for box in res.boxes:
            cls_id = int(box.cls[0].item())
            conf = float(box.conf[0].item())
            cls_name = class_names[cls_id]
            st.write(f"- {cls_name} ({conf:.2f})")
            det_summary[cls_name] = max(det_summary.get(cls_name, 0), conf)

        # Show confidence bar chart
        st.write("### Confidence Scores")
        st.bar_chart(det_summary)
    else:
        st.success("✅ No abnormalities detected")'''

import streamlit as st
from ultralytics import YOLO
from PIL import Image

@st.cache_resource
def load_model():
    
    model = YOLO("runs/detect/train/weights/last.pt")
    return model

model = load_model()


st.title("🩻 Chest X-Ray Abnormality Detection (YOLO)")
st.write("Upload a Chest X-ray image and the model will classify as **Normal** or **Abnormal**.")

uploaded_file = st.file_uploader("Upload an image", type=["jpg", "jpeg", "png"])

if uploaded_file:
   
    image = Image.open(uploaded_file).convert("RGB")
    st.image(image, caption="Uploaded Image", use_column_width=True)

   
    results = model.predict(image, imgsz=640, conf=0.4)
    res = results[0]

    annotated = res.plot()
    st.image(annotated, caption="Detections", use_column_width=True)

   
    if len(res.boxes) > 0:
        st.error("Potential abnormal findings detected in this chest X-ray.")
    else:
        st.success("No abnormalities detected in this chest X-ray.")

