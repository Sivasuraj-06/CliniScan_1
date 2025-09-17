import streamlit as st
import torch
import torch.nn as nn
from torchvision import transforms, models
from PIL import Image

# ---------------------------
# Load Model
# ---------------------------
@st.cache_resource
def load_model():
    model = models.resnet18(pretrained=False)
    model.fc = nn.Linear(model.fc.in_features, 2)  # Example: 2 classes
    model.load_state_dict(torch.load("models/model.pth", map_location="cpu"))
  # Load trained weights
    model.eval()
    return model

model = load_model()

# ---------------------------
# Image Transformations
# ---------------------------
transform = transforms.Compose([
    transforms.Resize((224, 224)),  
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], 
                         [0.229, 0.224, 0.225])
])

# ---------------------------
# Streamlit UI
# ---------------------------
st.title("🩻 Chest X-Ray Classification")
st.write("Upload a Chest X-ray image to classify it as **Normal** or **Pneumonia**.")

uploaded_file = st.file_uploader("Upload an image", type=["jpg", "jpeg", "png"])

if uploaded_file:
    # Load and display image
    image = Image.open(uploaded_file).convert("RGB")
    st.image(image, caption="Uploaded Image", use_column_width=True)

    # Preprocess image
    input_tensor = transform(image).unsqueeze(0)

    # Inference
    with torch.no_grad():
        outputs = model(input_tensor)
        probs = torch.softmax(outputs, dim=1)[0]
        pred_class = torch.argmax(probs).item()

    # Class labels
    class_names = ["Normal", "Pneumonia"]

    # Display results
    st.write(f"### Prediction: **{class_names[pred_class]}**")
    st.write(f"Confidence: {probs[pred_class].item() * 100:.2f}%")

    # Confidence bar chart
    st.bar_chart({class_names[i]: probs[i].item() for i in range(len(class_names))})
