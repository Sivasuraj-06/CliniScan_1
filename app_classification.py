import streamlit as st
import torch
import torch.nn as nn
from torchvision import transforms, models
from PIL import Image
import numpy as np
import cv2

@st.cache_resource
def load_model():
    model = models.resnet18(pretrained=False)
    model.fc = nn.Linear(model.fc.in_features, 2)  # 2 classes: abnormal, normal
    model.load_state_dict(torch.load("best_classification_model.pth", map_location="cpu"))
    model.eval()
    return model

model = load_model()


transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], 
                         [0.229, 0.224, 0.225])
])


def generate_gradcam(model, input_tensor, target_class):
    """
    Returns a Grad-CAM heatmap for a single image tensor
    """
    gradients = []
    activations = []

    def backward_hook(module, grad_in, grad_out):
        gradients.append(grad_out[0])

    def forward_hook(module, input, output):
        activations.append(output)

    layer = model.layer4[1].conv2
    hook_handle_forward = layer.register_forward_hook(forward_hook)
    hook_handle_backward = layer.register_backward_hook(backward_hook)

    output = model(input_tensor)
    model.zero_grad()
    loss = output[0, target_class]
    loss.backward()

    grads = gradients[0].detach().numpy()[0]
    acts = activations[0].detach().numpy()[0]

    weights = np.mean(grads, axis=(1, 2))
    cam = np.zeros(acts.shape[1:], dtype=np.float32)
    for i, w in enumerate(weights):
        cam += w * acts[i, :, :]
    cam = np.maximum(cam, 0)
    cam = cv2.resize(cam, (224, 224))
    cam = (cam - cam.min()) / (cam.max() - cam.min() + 1e-8)  

    hook_handle_forward.remove()
    hook_handle_backward.remove()

    return cam

def overlay_cam_on_image(img, cam):
    
    img = np.array(img.resize((224, 224)))
    heatmap = cv2.applyColorMap(np.uint8(255 * cam), cv2.COLORMAP_JET)
    heatmap = cv2.cvtColor(heatmap, cv2.COLOR_BGR2RGB)
    overlay = cv2.addWeighted(img, 0.6, heatmap, 0.4, 0)
    return overlay


st.title("🩻 Chest X-Ray Classification with Grad-CAM")
st.write("Upload a Chest X-ray image to classify it as **Normal** or **Abnormal** and visualize Grad-CAM.")

uploaded_file = st.file_uploader("Upload an image", type=["jpg", "jpeg", "png"])

if uploaded_file:
    
    image = Image.open(uploaded_file).convert("RGB")
    st.image(image, caption="Uploaded Image", use_column_width=True)

    
    input_tensor = transform(image).unsqueeze(0)

    
    with torch.no_grad():
        outputs = model(input_tensor)
        probs = torch.softmax(outputs, dim=1)[0]
        pred_class = torch.argmax(probs).item()

    class_names = ["abnormal", "normal"]
    st.write(f"### Prediction: **{class_names[pred_class]}**")
    st.write(f"Confidence: {probs[pred_class].item() * 100:.2f}%")

    
    st.bar_chart({class_names[i]: probs[i].item() for i in range(len(class_names))})

    
    cam = generate_gradcam(model, input_tensor, pred_class)
    overlay = overlay_cam_on_image(image, cam)

    st.image(overlay, caption="Grad-CAM Heatmap Overlay", use_column_width=True)
