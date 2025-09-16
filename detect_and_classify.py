import torch
from ultralytics import YOLO
from torchvision import models, transforms
from PIL import Image
import cv2
import os
import numpy as np

# ---------------------------
# Paths to trained models
# ---------------------------
YOLO_WEIGHTS = "runs/detect/train/weights/best.pt"   # your YOLOv8 best weights
RESNET_WEIGHTS = "models/resnet18_baseline.pth"     # your trained ResNet weights

# ---------------------------
# Load YOLO detection model
# ---------------------------
yolo_model = YOLO(YOLO_WEIGHTS)
device = "mps" if torch.backends.mps.is_available() else "cpu"

# ---------------------------
# Load ResNet classification model
# ---------------------------
resnet = models.resnet18(weights=None)
resnet.fc = torch.nn.Linear(resnet.fc.in_features, 2)  # 2 classes: NORMAL, PNEUMONIA
resnet.load_state_dict(torch.load(RESNET_WEIGHTS, map_location=device))
resnet.eval()
resnet.to(device)

# ---------------------------
# Preprocessing for ResNet
# ---------------------------
preprocess = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406],
                         [0.229, 0.224, 0.225])
])

classes = ["NORMAL", "PNEUMONIA"]

# ---------------------------
# Detection + Classification function
# ---------------------------
def detect_and_classify(image_path, save_path="results"):
    os.makedirs(save_path, exist_ok=True)
    
    # Load image
    img = cv2.imread(image_path)
    orig_img = img.copy()
    
    # YOLO detection
    results = yolo_model.predict(source=image_path, conf=0.5, save=False)
    result = results[0]
    boxes = result.boxes.xyxy.cpu().numpy()  # x1, y1, x2, y2

    for i, box in enumerate(boxes):
        x1, y1, x2, y2 = map(int, box)
        crop = Image.fromarray(orig_img[y1:y2, x1:x2])

        # ResNet classification
        input_tensor = preprocess(crop).unsqueeze(0).to(device)
        with torch.no_grad():
            pred_class = classes[torch.argmax(resnet(input_tensor), dim=1).item()]
        
        # Draw box and label
        cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.putText(img, pred_class, (x1, y1-10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)
    
    # Save annotated image
    save_file = os.path.join(save_path, os.path.basename(image_path))
    cv2.imwrite(save_file, img)
    return save_file
