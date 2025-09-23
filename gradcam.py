import os
import torch
import torch.nn as nn
from torchvision import models, transforms
from torch.utils.data import DataLoader
from torchvision.datasets import ImageFolder

from pytorch_grad_cam import GradCAM
from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget
from pytorch_grad_cam.utils.image import show_cam_on_image
import cv2
import numpy as np


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model = models.resnet18(pretrained=False)
num_features = model.fc.in_features
model.fc = nn.Linear(num_features, 2)  
model.load_state_dict(torch.load("best_classification_model.pth", map_location=device))
model = model.to(device)
model.eval()

transform = transforms.Compose([
    transforms.Resize((224,224)),
    transforms.ToTensor(),
])

val_dataset = ImageFolder("cls_dataset_binary/val", transform=transform)
val_loader = DataLoader(val_dataset, batch_size=1, shuffle=True)

target_layers = [model.layer4[-1]] 
cam = GradCAM(model=model, target_layers=target_layers)

os.makedirs("gradcam_examples/normal", exist_ok=True)
os.makedirs("gradcam_examples/abnormal", exist_ok=True)

saved_per_class = {"normal": 0, "abnormal": 0}
max_per_class = 5

for img_tensor, label in val_loader:
    if all(v >= max_per_class for v in saved_per_class.values()):
        break
    
    input_tensor = img_tensor.to(device)
    target_class = label.item()
    class_name = val_dataset.classes[target_class]
    
    if saved_per_class[class_name] >= max_per_class:
        continue

    img_np = input_tensor[0].cpu().permute(1,2,0).numpy()
    img_np = (img_np - img_np.min()) / (img_np.max() - img_np.min())
    
    targets = [ClassifierOutputTarget(target_class)]
    grayscale_cam = cam(input_tensor=input_tensor, targets=targets)[0]
    cam_image = show_cam_on_image(img_np, grayscale_cam, use_rgb=True)
    
    save_path = f"gradcam_examples/{class_name}/{saved_per_class[class_name]}.png"
    cv2.imwrite(save_path, cv2.cvtColor(cam_image, cv2.COLOR_RGB2BGR))
    saved_per_class[class_name] += 1
