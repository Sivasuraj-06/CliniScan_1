from ultralytics import YOLO
import torch

model = YOLO("yolov8n.pt")  


device = "mps" if torch.backends.mps.is_available() else "cpu"
print(f"Starting YOLOv8 detection training on {device}...")

results = model.train(
    data="data/chest_xray_yolo/data.yaml",
    epochs=50,       
    imgsz=640,
    batch=8,
    device=device,
    augment=True     
)


metrics = model.val()
print(f"📊 Detection mAP50: {metrics.box.map50:.4f}, mAP50-95: {metrics.box.map:.4f}")

print("✅ Detection training completed. Weights saved in 'runs/detect/train/weights/'")
