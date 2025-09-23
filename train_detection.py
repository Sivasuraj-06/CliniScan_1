import os
import glob
from ultralytics import YOLO

# ----------------------------
# 1️⃣ Settings
# ----------------------------
OUTPUT_DIR = "yolo_dataset"
YAML_PATH = os.path.join(OUTPUT_DIR, "dataset.yaml")
IMG_SIZE = 320
BATCH_SIZE = 2
EPOCHS = 30
CONF_THRESH = 0.25

# ----------------------------
# 2️⃣ Load pretrained YOLOv8 model
# ----------------------------
print("🔹 Loading YOLOv8n model...")
model = YOLO("yolov8n.pt")  # pretrained on COCO
print("✅ Model loaded!")

# ----------------------------
# 3️⃣ Train YOLOv8 with augmentations and tuned hyperparameters
# ----------------------------
print("🔹 Starting training...")
model.train(
    data=YAML_PATH,
    epochs=EPOCHS,
    imgsz=IMG_SIZE,
    batch=BATCH_SIZE,
    workers=0,
    lr0=0.001,
    lrf=0.1,
    momentum=0.937,
    weight_decay=0.0005,
    augment=True  # advanced augmentation: mosaic, mixup, HSV
)
print("✅ Training complete!")

# ----------------------------
# 4️⃣ Validation & metrics
# ----------------------------
print("🔹 Running validation...")
metrics = model.val(data=YAML_PATH)
print("✅ Validation complete!")
print("📊 Metrics:", metrics)

# ----------------------------
# 5️⃣ Inference and bounding box visualization
# ----------------------------
print("🔹 Running inference on validation images...")
val_images = glob.glob(os.path.join(OUTPUT_DIR, "images/val/*.png"))
os.makedirs(os.path.join(OUTPUT_DIR, "runs/val_preds"), exist_ok=True)

for i, img_path in enumerate(val_images, start=1):
    results = model.predict(
        img_path,
        imgsz=IMG_SIZE,
        conf=CONF_THRESH,
        save=True,
        save_dir=os.path.join(OUTPUT_DIR, "runs/val_preds")
    )
    print(f"[{i}/{len(val_images)}] ✅ Predictions saved for {os.path.basename(img_path)}")

print("🎯 All detection steps finished successfully!")
