import os
import shutil
import random
import glob
import cv2
import pandas as pd
from sklearn.model_selection import train_test_split
from ultralytics import YOLO


PNG_DIR = "png_train"
CSV_FILE = "vinbigdata-chest-xray-abnormalities-detection/train.csv"
OUTPUT_DIR = "yolo_dataset"
VAL_RATIO = 0.2
IMG_SIZE = 320     
BATCH_SIZE = 2     
EPOCHS =30
CONF_THRESH = 0.25

random.seed(42)
os.makedirs(OUTPUT_DIR, exist_ok=True)
for sub in ["images/train", "images/val", "labels/train", "labels/val"]:
    os.makedirs(os.path.join(OUTPUT_DIR, sub), exist_ok=True)


print("🔹 Generating YOLO labels...")
df = pd.read_csv(CSV_FILE)

LABEL_DIR = "labels_yolo"
os.makedirs(LABEL_DIR, exist_ok=True)

for image_id, group in df.groupby("image_id"):
    img_path = os.path.join(PNG_DIR, f"{image_id}.png")
    if not os.path.exists(img_path):
        continue
    img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
    h, w = img.shape[:2]

    label_path = os.path.join(LABEL_DIR, f"{image_id}.txt")
    with open(label_path, "w") as f:
        for _, row in group.iterrows():
            if row["class_name"] == "No finding":
                continue  
            x_min, y_min, x_max, y_max = row[["x_min", "y_min", "x_max", "y_max"]]
            x_center = (x_min + x_max) / 2.0 / w
            y_center = (y_min + y_max) / 2.0 / h
            width = (x_max - x_min) / w
            height = (y_max - y_min) / h
            class_id = 1  
            f.write(f"{class_id} {x_center} {y_center} {width} {height}\n")
print("✅ YOLO labels generated!")


print("🔹 Balancing dataset and splitting into train/val...")

normal_ids = []
abnormal_ids = []
for img_file in os.listdir(PNG_DIR):
    if not img_file.endswith(".png"):
        continue
    img_id = os.path.splitext(img_file)[0]
    lbl_path = os.path.join(LABEL_DIR, f"{img_id}.txt")
    if os.path.exists(lbl_path) and os.path.getsize(lbl_path) > 0:
        abnormal_ids.append(img_id)
    else:
        normal_ids.append(img_id)

if len(normal_ids) > len(abnormal_ids):
    normal_ids = random.sample(normal_ids, len(abnormal_ids))

all_ids = normal_ids + abnormal_ids
stratify_labels = [1 if i in abnormal_ids else 0 for i in all_ids]
train_ids, val_ids = train_test_split(all_ids, test_size=VAL_RATIO, random_state=42, stratify=stratify_labels)

def copy_files(ids, split):
    for img_id in ids:
        img_src = os.path.join(PNG_DIR, f"{img_id}.png")
        lbl_src = os.path.join(LABEL_DIR, f"{img_id}.txt")
        if os.path.exists(img_src):
            shutil.copy(img_src, os.path.join(OUTPUT_DIR, f"images/{split}", f"{img_id}.png"))
        lbl_dst = os.path.join(OUTPUT_DIR, f"labels/{split}", f"{img_id}.txt")
        if os.path.exists(lbl_src):
            shutil.copy(lbl_src, lbl_dst)
        else:
            open(lbl_dst, "w").close()  

copy_files(train_ids, "train")
copy_files(val_ids, "val")
print(f"✅ Balanced dataset split: {len(train_ids)} train, {len(val_ids)} val")

yaml_path = os.path.join(OUTPUT_DIR, "dataset.yaml")
with open(yaml_path, "w") as f:
    f.write(f"train: {os.path.abspath(os.path.join(OUTPUT_DIR,'images/train'))}\n")
    f.write(f"val: {os.path.abspath(os.path.join(OUTPUT_DIR,'images/val'))}\n\n")
    f.write("nc: 2\n")
    f.write('names: ["Normal", "Abnormal"]\n')
print("✅ dataset.yaml created with 2 classes")